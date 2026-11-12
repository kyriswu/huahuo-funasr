from contextlib import asynccontextmanager, contextmanager
import logging
import secrets
import threading
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import numpy as np
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.audio import AudioError, AudioTooLarge, decode_audio
from app.engine import SpeechEngine
from app.settings import Settings
from app.voiceprints import StoreError, VoiceprintExists, VoiceprintStore, match_voiceprint, validate_id

logger = logging.getLogger(__name__)
bearer = HTTPBearer(auto_error=False)


class RequestGuard:
    """Authenticate before multipart parsing; also bound streamed/chunked bodies."""

    def __init__(self, app, settings: Settings):
        self.app, self.settings = app, settings

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not scope["path"].startswith("/v1/"):
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers", []))
        if self.settings.api_token:
            expected = ("Bearer " + self.settings.api_token).encode()
            if not secrets.compare_digest(headers.get(b"authorization", b""), expected):
                return await JSONResponse(
                    {"detail": "Invalid or missing Bearer token"}, 401,
                    headers={"WWW-Authenticate": "Bearer"},
                )(scope, receive, send)
        # Two files for verification plus multipart framing.
        body_limit = 2 * self.settings.max_upload_mb * 1024 * 1024 + 1024 * 1024
        if b"content-length" in headers:
            try:
                size = int(headers[b"content-length"])
                if size < 0:
                    raise ValueError
            except ValueError:
                return await JSONResponse({"detail": "Invalid Content-Length"}, 400)(scope, receive, send)
            if size > body_limit:
                return await JSONResponse({"detail": "Request body exceeds upload limit"}, 413)(scope, receive, send)
        received = 0

        async def bounded_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > body_limit:
                    raise StarletteHTTPException(413, "Request body exceeds upload limit")
            return message

        await self.app(scope, bounded_receive, send)


def create_app(settings: Settings | None = None, engine_factory=SpeechEngine) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.ready = False
        engine = await run_in_threadpool(engine_factory, settings)
        application.state.engine = engine
        application.state.store = VoiceprintStore(settings.data_dir / "voiceprints", engine.model_id)
        application.state.ready = True
        yield
        application.state.ready = False

    application = FastAPI(
        title="FunASR ECI 离线语音服务", version="1.0.0", lifespan=lifespan,
        description="GPU 离线转写、说话人时间段标注、声纹注册与比对。所有时间戳单位为毫秒；相似度为余弦分数。",
    )
    application.state.ready = False
    application.state.inference_lock = threading.Lock()
    application.add_middleware(RequestGuard, settings=settings)

    @application.exception_handler(AudioTooLarge)
    async def audio_too_large(request, error):
        return JSONResponse({"detail": str(error)}, 413)

    @application.exception_handler(AudioError)
    async def invalid_audio(request, error):
        return JSONResponse({"detail": str(error)}, 422)

    @application.exception_handler(VoiceprintExists)
    async def already_registered(request, error):
        return JSONResponse({"detail": str(error)}, 409)

    @application.exception_handler(StoreError)
    async def invalid_store(request, error):
        logger.error("Voiceprint store is unavailable: %s", error)
        return JSONResponse({"detail": "Voiceprint store is corrupt or incompatible; check service logs"}, 503)

    @application.exception_handler(Exception)
    async def failed_inference(request, error):
        # Avoid logging exception values, which may contain input text or other audio data.
        logger.error("Request failed: %s", type(error).__name__)
        if type(error).__name__ == "OutOfMemoryError":
            return JSONResponse({"detail": "GPU memory exhausted; reduce BATCH_SIZE_S or audio length"}, 503)
        return JSONResponse({"detail": "Processing failed; check service logs"}, 500)

    def check_token(credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)]):
        # Middleware performs early authentication; this dependency also declares OpenAPI security.
        if settings.api_token and (credentials is None or not secrets.compare_digest(credentials.credentials.encode(), settings.api_token.encode())):
            raise HTTPException(401, "Invalid or missing Bearer token")

    @contextmanager
    def inference_slot(request: Request):
        if not request.app.state.ready:
            raise HTTPException(503, "Models are not ready")
        lock = request.app.state.inference_lock
        if not lock.acquire(blocking=False):
            raise HTTPException(429, "Inference is busy; retry later", headers={"Retry-After": "2"})
        try:
            yield request.app.state.engine
        finally:
            lock.release()

    def selected_threshold(value):
        return settings.voiceprint_threshold if value is None else value

    def ensure_valid_id(speaker_id):
        try:
            return validate_id(speaker_id)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    @application.get("/healthz", include_in_schema=False)
    async def health():
        return {"status": "ok"}

    @application.get("/readyz", include_in_schema=False)
    async def ready(request: Request):
        if not request.app.state.ready:
            raise HTTPException(503, "Models are not ready")
        return {"status": "ready", "device": settings.device}

    @application.post("/v1/transcriptions", dependencies=[Depends(check_token)])
    def transcribe(
        request: Request,
        file: Annotated[UploadFile, File(description="待转写的完整音频")],
        speaker_count: Annotated[int | None, Form(ge=1, le=30)] = None,
        identify_speakers: Annotated[bool, Form()] = False,
        threshold: Annotated[float | None, Form(ge=-1, le=1)] = None,
    ):
        with inference_slot(request) as engine:
            clip = decode_audio(file.file, settings, settings.max_audio_seconds)
            records = request.app.state.store.records() if identify_speakers else None
            return engine.transcribe(clip, speaker_count, records, selected_threshold(threshold))

    @application.get("/v1/voiceprints", dependencies=[Depends(check_token)])
    def list_voiceprints(request: Request):
        return {"voiceprints": [VoiceprintStore.public(item) for item in request.app.state.store.records()]}

    # Static routes precede /{speaker_id} so identify/verify cannot be mistaken for IDs.
    @application.post("/v1/voiceprints/identify", dependencies=[Depends(check_token)])
    def identify(
        request: Request, file: Annotated[UploadFile, File()],
        threshold: Annotated[float | None, Form(ge=-1, le=1)] = None,
        top_k: Annotated[int, Form(ge=1, le=20)] = 5,
    ):
        with inference_slot(request) as engine:
            clip = decode_audio(file.file, settings, settings.max_voiceprint_seconds)
            embedding, seconds = engine.embedding(clip)
            result = match_voiceprint(embedding, request.app.state.store.records(), selected_threshold(threshold), top_k)
            return {**result, "speech_seconds": round(seconds, 3), "model_id": engine.model_id}

    @application.post("/v1/voiceprints/verify", dependencies=[Depends(check_token)])
    def verify(
        request: Request,
        reference: Annotated[UploadFile, File(description="已知单人参考音频")],
        query: Annotated[UploadFile, File(description="待核验单人音频")],
        threshold: Annotated[float | None, Form(ge=-1, le=1)] = None,
    ):
        with inference_slot(request) as engine:
            first, first_seconds = engine.embedding(decode_audio(reference.file, settings, settings.max_voiceprint_seconds))
            second, second_seconds = engine.embedding(decode_audio(query.file, settings, settings.max_voiceprint_seconds))
            score = float(np.clip(np.dot(first, second), -1, 1))
            value = selected_threshold(threshold)
            return {"same_speaker": score >= value, "score": round(score, 6), "threshold": value,
                    "reference_speech_seconds": round(first_seconds, 3), "query_speech_seconds": round(second_seconds, 3)}

    @application.post("/v1/voiceprints/{speaker_id}", dependencies=[Depends(check_token)], status_code=201)
    def enroll(
        request: Request, speaker_id: str, file: Annotated[UploadFile, File(description="干净单人音频，建议 10-30 秒")],
        name: Annotated[str, Form(max_length=128)] = "",
        replace: Annotated[bool, Form()] = False,
    ):
        ensure_valid_id(speaker_id)
        if speaker_id in {"identify", "verify"}:
            raise HTTPException(422, "speaker_id is reserved")
        with inference_slot(request) as engine:
            clip = decode_audio(file.file, settings, settings.max_voiceprint_seconds)
            embedding, seconds = engine.embedding(clip)
            return request.app.state.store.put(speaker_id, name, embedding, seconds, replace)

    @application.delete("/v1/voiceprints/{speaker_id}", dependencies=[Depends(check_token)])
    def remove(request: Request, speaker_id: str):
        ensure_valid_id(speaker_id)
        try:
            request.app.state.store.delete(speaker_id)
        except KeyError:
            raise HTTPException(404, "Voiceprint not found") from None
        return {"deleted": speaker_id}

    return application


app = create_app()
