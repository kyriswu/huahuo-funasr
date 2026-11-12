import threading

from fastapi.testclient import TestClient
import numpy as np
import pytest

from app.audio import AudioError
from app.main import create_app
from app.settings import Settings


class FakeEngine:
    model_id = "test-cam++"

    def __init__(self, settings):
        self.entered = threading.Event()
        self.unblock = threading.Event()
        self.block = False
        self.counts = []

    def embedding(self, clip):
        if clip.duration_seconds < 3:
            raise AudioError("Need at least 3 seconds of detected speech")
        result = np.zeros(192, dtype=np.float32)
        result[0] = 1
        return result, clip.duration_seconds

    def transcribe(self, clip, speaker_count, records, threshold):
        self.counts.append(speaker_count)
        if self.block:
            self.entered.set()
            assert self.unblock.wait(10)
        return {"text": "测试", "segments": [], "speakers": []}


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(Settings(data_dir=tmp_path), FakeEngine)) as instance:
        yield instance


def test_probe_and_enrollment_lifecycle(client, audio_bytes):
    assert client.get("/readyz").status_code == 200
    registered = client.post("/v1/voiceprints/alice", files={"file": ("audio.wav", audio_bytes)}, data={"name": "Alice"})
    assert registered.status_code == 201
    assert "embedding" not in registered.json()
    assert client.post("/v1/voiceprints/alice", files={"file": ("audio.wav", audio_bytes)}).status_code == 409
    gallery = client.get("/v1/voiceprints").json()["voiceprints"]
    assert len(gallery) == 1 and "embedding" not in gallery[0]
    identified = client.post("/v1/voiceprints/identify", files={"file": ("audio.wav", audio_bytes)})
    assert identified.status_code == 200 and identified.json()["speaker_id"] == "alice"
    verified = client.post("/v1/voiceprints/verify", files={"reference": ("a.wav", audio_bytes), "query": ("b.wav", audio_bytes)})
    assert verified.status_code == 200 and verified.json()["same_speaker"]
    assert client.delete("/v1/voiceprints/alice").status_code == 200
    assert client.delete("/v1/voiceprints/alice").status_code == 404


def test_token_rejected_before_audio_parsing(tmp_path):
    with TestClient(create_app(Settings(data_dir=tmp_path, api_token="a-secret"), FakeEngine)) as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/v1/voiceprints").status_code == 401
        assert client.get("/v1/voiceprints", headers={"Authorization": "Bearer a-secret"}).status_code == 200
        assert client.post("/v1/transcriptions", content=b"not multipart").status_code == 401


def test_input_errors_and_unknown_gallery(client, audio_bytes):
    assert client.post("/v1/transcriptions", files={"file": ("bad.wav", b"invalid")}).status_code == 422
    assert client.post("/v1/transcriptions", files={"file": ("a.wav", audio_bytes)}, data={"speaker_count": 0}).status_code == 422
    assert client.post("/v1/voiceprints/bad.id", files={"file": ("a.wav", audio_bytes)}).status_code == 422
    unknown = client.post("/v1/voiceprints/identify", files={"file": ("a.wav", audio_bytes)}).json()
    assert not unknown["matched"] and unknown["speaker_id"] is None


def test_requests_do_not_reuse_previous_speaker_count(client, audio_bytes):
    assert client.post("/v1/transcriptions", files={"file": ("a.wav", audio_bytes)}, data={"speaker_count": 2}).status_code == 200
    assert client.post("/v1/transcriptions", files={"file": ("a.wav", audio_bytes)}).status_code == 200
    assert client.app.state.engine.counts == [2, None]


def test_inference_busy_does_not_block_health(client, audio_bytes):
    engine = client.app.state.engine
    engine.block = True
    responses = []
    thread = threading.Thread(target=lambda: responses.append(client.post("/v1/transcriptions", files={"file": ("a.wav", audio_bytes)})))
    thread.start()
    assert engine.entered.wait(5)
    try:
        busy = client.post("/v1/transcriptions", files={"file": ("a.wav", audio_bytes)})
        assert busy.status_code == 429 and busy.headers["Retry-After"] == "2"
        assert client.get("/healthz").status_code == 200
    finally:
        engine.unblock.set()
        thread.join(10)
    assert responses[0].status_code == 200


def test_streamed_request_limit(tmp_path):
    settings = Settings(data_dir=tmp_path, max_upload_mb=1)
    prefix = b'--boundary\r\nContent-Disposition: form-data; name="file"; filename="huge.wav"\r\nContent-Type: audio/wav\r\n\r\n'

    def chunks():
        yield prefix
        for _ in range(5):
            yield b"x" * (1024 * 1024)
        yield b"\r\n--boundary--\r\n"

    with TestClient(create_app(settings, FakeEngine)) as client:
        response = client.post("/v1/transcriptions", content=chunks(), headers={"Content-Type": "multipart/form-data; boundary=boundary"})
        assert response.status_code == 413
