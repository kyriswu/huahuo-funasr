"""Bounded, process-local jobs. Only an explicit ACK or callback 2xx consumes a result."""

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
import random
import tempfile
import threading
import time
from typing import BinaryIO
import uuid

from app.audio import AudioError, AudioTooLarge, decode_audio
from app.callbacks import CallbackTarget, send_callback
from app.settings import Settings
from app.voiceprints import StoreError

logger = logging.getLogger(__name__)


class TaskError(Exception):
    def __init__(self, status: int, detail: str):
        self.status, self.detail = status, detail
        super().__init__(detail)


def _timestamp(value: float | None) -> str | None:
    return datetime.fromtimestamp(value, timezone.utc).isoformat() if value is not None else None


@dataclass
class Task:
    task_id: str
    event_id: str
    options: dict
    callback: CallbackTarget | None
    created: float
    audio: BinaryIO | None = None
    status: str = "uploading"
    started: float | None = None
    finished: float | None = None
    consumed: float | None = None
    consumed_by: str | None = None
    audio_bytes: int = 0
    body: bytes | None = None
    callback_status: str = "not_requested"
    callback_attempts: int = 0
    callback_next: float = 0.0
    callback_http_status: int | None = None
    fingerprint: str | None = None
    idempotency_key: str | None = None


class TaskManager:
    def __init__(self, settings: Settings, engine, store, inference_lock, metrics, sender=None):
        self.settings, self.engine, self.store = settings, engine, store
        self.inference_lock = inference_lock
        self.metrics = metrics
        self.sender = sender or send_callback
        self._condition = threading.Condition()
        self._tasks: dict[str, Task] = {}
        self._keys: dict[str, str] = {}
        self._queue = deque()
        self._history = deque()
        self._active = 0
        self._audio_bytes = 0
        self._stopping = False
        self._inference_thread = threading.Thread(target=self._inference_loop, name="funasr-inference", daemon=True)
        self._callback_thread = threading.Thread(target=self._callback_loop, name="funasr-callback", daemon=True)

    def start(self):
        self._inference_thread.start()
        self._callback_thread.start()

    def close(self):
        with self._condition:
            self._stopping = True
            self._condition.notify_all()
        self._inference_thread.join()
        self._callback_thread.join()
        with self._condition:
            for task in self._tasks.values():
                if task.status == "queued":
                    self._delete_audio(task)
            self._tasks.clear()
            self._keys.clear()
            self._queue.clear()
            self._history.clear()

    def has_pending_inference(self) -> bool:
        with self._condition:
            return any(task.status in {"uploading", "queued", "running"} for task in self._tasks.values())

    def capacity_usage(self) -> tuple[int, int]:
        with self._condition:
            return self._active, self._audio_bytes

    def _prune_locked(self):
        now = time.time()
        while self._history:
            task_id = self._history[0]
            task = self._tasks[task_id]
            if len(self._history) <= self.settings.task_history_limit and now - task.consumed < self.settings.task_status_ttl_seconds:
                break
            self._history.popleft()
            del self._tasks[task_id]
            if task.idempotency_key:
                self._keys.pop(task.idempotency_key, None)

    def _get_locked(self, task_id: str) -> Task:
        self._prune_locked()
        task = self._tasks.get(task_id)
        if task is None:
            raise TaskError(404, "Task not found; tasks are not restored after a service restart")
        return task

    @staticmethod
    def _public(task: Task) -> dict:
        return {
            "task_id": task.task_id,
            "status": task.status,
            "created_at": _timestamp(task.created),
            "started_at": _timestamp(task.started),
            "finished_at": _timestamp(task.finished),
            "result_available": task.body is not None,
            "consumed_at": _timestamp(task.consumed),
            "consumed_by": task.consumed_by,
            "callback": {
                "status": task.callback_status,
                "attempts": task.callback_attempts,
                "last_http_status": task.callback_http_status,
            },
        }

    def status(self, task_id: str) -> dict:
        with self._condition:
            return self._public(self._get_locked(task_id))

    def result(self, task_id: str) -> bytes:
        with self._condition:
            task = self._get_locked(task_id)
            if task.consumed is not None:
                raise TaskError(410, "Result has been consumed and deleted")
            if task.body is None:
                raise TaskError(409, "Task has not finished")
            return task.body

    def _consume_locked(self, task: Task, by: str):
        if task.consumed is not None:
            return
        task.body = None
        task.options = {}
        task.consumed, task.consumed_by = time.time(), by
        if task.callback is not None:
            task.callback_status = "delivered" if by == "callback" else "cancelled"
        task.callback = None
        self._active -= 1
        self._history.append(task.task_id)
        self._prune_locked()
        self._condition.notify_all()

    def ack(self, task_id: str, event_id: str) -> dict:
        with self._condition:
            task = self._get_locked(task_id)
            if event_id != task.event_id:
                raise TaskError(409, "event_id does not match this task")
            if task.finished is None:
                raise TaskError(409, "Task has not finished")
            self._consume_locked(task, "ack")
            return {"task_id": task_id, "event_id": event_id, "acknowledged": True, "consumed_by": task.consumed_by}

    def submit(self, source: BinaryIO, options: dict, callback: CallbackTarget | None, idempotency_key: str | None) -> dict:
        if idempotency_key is not None and (
            not 1 <= len(idempotency_key) <= 128
            or not idempotency_key.isascii()
            or any(ord(char) <= 32 or ord(char) == 127 for char in idempotency_key)
        ):
            raise TaskError(422, "Idempotency-Key must contain 1-128 printable ASCII characters without spaces")
        task_id = "task_" + uuid.uuid4().hex
        task = Task(task_id, "evt_" + uuid.uuid4().hex, options, callback, time.time())
        if callback is not None:
            task.callback_status = "pending"
        with self._condition:
            self._prune_locked()
            # A repeated key may return its existing task even when the queue is full.
            duplicate = idempotency_key is not None and idempotency_key in self._keys
            if self._stopping or (self._active >= self.settings.max_pending_tasks and not duplicate):
                raise TaskError(429, "Task capacity is full; consume pending results before submitting more")
            if not duplicate:
                self._active += 1
                self._tasks[task_id] = task
        digest = hashlib.sha256()
        try:
            source.seek(0)
            # An anonymous temporary file is reclaimed when its descriptors close, even on a crash.
            target = task.audio = None if duplicate else tempfile.TemporaryFile(prefix="funasr-task-")
            encoded_bytes = 0
            while chunk := source.read(1024 * 1024):
                encoded_bytes += len(chunk)
                if encoded_bytes > self.settings.max_upload_mb * 1024 * 1024:
                    raise AudioTooLarge("Encoded audio exceeds MAX_UPLOAD_MB")
                if target is not None:
                    with self._condition:
                        if self._audio_bytes + len(chunk) > self.settings.max_task_audio_mb * 1024 * 1024:
                            raise TaskError(429, "Temporary task audio capacity is full")
                        self._audio_bytes += len(chunk)
                        task.audio_bytes += len(chunk)
                    target.write(chunk)
                digest.update(chunk)
            if target is not None:
                target.flush()
            if encoded_bytes == 0:
                raise AudioError("Audio upload is empty")
            descriptor = {
                "audio_sha256": digest.hexdigest(), "options": options,
                "callback": vars(callback) if callback is not None else None,
            }
            fingerprint = hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest()
            with self._condition:
                if self._stopping:
                    raise TaskError(503, "Service is stopping")
                existing_id = self._keys.get(idempotency_key) if idempotency_key is not None else None
                if existing_id is not None:
                    existing = self._tasks[existing_id]
                    if existing.fingerprint != fingerprint:
                        raise TaskError(409, "Idempotency-Key was already used with different audio or options")
                    return self._public(existing)
                if duplicate:
                    raise TaskError(409, "Previous task was removed; submit again with a new Idempotency-Key")
                task.idempotency_key, task.fingerprint = idempotency_key, fingerprint
                if idempotency_key is not None:
                    self._keys[idempotency_key] = task_id
                task.status = "queued"
                self._queue.append(task_id)
                self._condition.notify_all()
                return self._public(task)
        except OSError as error:
            logger.warning("Temporary audio write failed: %s", type(error).__name__)
            raise TaskError(503, "Temporary audio storage is unavailable or full") from error
        finally:
            # A duplicate/error never enters the inference queue and must release its reservation.
            if task.status == "uploading":
                self._delete_audio(task)
                if not duplicate:
                    with self._condition:
                        self._tasks.pop(task_id, None)
                        self._active -= 1

    def _delete_audio(self, task: Task):
        if task.audio is not None:
            try:
                task.audio.close()
            except OSError as error:
                logger.warning("Temporary audio close failed: %s", type(error).__name__)
            finally:
                task.audio = None
        with self._condition:
            self._audio_bytes -= task.audio_bytes
            task.audio_bytes = 0

    def _run(self, task: Task) -> tuple[dict | None, dict | None]:
        try:
            try:
                assert task.audio is not None
                with self.metrics.stage.labels("async", "decode").time():
                    clip = decode_audio(task.audio, self.settings, self.settings.max_audio_seconds)
            finally:
                self._delete_audio(task)
            records = self.store.records() if task.options["identify_speakers"] else None
            with self.metrics.stage.labels("async", "inference").time():
                result = self.engine.transcribe(clip, task.options["speaker_count"], records, task.options["threshold"])
            return result, None
        except AudioTooLarge as error:
            return None, {"code": "audio_too_large", "message": str(error)}
        except AudioError as error:
            return None, {"code": "invalid_audio", "message": str(error)}
        except StoreError:
            return None, {"code": "voiceprint_store_unavailable", "message": "Voiceprint store is corrupt or incompatible"}
        except Exception as error:
            logger.error("Async inference failed: %s", type(error).__name__)
            if type(error).__name__ == "OutOfMemoryError":
                self.metrics.gpu_oom.inc()
                return None, {"code": "gpu_memory_exhausted", "message": "Reduce BATCH_SIZE_S or audio length"}
            return None, {"code": "processing_failed", "message": "Processing failed; check service logs"}

    def _finish(self, task: Task, result: dict | None, error: dict | None):
        status = "failed" if error is not None else "succeeded"
        event = {
            "event_id": task.event_id, "event": f"task.{status}", "task_id": task.task_id,
            "status": status, "result": result, "error": error,
        }
        try:
            body = json.dumps(event, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
            if len(body) > self.settings.max_task_result_mb * 1024 * 1024:
                raise ValueError("Result limit")
        except (ValueError, TypeError):
            status = "failed"
            event.update(status=status, event="task.failed", result=None,
                         error={"code": "result_unavailable", "message": "Result is invalid or exceeds MAX_TASK_RESULT_MB"})
            body = json.dumps(event, separators=(",", ":")).encode()
        with self._condition:
            self.metrics.completed.labels(status).inc()
            task.status, task.body, task.finished = status, body, time.time()
            task.callback_next = time.monotonic()
            self._condition.notify_all()

    def _inference_loop(self):
        while True:
            with self._condition:
                while not self._queue and not self._stopping:
                    self._condition.wait()
                if self._stopping:
                    return
                task = self._tasks[self._queue.popleft()]
            while not self.inference_lock.acquire(timeout=0.1):
                if self._stopping:
                    return
            try:
                if self._stopping:
                    return
                with self._condition:
                    task.status, task.started = "running", time.time()
                result, error = self._run(task)
            finally:
                self.inference_lock.release()
            self._finish(task, result, error)
            result = error = None

    def _next_callback_locked(self):
        while not self._stopping:
            candidates = [
                task for task in self._tasks.values()
                if task.body is not None and task.callback is not None
                and task.callback_status in {"pending", "retrying"}
            ]
            if candidates:
                task = min(candidates, key=lambda item: item.callback_next)
                delay = task.callback_next - time.monotonic()
                if delay <= 0:
                    task.callback_attempts += 1
                    task.callback_status = "sending"
                    return task, task.body, task.callback
                self._condition.wait(timeout=min(delay, 30))
            else:
                self._condition.wait()
        return None

    def _callback_loop(self):
        while True:
            with self._condition:
                delivery = self._next_callback_locked()
            if delivery is None:
                return
            task, body, target = delivery
            delivery = None
            try:
                code = self.sender(target, body, task.event_id, self.settings.callback_secret, self.settings.callback_timeout_seconds)
            except Exception as error:
                logger.warning("Callback attempt failed: %s", type(error).__name__)
                code = None
            body = None
            delivered = code is not None and 200 <= code < 300
            self.metrics.callback_attempts.labels("delivered" if delivered else "failed").inc()
            with self._condition:
                if task.consumed is not None:
                    continue
                task.callback_http_status = code
                if delivered:
                    self._consume_locked(task, "callback")
                elif task.callback_attempts >= self.settings.callback_max_attempts:
                    task.callback_status = "failed"
                else:
                    task.callback_status = "retrying"
                    delay = min(3600, self.settings.callback_retry_seconds * 2 ** min(task.callback_attempts - 1, 16))
                    task.callback_next = time.monotonic() + delay * random.uniform(0.8, 1.2)
