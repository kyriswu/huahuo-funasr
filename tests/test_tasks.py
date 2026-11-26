import json
import threading
import time

from fastapi.testclient import TestClient
import pytest

from app.audio import decode_audio
from app.main import create_app
from app.settings import Settings
from conftest import wav_bytes
from test_api import FakeEngine


def wait_status(client, task_id, predicate=lambda data: data["status"] in {"succeeded", "failed"}):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        response = client.get(f"/v1/tasks/{task_id}")
        assert response.status_code == 200
        status = response.json()
        if predicate(status):
            return status
        time.sleep(0.01)
    pytest.fail(f"Task did not reach expected state: {status}")


def submit(client, audio, **data):
    response = client.post("/v1/tasks", files={"file": ("input.wav", audio)}, data=data)
    assert response.status_code == 202, response.text
    assert response.headers["cache-control"] == "no-store"
    return response.json()["task_id"]


def acknowledge(client, task_id, event):
    return client.post(f"/v1/tasks/{task_id}/ack", json={"event_id": event["event_id"]})


def test_status_result_ack_and_audio_lifecycle(tmp_path, audio_bytes, monkeypatch):
    sources = []

    def decode_and_record(source, settings, max_seconds):
        sources.append(source)
        return decode_audio(source, settings, max_seconds)

    monkeypatch.setattr("app.tasks.decode_audio", decode_and_record)
    with TestClient(create_app(Settings(data_dir=tmp_path, max_pending_tasks=1), FakeEngine)) as client:
        engine, manager = client.app.state.engine, client.app.state.tasks
        engine.block = True
        task_id = submit(client, audio_bytes, speaker_count=2)
        assert engine.entered.wait(5)
        try:
            staged, = sources
            assert staged.closed and manager._tasks[task_id].audio is None
            assert manager.capacity_usage() == (1, 0)
            status = client.get(f"/v1/tasks/{task_id}").json()
            assert status["status"] == "running" and "result" not in status
            assert client.get(f"/v1/tasks/{task_id}/result").status_code == 409
            premature = client.post(f"/v1/tasks/{task_id}/ack", json={"event_id": manager._tasks[task_id].event_id})
            assert premature.status_code == 409
            assert client.post("/v1/transcriptions", files={"file": ("a.wav", audio_bytes)}).status_code == 429
            assert client.get("/healthz").status_code == 200
        finally:
            engine.unblock.set()
        wait_status(client, task_id)
        first = client.get(f"/v1/tasks/{task_id}/result")
        second = client.get(f"/v1/tasks/{task_id}/result")
        assert first.status_code == 200 and first.content == second.content
        event = first.json()
        assert event["result"]["text"] == "测试" and event["event"] == "task.succeeded"
        assert engine.counts == [2]
        assert client.get(f"/v1/tasks/{task_id}").json()["result_available"]
        wrong = client.post(f"/v1/tasks/{task_id}/ack", json={"event_id": "evt_" + "0" * 32})
        assert wrong.status_code == 409
        assert acknowledge(client, task_id, event).json()["acknowledged"]
        assert acknowledge(client, task_id, event).status_code == 200
        assert client.get(f"/v1/tasks/{task_id}/result").status_code == 410
        assert not client.get(f"/v1/tasks/{task_id}").json()["result_available"]
        assert manager._tasks[task_id].body is None


def test_unconsumed_results_never_expire_and_hold_capacity(tmp_path, audio_bytes):
    with TestClient(create_app(Settings(data_dir=tmp_path, max_pending_tasks=1), FakeEngine)) as client:
        task_id = submit(client, audio_bytes)
        wait_status(client, task_id)
        manager = client.app.state.tasks
        manager._tasks[task_id].created -= 86400
        manager._tasks[task_id].finished -= 86400
        for _ in range(2):
            assert client.get(f"/v1/tasks/{task_id}").json()["result_available"]
            assert client.get(f"/v1/tasks/{task_id}/result").status_code == 200
        rejected = client.post("/v1/tasks", files={"file": ("a.wav", audio_bytes)})
        assert rejected.status_code == 429 and rejected.headers["retry-after"] == "2"
        event = client.get(f"/v1/tasks/{task_id}/result").json()
        acknowledge(client, task_id, event)
        assert submit(client, audio_bytes) != task_id


def test_idempotent_retry_works_with_audio_capacity_reserved(tmp_path):
    settings = Settings(data_dir=tmp_path, max_pending_tasks=1, max_task_audio_mb=1, max_upload_mb=1)
    audio = wav_bytes(seconds=20)
    headers = {"Idempotency-Key": "unsure-202"}
    with TestClient(create_app(settings, FakeEngine)) as client:
        manager = client.app.state.tasks
        with manager.inference_lock:
            original = client.post("/v1/tasks", files={"file": ("audio.wav", audio)}, headers=headers)
            assert original.status_code == 202 and manager.capacity_usage() == (1, len(audio))
            duplicate = client.post("/v1/tasks", files={"file": ("audio.wav", audio)}, headers=headers)
            assert duplicate.status_code == 202 and duplicate.json()["task_id"] == original.json()["task_id"]
            conflict = client.post("/v1/tasks", files={"file": ("audio.wav", audio)}, headers=headers,
                                   data={"speaker_count": 2})
            assert conflict.status_code == 409
            oversized = client.post("/v1/tasks", files={"file": ("audio.wav", b"x" * (1024 * 1024 + 1))}, headers=headers)
            assert oversized.status_code == 413
            assert manager.capacity_usage() == (1, len(audio))


def test_failed_audio_can_be_delivered_and_acknowledged(tmp_path):
    with TestClient(create_app(Settings(data_dir=tmp_path, max_pending_tasks=1), FakeEngine)) as client:
        task_id = submit(client, b"not audio")
        assert wait_status(client, task_id)["status"] == "failed"
        event = client.get(f"/v1/tasks/{task_id}/result").json()
        assert event["event"] == "task.failed" and event["result"] is None
        assert event["error"]["code"] == "invalid_audio"
        assert client.app.state.tasks.capacity_usage()[1] == 0
        assert acknowledge(client, task_id, event).status_code == 200


def test_async_duration_limit_accepts_boundary_and_reports_overlong_audio(tmp_path):
    with TestClient(create_app(Settings(data_dir=tmp_path, max_audio_seconds=1), FakeEngine)) as client:
        accepted = submit(client, wav_bytes(seconds=1))
        assert wait_status(client, accepted)["status"] == "succeeded"
        overlong = submit(client, wav_bytes(seconds=2))
        assert wait_status(client, overlong)["status"] == "failed"
        event = client.get(f"/v1/tasks/{overlong}/result").json()
        assert event["error"]["code"] == "audio_too_large"
        assert client.app.state.engine.counts == [None]


def test_callbacks_retry_same_event_and_automatically_consume(tmp_path, audio_bytes):
    deliveries = []

    def sender(target, body, event_id, secret, timeout):
        deliveries.append((body, event_id))
        return 503 if len(deliveries) == 1 else 204

    settings = Settings(data_dir=tmp_path, callback_secret="test-key", callback_allowed_origins=("https://receiver.example",),
                        callback_retry_seconds=0.01)
    with TestClient(create_app(settings, FakeEngine, sender)) as client:
        task_id = submit(client, audio_bytes, callback_url="https://receiver.example/completed")
        status = wait_status(client, task_id, lambda data: data["consumed_by"] == "callback")
        assert status["callback"]["status"] == "delivered" and status["callback"]["attempts"] == 2
        assert len(deliveries) == 2 and deliveries[0] == deliveries[1]
        assert json.loads(deliveries[0][0])["result"]["text"] == "测试"
        assert client.get(f"/v1/tasks/{task_id}/result").status_code == 410
        assert client.app.state.engine.counts == [None], "Delivery failure must not re-run inference"
        assert acknowledge(client, task_id, json.loads(deliveries[0][0])).status_code == 200


def test_exhausted_callback_retains_result_for_manual_ack(tmp_path, audio_bytes):
    settings = Settings(data_dir=tmp_path, callback_secret="test-key", callback_allowed_origins=("http://receiver.example",),
                        callback_retry_seconds=0.01, callback_max_attempts=2)

    def sender(*args):
        raise TimeoutError("Receiver is unavailable")

    with TestClient(create_app(settings, FakeEngine, sender)) as client:
        task_id = submit(client, audio_bytes, callback_url="http://receiver.example/hook")
        status = wait_status(client, task_id, lambda data: data["callback"]["status"] == "failed")
        assert status["result_available"] and status["callback"]["attempts"] == 2
        event = client.get(f"/v1/tasks/{task_id}/result").json()
        assert acknowledge(client, task_id, event).status_code == 200


def test_manual_ack_racing_callback_is_idempotent(tmp_path, audio_bytes):
    entered, release = threading.Event(), threading.Event()
    deliveries = []

    def sender(target, body, *args):
        deliveries.append(body)
        entered.set()
        assert release.wait(5)
        return 200

    settings = Settings(data_dir=tmp_path, max_pending_tasks=1, callback_secret="test-key",
                        callback_allowed_origins=("https://receiver.example",))
    with TestClient(create_app(settings, FakeEngine, sender)) as client:
        task_id = submit(client, audio_bytes, callback_url="https://receiver.example/hook")
        assert entered.wait(5)
        try:
            event = client.get(f"/v1/tasks/{task_id}/result").json()
            assert acknowledge(client, task_id, event).status_code == 200
            next_id = submit(client, audio_bytes)
        finally:
            release.set()
        wait_status(client, next_id)
        status = client.get(f"/v1/tasks/{task_id}").json()
        assert status["consumed_by"] == "ack" and status["callback"]["status"] == "cancelled"
        assert len(deliveries) == 1 and client.app.state.tasks._active == 1


def test_idempotency_handles_full_capacity_conflicts_and_consumed_tasks(tmp_path, audio_bytes):
    with TestClient(create_app(Settings(data_dir=tmp_path, max_pending_tasks=1), FakeEngine)) as client:
        headers = {"Idempotency-Key": "meeting-1"}
        first = client.post("/v1/tasks", files={"file": ("a.wav", audio_bytes)}, headers=headers)
        assert first.status_code == 202
        task_id = first.json()["task_id"]
        wait_status(client, task_id)
        repeated = client.post("/v1/tasks", files={"file": ("a.wav", audio_bytes)}, headers=headers)
        assert repeated.status_code == 202 and repeated.json()["task_id"] == task_id
        conflict = client.post("/v1/tasks", files={"file": ("b.wav", b"different")}, headers=headers)
        assert conflict.status_code == 409
        event = client.get(f"/v1/tasks/{task_id}/result").json()
        acknowledge(client, task_id, event)
        consumed = client.post("/v1/tasks", files={"file": ("a.wav", audio_bytes)}, headers=headers)
        assert consumed.json()["consumed_by"] == "ack"
        assert client.app.state.engine.counts == [None]
        assert client.app.state.tasks.capacity_usage()[1] == 0


def test_audio_capacity_and_upload_limit_release_reservations(tmp_path, audio_bytes):
    settings = Settings(data_dir=tmp_path, max_pending_tasks=1, max_task_audio_mb=1, max_upload_mb=2)
    with TestClient(create_app(settings, FakeEngine)) as client:
        rejected = client.post("/v1/tasks", files={"file": ("large.wav", b"x" * (1024 * 1024 + 1))})
        assert rejected.status_code == 429
        assert client.app.state.tasks._active == 0 and client.app.state.tasks._audio_bytes == 0
        assert client.post("/v1/tasks", files={"file": ("empty.wav", b"")}).status_code == 422
        assert submit(client, audio_bytes)


def test_individual_upload_limit_is_checked_before_task_acceptance(tmp_path):
    with TestClient(create_app(Settings(data_dir=tmp_path, max_upload_mb=1), FakeEngine)) as client:
        response = client.post("/v1/tasks", files={"file": ("large.wav", b"x" * (1024 * 1024 + 1))})
        assert response.status_code == 413 and client.app.state.tasks._active == 0


def test_oversized_result_becomes_bounded_failed_event(tmp_path, audio_bytes):
    class LargeEngine(FakeEngine):
        def transcribe(self, *args):
            return {"text": "x" * (2 * 1024 * 1024)}

    with TestClient(create_app(Settings(data_dir=tmp_path, max_task_result_mb=1), LargeEngine)) as client:
        task_id = submit(client, audio_bytes)
        assert wait_status(client, task_id)["status"] == "failed"
        result = client.get(f"/v1/tasks/{task_id}/result")
        assert len(result.content) < 1024 and result.json()["error"]["code"] == "result_unavailable"
        acknowledge(client, task_id, result.json())


def test_tasks_are_not_restored(tmp_path, audio_bytes):
    settings = Settings(data_dir=tmp_path)
    with TestClient(create_app(settings, FakeEngine)) as client:
        task_id = submit(client, audio_bytes)
        wait_status(client, task_id)
    with TestClient(create_app(settings, FakeEngine)) as restarted:
        assert restarted.get(f"/v1/tasks/{task_id}").status_code == 404
        assert restarted.get(f"/v1/tasks/{task_id}/result").status_code == 404


def test_shutdown_closes_queued_audio_while_inference_is_busy(tmp_path, audio_bytes):
    app = create_app(Settings(data_dir=tmp_path), FakeEngine)
    try:
        with TestClient(app) as client:
            manager = app.state.tasks
            manager.inference_lock.acquire()
            task_id = submit(client, audio_bytes)
            assert client.get(f"/v1/tasks/{task_id}").json()["status"] == "queued"
            staged = manager._tasks[task_id].audio
            assert not staged.closed
        assert staged.closed and manager.capacity_usage()[1] == 0
    finally:
        if app.state.inference_lock.locked():
            app.state.inference_lock.release()


def test_consumed_history_is_bounded_without_expiring_pending_results(tmp_path, audio_bytes):
    settings = Settings(data_dir=tmp_path, task_history_limit=1)
    with TestClient(create_app(settings, FakeEngine)) as client:
        pending = submit(client, audio_bytes)
        wait_status(client, pending)
        history = []
        for _ in range(2):
            task_id = submit(client, audio_bytes)
            wait_status(client, task_id)
            acknowledge(client, task_id, client.get(f"/v1/tasks/{task_id}/result").json())
            history.append(task_id)
        assert client.get(f"/v1/tasks/{history[0]}").status_code == 404
        assert client.get(f"/v1/tasks/{history[1]}/result").status_code == 410
        assert client.get(f"/v1/tasks/{pending}/result").status_code == 200


def test_task_authentication_and_callback_configuration(tmp_path, audio_bytes):
    with TestClient(create_app(Settings(data_dir=tmp_path, api_token="secret"), FakeEngine)) as client:
        assert client.post("/v1/tasks", content=b"invalid").status_code == 401
        assert client.get("/v1/tasks/missing").status_code == 401
        assert client.post("/v1/tasks/missing/ack", json={}).status_code == 401
        response = client.post("/v1/tasks", headers={"Authorization": "Bearer secret"},
                               files={"file": ("a.wav", audio_bytes)}, data={"callback_url": "https://receiver.example/hook"})
        assert response.status_code == 503
    settings = Settings(data_dir=tmp_path, callback_secret="secret", callback_allowed_origins=("https://receiver.example",))
    with TestClient(create_app(settings, FakeEngine)) as client:
        response = client.post("/v1/tasks", files={"file": ("a.wav", audio_bytes)},
                               data={"callback_url": "https://unconfigured.example/hook"})
        assert response.status_code == 422 and client.app.state.tasks._active == 0
