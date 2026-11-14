#!/usr/bin/env python3
"""Exercise real HTTP inference using baked public samples, including a restart.

Run with: docker run --rm --gpus all --network none --entrypoint python IMAGE scripts/offline_smoke.py
The script starts its own localhost-only server and cleans up its private test gallery.
"""

import argparse
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import tempfile
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import wave


def join_wavs(parts: list[bytes]) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(16000)
        for content in parts:
            with wave.open(io.BytesIO(content), "rb") as source:
                assert (source.getnchannels(), source.getsampwidth(), source.getframerate()) == (1, 2, 16000)
                target.writeframes(source.readframes(source.getnframes()))
            target.writeframes(b"\0" * 16000)  # Half a second between utterances.
    return output.getvalue()


def silent_wav() -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(16000)
        target.writeframes(b"\0" * 16000 * 2 * 4)
    return output.getvalue()


def multipart(files: dict[str, bytes], fields: dict) -> tuple[bytes, str]:
    boundary = "funasr-" + secrets.token_hex(12)
    chunks = []
    for key, value in fields.items():
        chunks.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode())
    for key, value in files.items():
        chunks.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"; filename="sample.wav"\r\nContent-Type: audio/wav\r\n\r\n'.encode())
        chunks.extend([value, b"\r\n"])
    chunks.append(f"--{boundary}--\r\n".encode())
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    root = Path(os.getenv("MODEL_ROOT", "/opt/models"))
    examples = root / "speaker" / "examples"
    first = (examples / "speaker1_a_cn_16k.wav").read_bytes()
    other_text_same_person = (examples / "speaker1_b_cn_16k.wav").read_bytes()
    other_person = (examples / "speaker2_a_cn_16k.wav").read_bytes()
    # Repeat these short public samples to exceed the minimum effective speech requirement.
    reference = join_wavs([first, first])
    positive = join_wavs([other_text_same_person, other_text_same_person])
    negative = join_wavs([other_person, other_person])
    conversation = join_wavs([first, other_person, other_text_same_person, other_person])
    token = secrets.token_hex(24)
    callback_secret = secrets.token_hex(24)
    deliveries = []
    callback_errors = []

    class CallbackReceiver(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            try:
                stamp, event_id = self.headers["X-FunASR-Timestamp"], self.headers["X-FunASR-Event-Id"]
                expected = "v1=" + hmac.new(
                    callback_secret.encode(), stamp.encode() + b"." + event_id.encode() + b"." + body, hashlib.sha256,
                ).hexdigest()
                assert hmac.compare_digest(self.headers["X-FunASR-Signature"], expected)
                assert abs(time.time() - int(stamp)) < 300
                event = json.loads(body)
                assert event["event_id"] == event_id and event["result"]["text"].strip()
                deliveries.append(body)
                # The first notification fails; the second is accepted only after saving it.
                self.send_response(503 if len(deliveries) == 1 else 204)
            except Exception as error:
                callback_errors.append(type(error).__name__)
                self.send_response(400)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *args):
            pass

    receiver = ThreadingHTTPServer(("127.0.0.1", 0), CallbackReceiver)
    receiver_thread = threading.Thread(target=receiver.serve_forever, daemon=True)
    receiver_thread.start()
    callback_origin = f"http://127.0.0.1:{receiver.server_port}"
    report = {"checks": [], "device": os.getenv("DEVICE", "cuda:0")}
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    base = f"http://127.0.0.1:{port}"

    def request(path, files=None, fields=None, method=None, expected=200, auth=token, json_body=None):
        headers = {"Authorization": f"Bearer {auth}"}
        body = None
        if files is not None:
            body, content_type = multipart(files, fields or {})
            headers["Content-Type"] = content_type
        if json_body is not None:
            body = json.dumps(json_body).encode()
            headers["Content-Type"] = "application/json"
        operation = Request(base + path, data=body, headers=headers, method=method)
        try:
            with urlopen(operation, timeout=600) as response:
                status, data = response.status, json.load(response)
        except HTTPError as error:
            status, data = error.code, json.loads(error.read())
        assert status == expected, f"{path}: expected {expected}, got {status}: {data}"
        return data

    def wait_task(task_id, predicate):
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            status = request(f"/v1/tasks/{task_id}")
            if predicate(status):
                return status
            time.sleep(0.05)
        raise TimeoutError(f"Async task did not reach expected state: {status}")

    data_parent = Path(os.getenv("DATA_DIR", "/data"))
    data_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="funasr-smoke-", dir=data_parent) as workspace:
        log_path = Path(workspace) / "server.log"
        process = None
        log = log_path.open("w")
        environment = {
            **os.environ, "API_TOKEN": token, "DATA_DIR": workspace,
            "CALLBACK_SECRET": callback_secret, "CALLBACK_ALLOWED_ORIGINS": callback_origin,
            "CALLBACK_RETRY_SECONDS": "0.05",
        }

        def start_server():
            child = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port), "--no-access-log"],
                env=environment, stdout=log, stderr=subprocess.STDOUT,
            )
            deadline = time.monotonic() + 600
            while time.monotonic() < deadline:
                if child.poll() is not None:
                    raise RuntimeError("Server failed to start:\n" + log_path.read_text()[-12000:])
                try:
                    request("/readyz")
                    return child
                except (URLError, TimeoutError, ConnectionError):
                    time.sleep(0.5)
            child.kill()
            child.wait()
            raise TimeoutError("Model startup exceeded 600 seconds")

        def stop_server(child):
            if child is not None and child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()

        try:
            process = start_server()
            report["checks"].append("local_model_startup")
            print("Models ready; exercising real inference", flush=True)
            request("/v1/voiceprints", expected=401, auth="wrong")
            request("/v1/voiceprints/alice", {"file": reference}, {"name": "Alice"}, expected=201)
            request("/v1/voiceprints/bob", {"file": negative}, {"name": "Bob"}, expected=201)
            report["checks"].extend(["authentication", "enrollment"])
            same = request("/v1/voiceprints/verify", {"reference": reference, "query": positive})
            different = request("/v1/voiceprints/verify", {"reference": reference, "query": negative})
            report["verification_positive"] = same
            report["verification_negative"] = different
            assert same["same_speaker"] and not different["same_speaker"], (same, different)
            report["checks"].extend(["same_speaker_different_utterance", "different_speaker_rejected"])
            identity = request("/v1/voiceprints/identify", {"file": positive})
            assert identity["speaker_id"] == "alice", identity
            report["identification"] = identity
            report["checks"].append("one_to_many_identification")
            transcript = request("/v1/transcriptions", {"file": conversation}, {"speaker_count": 2, "identify_speakers": "true"})
            assert transcript["text"].strip() and transcript["timestamps_ms"] and transcript["segments"], transcript
            assert len(transcript["speakers"]) == 2, transcript
            assert all(0 <= segment["start_ms"] <= segment["end_ms"] <= transcript["duration_seconds"] * 1000 + 100 for segment in transcript["segments"])
            assert {speaker["voiceprint"]["speaker_id"] for speaker in transcript["speakers"]} == {"alice", "bob"}, transcript
            report["transcription"] = transcript
            report["checks"].extend(["asr_with_timestamps", "two_speaker_diarization_known_count", "transcription_identity_mapping"])
            automatic = request("/v1/transcriptions", {"file": reference})
            assert automatic["text"].strip(), automatic
            report["automatic_speaker_count"] = len(automatic["speakers"])
            report["checks"].append("automatic_count_after_explicit_count")
            silence = silent_wav()
            quiet = request("/v1/transcriptions", {"file": silence})
            assert not quiet["text"] and not quiet["segments"], quiet
            request("/v1/voiceprints/identify", {"file": silence}, expected=422)
            request("/v1/transcriptions", {"file": b"invalid audio"}, expected=422)
            report["checks"].extend(["silence_handling", "invalid_audio_handling"])
            task_id = request("/v1/tasks", {"file": conversation},
                              {"speaker_count": 2, "identify_speakers": "true"}, expected=202)["task_id"]
            wait_task(task_id, lambda state: state["status"] in {"succeeded", "failed"})
            event = request(f"/v1/tasks/{task_id}/result")
            assert event == request(f"/v1/tasks/{task_id}/result")
            assert event["status"] == "succeeded" and len(event["result"]["speakers"]) == 2
            assert request(f"/v1/tasks/{task_id}")["result_available"]
            for directory in Path(tempfile.gettempdir()).glob("funasr-tasks-*"):
                assert not list(directory.iterdir()), "Audio must be deleted before consumption"
            ack_body = {"event_id": event["event_id"]}
            assert request(f"/v1/tasks/{task_id}/ack", json_body=ack_body)["acknowledged"]
            assert request(f"/v1/tasks/{task_id}/ack", json_body=ack_body)["acknowledged"]
            request(f"/v1/tasks/{task_id}/result", expected=410)
            report["checks"].extend(["async_result_repeatable_until_ACK", "audio_deleted_before_delivery", "idempotent_ACK_cleanup"])
            report["async_transcription"] = event["result"]
            callback_id = request("/v1/tasks", {"file": reference}, {"callback_url": callback_origin + "/completed"}, expected=202)["task_id"]
            wait_task(callback_id, lambda state: state["consumed_by"] == "callback")
            assert not callback_errors and len(deliveries) == 2 and deliveries[0] == deliveries[1]
            request(f"/v1/tasks/{callback_id}/result", expected=410)
            report["checks"].extend(["signed_callback_retry_same_event", "callback_2xx_cleanup_without_ACK"])
            discarded_id = request("/v1/tasks", {"file": b"invalid"}, expected=202)["task_id"]
            assert wait_task(discarded_id, lambda state: state["status"] == "failed")["result_available"]
            stop_server(process)
            process = start_server()
            persisted = request("/v1/voiceprints")["voiceprints"]
            assert {item["speaker_id"] for item in persisted} == {"alice", "bob"}
            again = request("/v1/voiceprints/identify", {"file": positive})
            assert again["speaker_id"] == "alice"
            request(f"/v1/tasks/{discarded_id}", expected=404)
            report["checks"].extend(["gallery_persistence_after_process_restart", "restart_discards_unconsumed_tasks"])
        except Exception:
            print(log_path.read_text()[-16000:], file=sys.stderr)
            raise
        finally:
            stop_server(process)
            log.close()
            receiver.shutdown()
            receiver.server_close()
            receiver_thread.join(5)
    report["status"] = "passed"
    try:
        hardware = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        report["hardware"] = hardware.stdout.strip()
    except OSError:
        pass
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
