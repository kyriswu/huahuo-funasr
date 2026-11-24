import hashlib
import json
import sys
from email.message import Message

import pytest

from scripts import download_models


MODEL_BYTES = b"test model data"
MODEL_SHA256 = hashlib.sha256(MODEL_BYTES).hexdigest()


class Response:
    def __init__(self, body, *, status=200, content_range=None, interrupted=False):
        self.status = status
        self.headers = Message()
        if content_range is not None:
            self.headers["Content-Range"] = content_range
        self.headers["Content-Length"] = str(len(MODEL_BYTES) if interrupted else len(body))
        self.body = body
        self.interrupted = interrupted

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def getcode(self):
        return self.status

    def read(self, size):
        if self.interrupted and size > len(self.body):
            raise TimeoutError("simulated buffered read timeout")
        return self.read1(size)

    def read1(self, size):
        if not self.body and self.interrupted:
            raise TimeoutError("simulated read timeout")
        chunk, self.body = self.body[:size], self.body[size:]
        return chunk


def run_download(monkeypatch, tmp_path, *, cache=None, path="model.bin", role="asr", digest=MODEL_SHA256):
    lock = tmp_path / "source.json"
    output = tmp_path / "models"
    tmp_path.mkdir(exist_ok=True)
    lock.write_text(json.dumps({
        "schema_version": 1,
        "models": {
            role: {
                "repo": "example/test-model",
                "revision": "v1.0",
                "files": [{"path": path, "sha256": digest, "size": len(MODEL_BYTES)}],
            },
        },
    }))
    args = ["download_models.py", "--lock", str(lock), "--output", str(output)]
    if cache is not None:
        args.extend(["--cache-dir", str(cache)])
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(download_models.time, "sleep", lambda _: None)
    download_models.main()
    return output


def resume_request(offset, size=len(MODEL_BYTES)):
    return {f"bytes={offset}-", f"bytes={offset}-{size - 1}"}


def resumed_response(offset):
    return Response(
        MODEL_BYTES[offset:], status=206,
        content_range=f"bytes {offset}-{len(MODEL_BYTES) - 1}/{len(MODEL_BYTES)}",
    )


def test_retries_an_interrupted_transfer_from_the_last_written_byte(monkeypatch, tmp_path):
    split = 5
    requests = []

    def open_model(request, **kwargs):
        requests.append(request.get_header("Range"))
        if len(requests) == 1:
            return Response(MODEL_BYTES[:split], interrupted=True)
        if requests[-1] is None:
            return Response(MODEL_BYTES)
        return resumed_response(split)

    monkeypatch.setattr(download_models, "urlopen", open_model)

    output = run_download(monkeypatch, tmp_path)

    assert (output / "asr/model.bin").read_bytes() == MODEL_BYTES
    assert requests[0] is None
    assert requests[1] in resume_request(split)


def test_resumes_cached_partial_after_the_previous_run_failed(monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    offset = 5
    requests = []

    def interrupted_download(request, **kwargs):
        requests.append(request.get_header("Range"))
        if len(requests) == 1:
            return Response(MODEL_BYTES[:offset], interrupted=True)
        raise TimeoutError("simulated connection timeout")

    monkeypatch.setattr(download_models, "urlopen", interrupted_download)

    with pytest.raises(TimeoutError):
        run_download(monkeypatch, tmp_path, cache=cache)

    assert requests[0] is None
    assert all(request in resume_request(offset) for request in requests[1:])
    partial = cache / f"{MODEL_SHA256}.part"
    assert partial.read_bytes() == MODEL_BYTES[:offset]
    assert not (tmp_path / "models/asr/model.bin").exists()

    def resume_download(request, **kwargs):
        assert request.get_header("Range") in resume_request(offset)
        return resumed_response(offset)

    monkeypatch.setattr(download_models, "urlopen", resume_download)

    output = run_download(monkeypatch, tmp_path, cache=cache)

    assert (output / "asr/model.bin").read_bytes() == MODEL_BYTES
    assert (cache / MODEL_SHA256).read_bytes() == MODEL_BYTES
    assert not partial.exists()


@pytest.mark.parametrize("start,total_delta", [(4, 0), (5, 1)])
def test_rejects_inconsistent_content_range(monkeypatch, tmp_path, start, total_delta):
    cache = tmp_path / "cache"
    cache.mkdir()
    partial = cache / f"{MODEL_SHA256}.part"
    offset = 5
    prefix = MODEL_BYTES[:offset]
    partial.write_bytes(prefix)

    def open_model(request, **kwargs):
        assert request.get_header("Range") in resume_request(offset)
        size = len(MODEL_BYTES)
        return Response(
            MODEL_BYTES[offset:], status=206,
            content_range=f"bytes {start}-{size - 1}/{size + total_delta}",
        )

    monkeypatch.setattr(download_models, "urlopen", open_model)

    with pytest.raises(ValueError, match="Content-Range"):
        run_download(monkeypatch, tmp_path, cache=cache)

    assert partial.read_bytes() == prefix
    assert not (tmp_path / "models/asr/model.bin").exists()


def test_full_response_replaces_a_partial_when_server_ignores_range(monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    partial = cache / f"{MODEL_SHA256}.part"
    partial.write_bytes(b"wrong")
    requests = []

    def open_model(request, **kwargs):
        requests.append(request.get_header("Range"))
        return Response(MODEL_BYTES)

    monkeypatch.setattr(download_models, "urlopen", open_model)

    output = run_download(monkeypatch, tmp_path, cache=cache)

    assert len(requests) == 1
    assert requests[0] in resume_request(5)
    assert (output / "asr/model.bin").read_bytes() == MODEL_BYTES
    assert not partial.exists()


def test_retries_without_range_after_resumed_file_fails_sha256(monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    partial = cache / f"{MODEL_SHA256}.part"
    partial.write_bytes(b"wrong")
    requests = []

    def open_model(request, **kwargs):
        requests.append(request.get_header("Range"))
        if len(requests) == 1:
            return resumed_response(5)
        return Response(MODEL_BYTES)

    monkeypatch.setattr(download_models, "urlopen", open_model)

    output = run_download(monkeypatch, tmp_path, cache=cache)

    assert len(requests) == 2
    assert requests[0] in resume_request(5)
    assert requests[1] is None
    assert (output / "asr/model.bin").read_bytes() == MODEL_BYTES
    assert (cache / MODEL_SHA256).read_bytes() == MODEL_BYTES


def test_replaces_corrupt_cached_blob_and_reuses_it_in_a_different_lock(monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    blob = cache / MODEL_SHA256
    blob.write_bytes(b"\0" * len(MODEL_BYTES))
    requests = []

    def open_model(request, **kwargs):
        requests.append(request.get_header("Range"))
        return Response(MODEL_BYTES)

    monkeypatch.setattr(download_models, "urlopen", open_model)

    first = run_download(monkeypatch, tmp_path / "first", cache=cache)
    second = run_download(
        monkeypatch, tmp_path / "second", cache=cache, role="vad", path="weights/renamed.bin",
    )

    assert requests == [None]
    assert blob.read_bytes() == MODEL_BYTES
    assert (first / "asr/model.bin").read_bytes() == MODEL_BYTES
    assert (second / "vad/weights/renamed.bin").read_bytes() == MODEL_BYTES
    manifest = json.loads((second / "manifest.json").read_text())
    assert manifest["models"]["vad"]["files"][0]["sha256"] == MODEL_SHA256


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"path": "../escape"}, "Unsafe path"),
        ({"path": "/escape"}, "Unsafe path"),
        ({"role": "../escape"}, "Unsafe role"),
        ({"role": "asr/nested"}, "Unsafe role"),
        ({"digest": "../escape"}, "Invalid SHA-256"),
        ({"digest": "z" * 64}, "Invalid SHA-256"),
    ],
)
def test_rejects_unsafe_lock_entries_before_fetching(monkeypatch, tmp_path, overrides, message):
    def unexpected_request(*args, **kwargs):
        pytest.fail("An invalid lock entry must never issue a request")

    monkeypatch.setattr(download_models, "urlopen", unexpected_request)

    with pytest.raises(ValueError, match=message):
        run_download(monkeypatch, tmp_path, cache=tmp_path / "cache", **overrides)
