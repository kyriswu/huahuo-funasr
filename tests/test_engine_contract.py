from contextlib import nullcontext
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from app.audio import AudioClip
from app.engine import SpeechEngine
from app.settings import Settings


def test_sdk_request_options_reset_and_center_label_alignment(vector):
    calls = []
    result = {
        "text": "你好", "timestamp": [[10, 100], [100, 200]],
        "sentence_info": [{"text": "你好", "start": 10, "end": 200, "spk": 1}],
        "spk_embedding_center": np.stack([np.roll(vector, 1), vector]),
    }

    def generate(**kwargs):
        calls.append(kwargs)
        return [result]

    engine = SpeechEngine.__new__(SpeechEngine)
    engine.settings = Settings()
    engine.torch = SimpleNamespace(inference_mode=nullcontext)
    engine.pipeline = SimpleNamespace(generate=generate)
    clip = AudioClip(np.zeros(16000, dtype=np.float32))
    record = {"speaker_id": "alice", "name": "Alice", "embedding": vector}
    first = engine.transcribe(clip, 2, [record], 0.5)
    engine.transcribe(clip, None, None, 0.5)
    assert calls[0]["preset_spk_num"] == 2 and calls[1]["preset_spk_num"] is None
    assert calls[0]["return_spk_center"] and not calls[1]["return_spk_center"]
    assert first["speakers"][0]["voiceprint"]["speaker_id"] == "alice"
    assert first["segments"][0]["speaker_id"] == "speaker_1"
    assert first["segments"][0]["start_ms"] == 10


def test_gpu_mode_refuses_missing_cuda(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    monkeypatch.setitem(sys.modules, "funasr", SimpleNamespace(AutoModel=lambda **kwargs: None))
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        SpeechEngine(Settings())


@pytest.fixture
def startup_models(monkeypatch, tmp_path):
    calls = []

    def load(path, **kwargs):
        calls.append((path.parent.name, kwargs, loader.torch is not torch))
        if path.read_bytes().endswith(b"corrupt"):
            raise RuntimeError("Broken checkpoint")

    torch = SimpleNamespace(load=load, set_num_threads=lambda value: None)
    loader = SimpleNamespace(torch=torch)

    class AutoModel:
        def __init__(self, **kwargs):
            for name, field in (("kwargs", "model"), ("vad_kwargs", "vad_model"),
                                ("punc_kwargs", "punc_model"), ("spk_kwargs", "spk_model")):
                setattr(self, name, self.build_model(model=kwargs[field], device=kwargs["device"])[1])

        @staticmethod
        def build_model(**kwargs):
            model_path = Path(kwargs["model"])
            checkpoint = "campplus_cn_common.bin" if model_path.name == "speaker" else "model.pt"
            loader.torch.load(model_path / checkpoint, map_location="cpu")
            return None, kwargs

    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "funasr", SimpleNamespace(AutoModel=AutoModel))
    monkeypatch.setitem(sys.modules, "funasr.train_utils.load_pretrained_model", loader)
    monkeypatch.setattr("app.engine.verify_models", lambda *_: ({}, "test-model"))
    for role in ("asr", "vad", "punc", "speaker"):
        root = tmp_path / role
        root.mkdir()
        checkpoint = "campplus_cn_common.bin" if role == "speaker" else "model.pt"
        contents = b"legacy" if role == "punc" else b"PK\x03\x04"
        (root / checkpoint).write_bytes(contents)
    return torch, loader, calls


@pytest.mark.parametrize(("mmap", "punc_zip"), [(False, True), (True, True), (True, False)])
def test_startup_only_maps_eligible_weights_when_enabled(tmp_path, startup_models, mmap, punc_zip):
    torch, loader, calls = startup_models
    if punc_zip:
        (tmp_path / "punc" / "model.pt").write_bytes(b"PK\x03\x04")
    SpeechEngine(Settings(model_root=tmp_path, device="cpu", model_checkpoint_mmap=mmap))
    for role, options, hooked in calls:
        assert hooked == mmap
        mapped = mmap and (role == "asr" or (role == "punc" and punc_zip))
        assert options == ({"map_location": "cpu", "mmap": True} if mapped else {"map_location": "cpu"})
    assert len(calls) == 4
    assert loader.torch is torch


def test_startup_restores_checkpoint_loader_after_failure(tmp_path, startup_models):
    torch, loader, calls = startup_models
    (tmp_path / "asr" / "model.pt").write_bytes(b"PK\x03\x04corrupt")
    with pytest.raises(RuntimeError, match="Broken checkpoint"):
        SpeechEngine(Settings(model_root=tmp_path, device="cpu", model_checkpoint_mmap=True))
    assert calls == [("asr", {"map_location": "cpu", "mmap": True}, True)]
    assert loader.torch is torch
