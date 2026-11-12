from contextlib import nullcontext
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
    import sys
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    monkeypatch.setitem(sys.modules, "funasr", SimpleNamespace(AutoModel=lambda **kwargs: None))
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        SpeechEngine(Settings())
