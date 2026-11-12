import io

import pytest

from app.audio import AudioError, AudioTooLarge, decode_audio
from app.settings import Settings
from conftest import wav_bytes


def test_decode_resamples_and_downmixes():
    clip = decode_audio(io.BytesIO(wav_bytes(seconds=2, rate=8000, channels=2)), Settings(), 10)
    assert clip.samples.ndim == 1
    assert abs(clip.duration_seconds - 2) < 0.02


@pytest.mark.parametrize("content", [b"", b"this is not audio"])
def test_bad_audio_is_rejected(content):
    with pytest.raises(AudioError):
        decode_audio(io.BytesIO(content), Settings(), 10)


def test_duration_limit_rejects_instead_of_truncating():
    with pytest.raises(AudioTooLarge):
        decode_audio(io.BytesIO(wav_bytes(seconds=3)), Settings(), 1)


def test_encoded_size_limit():
    with pytest.raises(AudioTooLarge):
        decode_audio(io.BytesIO(b"0" * (1024 * 1024 + 1)), Settings(max_upload_mb=1), 10)
