import io
import wave

import numpy as np
import pytest


def wav_bytes(seconds=4, rate=16000, channels=1):
    samples = (np.sin(np.arange(int(seconds * rate)) * 2 * np.pi * 220 / rate) * 4000).astype("<i2")
    if channels == 2:
        samples = np.column_stack([samples, samples]).reshape(-1)
    stream = io.BytesIO()
    with wave.open(stream, "wb") as target:
        target.setnchannels(channels)
        target.setsampwidth(2)
        target.setframerate(rate)
        target.writeframes(samples.tobytes())
    return stream.getvalue()


@pytest.fixture
def audio_bytes():
    return wav_bytes()


@pytest.fixture
def vector():
    result = np.zeros(192, dtype=np.float32)
    result[0] = 1
    return result
