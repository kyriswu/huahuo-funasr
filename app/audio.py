from dataclasses import dataclass
from io import SEEK_END, UnsupportedOperation
import shutil
import subprocess
import tempfile
from typing import BinaryIO

import numpy as np

from app.settings import Settings

SAMPLE_RATE = 16000


class AudioError(ValueError):
    pass


class AudioTooLarge(AudioError):
    pass


@dataclass
class AudioClip:
    samples: np.ndarray

    @property
    def duration_seconds(self) -> float:
        return self.samples.size / SAMPLE_RATE


def decode_audio(source: BinaryIO, settings: Settings, max_seconds: int) -> AudioClip:
    """Bound encoded bytes, decoded duration, and FFmpeg execution time."""
    size = source.seek(0, SEEK_END)
    if size > settings.max_upload_mb * 1024 * 1024:
        raise AudioTooLarge("Encoded audio exceeds MAX_UPLOAD_MB")
    if not size:
        raise AudioError("Audio upload is empty")
    source.seek(0)
    try:
        source.fileno()
    except UnsupportedOperation:
        with tempfile.TemporaryFile(prefix="funasr-encoded-") as staged:
            shutil.copyfileobj(source, staged, length=1024 * 1024)
            staged.seek(0)
            return _decode_file(staged, settings, max_seconds)
    return _decode_file(source, settings, max_seconds)


def _decode_file(source: BinaryIO, settings: Settings, max_seconds: int) -> AudioClip:
    # A regular descriptor stays seekable for formats such as M4A with metadata at the end.
    command = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-protocol_whitelist", "file,pipe", "-threads", "1", "-i", "/dev/fd/0",
        "-map", "0:a:0", "-vn", "-sn", "-dn", "-t", str(max_seconds + 1),
        "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-threads", "1", "pipe:1",
    ]
    with tempfile.TemporaryFile(prefix="funasr-decoded-") as decoded:
        try:
            result = subprocess.run(
                command, stdin=source, stdout=decoded, stderr=subprocess.DEVNULL,
                timeout=settings.decode_timeout_seconds, check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise AudioError("Audio decoding timed out") from error
        if result.returncode:
            raise AudioError("Unsupported or invalid audio; upload WAV, MP3, FLAC, M4A, or OGG")
        # Decode up to one extra second to detect overlong audio without truncating it silently.
        if decoded.seek(0, SEEK_END) > max_seconds * SAMPLE_RATE * 4:
            raise AudioTooLarge(f"Audio duration exceeds {max_seconds} seconds")
        decoded.seek(0)
        samples = np.fromfile(decoded, dtype="<f4")
    if samples.size < SAMPLE_RATE // 10:
        raise AudioError("Audio must be at least 0.1 seconds")
    # Keep the finite-value check's temporary array small even for six-hour inputs.
    for offset in range(0, samples.size, 1024 * 1024):
        if not np.isfinite(samples[offset:offset + 1024 * 1024]).all():
            raise AudioError("Audio contains invalid sample values")
    return AudioClip(samples)
