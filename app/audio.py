from dataclasses import dataclass
from pathlib import Path
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
    limit_bytes = settings.max_upload_mb * 1024 * 1024
    with tempfile.TemporaryDirectory(prefix="funasr-audio-") as directory:
        input_path = Path(directory) / "input.audio"
        output_path = Path(directory) / "decoded.f32"
        total = 0
        source.seek(0)
        with input_path.open("wb") as target:
            while chunk := source.read(1024 * 1024):
                total += len(chunk)
                if total > limit_bytes:
                    raise AudioTooLarge("Encoded audio exceeds MAX_UPLOAD_MB")
                target.write(chunk)
        if not total:
            raise AudioError("Audio upload is empty")
        # Read one extra second so overlong recordings are rejected, never silently truncated.
        command = [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
            "-protocol_whitelist", "file,pipe", "-threads", "1", "-i", str(input_path),
            "-map", "0:a:0", "-vn", "-sn", "-dn", "-t", str(max_seconds + 1),
            "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-threads", "1",
            str(output_path),
        ]
        try:
            result = subprocess.run(
                command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=settings.decode_timeout_seconds, check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise AudioError("Audio decoding timed out") from error
        if result.returncode or not output_path.is_file():
            raise AudioError("Unsupported or invalid audio; upload WAV, MP3, FLAC, M4A, or OGG")
        if output_path.stat().st_size > max_seconds * SAMPLE_RATE * 4:
            raise AudioTooLarge(f"Audio duration exceeds {max_seconds} seconds")
        samples = np.fromfile(output_path, dtype="<f4")
    if samples.size < SAMPLE_RATE // 10:
        raise AudioError("Audio must be at least 0.1 seconds")
    if not np.isfinite(samples).all():
        raise AudioError("Audio contains invalid sample values")
    return AudioClip(samples)
