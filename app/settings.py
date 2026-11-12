from dataclasses import dataclass
import os
from pathlib import Path
import re


@dataclass(frozen=True)
class Settings:
    model_root: Path = Path("/opt/models")
    model_lock: Path = Path("/opt/app/model-lock.json")
    data_dir: Path = Path("/data")
    device: str = "cuda:0"
    api_token: str = ""
    cpu_threads: int = 4
    max_upload_mb: int = 256
    max_audio_seconds: int = 7200
    max_voiceprint_seconds: int = 120
    min_voice_seconds: float = 3.0
    decode_timeout_seconds: int = 120
    batch_size_s: int = 120
    max_speakers: int = 15
    voiceprint_threshold: float = 0.5

    def __post_init__(self):
        if not re.fullmatch(r"cpu|cuda(?::\d+)?", self.device):
            raise ValueError("DEVICE must be cpu, cuda, or cuda:N")
        for field in (
            "cpu_threads", "max_upload_mb", "max_audio_seconds", "max_voiceprint_seconds",
            "decode_timeout_seconds", "batch_size_s", "max_speakers",
        ):
            if getattr(self, field) < 1:
                raise ValueError(f"{field} must be positive")
        if not -1 <= self.voiceprint_threshold <= 1:
            raise ValueError("VOICEPRINT_THRESHOLD must be in [-1, 1]")
        if not 0 < self.min_voice_seconds <= self.max_voiceprint_seconds:
            raise ValueError("Invalid MIN_VOICE_SECONDS")

    @classmethod
    def from_env(cls):
        return cls(
            model_root=Path(os.getenv("MODEL_ROOT", "/opt/models")),
            model_lock=Path(os.getenv("MODEL_LOCK", "/opt/app/model-lock.json")),
            data_dir=Path(os.getenv("DATA_DIR", "/data")),
            device=os.getenv("DEVICE", "cuda:0"),
            api_token=os.getenv("API_TOKEN", ""),
            cpu_threads=int(os.getenv("CPU_THREADS", "4")),
            max_upload_mb=int(os.getenv("MAX_UPLOAD_MB", "256")),
            max_audio_seconds=int(os.getenv("MAX_AUDIO_SECONDS", "7200")),
            max_voiceprint_seconds=int(os.getenv("MAX_VOICEPRINT_SECONDS", "120")),
            min_voice_seconds=float(os.getenv("MIN_VOICE_SECONDS", "3")),
            decode_timeout_seconds=int(os.getenv("DECODE_TIMEOUT_SECONDS", "120")),
            batch_size_s=int(os.getenv("BATCH_SIZE_S", "120")),
            max_speakers=int(os.getenv("MAX_SPEAKERS", "15")),
            voiceprint_threshold=float(os.getenv("VOICEPRINT_THRESHOLD", "0.5")),
        )
