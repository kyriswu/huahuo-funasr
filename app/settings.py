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
    max_pending_tasks: int = 16
    max_task_audio_mb: int = 1024
    max_task_result_mb: int = 16
    task_status_ttl_seconds: int = 3600
    task_history_limit: int = 1024
    callback_allowed_origins: tuple[str, ...] = ()
    callback_secret: str = ""
    callback_timeout_seconds: float = 10.0
    callback_max_attempts: int = 10
    callback_retry_seconds: float = 10.0
    metrics_port: int | None = None

    def __post_init__(self):
        if not re.fullmatch(r"cpu|cuda(?::\d+)?", self.device):
            raise ValueError("DEVICE must be cpu, cuda, or cuda:N")
        if self.metrics_port is not None and not 0 <= self.metrics_port <= 65535:
            raise ValueError("METRICS_PORT must be a valid TCP port")
        for field in (
            "cpu_threads", "max_upload_mb", "max_audio_seconds", "max_voiceprint_seconds",
            "decode_timeout_seconds", "batch_size_s", "max_speakers",
            "max_pending_tasks", "max_task_audio_mb", "max_task_result_mb",
            "task_status_ttl_seconds", "task_history_limit", "callback_max_attempts",
        ):
            if getattr(self, field) < 1:
                raise ValueError(f"{field} must be positive")
        if not -1 <= self.voiceprint_threshold <= 1:
            raise ValueError("VOICEPRINT_THRESHOLD must be in [-1, 1]")
        if not 0 < self.min_voice_seconds <= self.max_voiceprint_seconds:
            raise ValueError("Invalid MIN_VOICE_SECONDS")
        for field in ("callback_timeout_seconds", "callback_retry_seconds"):
            if not 0 < getattr(self, field) <= 3600:
                raise ValueError(f"{field} must be in (0, 3600]")

    @classmethod
    def from_env(cls):
        metrics_port = os.getenv("METRICS_PORT")
        default_batch_size_s = "10" if os.getenv("FUNASR_BUILD_PROFILE") == "a10-2gb" else "120"
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
            batch_size_s=int(os.getenv("BATCH_SIZE_S", default_batch_size_s)),
            max_speakers=int(os.getenv("MAX_SPEAKERS", "15")),
            voiceprint_threshold=float(os.getenv("VOICEPRINT_THRESHOLD", "0.5")),
            max_pending_tasks=int(os.getenv("MAX_PENDING_TASKS", "16")),
            max_task_audio_mb=int(os.getenv("MAX_TASK_AUDIO_MB", "1024")),
            max_task_result_mb=int(os.getenv("MAX_TASK_RESULT_MB", "16")),
            task_status_ttl_seconds=int(os.getenv("TASK_STATUS_TTL_SECONDS", "3600")),
            task_history_limit=int(os.getenv("TASK_HISTORY_LIMIT", "1024")),
            callback_allowed_origins=tuple(
                item.strip() for item in os.getenv("CALLBACK_ALLOWED_ORIGINS", "").split(",") if item.strip()
            ),
            callback_secret=os.getenv("CALLBACK_SECRET", ""),
            callback_timeout_seconds=float(os.getenv("CALLBACK_TIMEOUT_SECONDS", "10")),
            callback_max_attempts=int(os.getenv("CALLBACK_MAX_ATTEMPTS", "10")),
            callback_retry_seconds=float(os.getenv("CALLBACK_RETRY_SECONDS", "10")),
            metrics_port=int(metrics_port) if metrics_port else None,
        )
