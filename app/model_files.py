import hashlib
import json
from pathlib import Path, PurePosixPath


def verify_models(root: Path, lock_path: Path) -> tuple[dict, str]:
    """Verify every baked file against the reviewed model lock, not hub aliases."""
    lock = json.loads(lock_path.read_text())
    if lock.get("schema_version") != 1 or set(lock["models"]) != {"asr", "vad", "punc", "speaker"}:
        raise ValueError("Invalid model lock")
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest != lock:
        raise ValueError("Model manifest does not match this service's model lock")
    for role, model in lock["models"].items():
        for entry in model["files"]:
            relative = PurePosixPath(entry["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe model path")
            path = root / role / str(relative)
            if not path.is_file() or path.stat().st_size != entry["size"]:
                raise ValueError(f"Missing or incomplete local model file: {role}/{relative}")
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != entry["sha256"]:
                raise ValueError(f"Model checksum mismatch: {role}/{relative}")
    speaker = lock["models"]["speaker"]
    fingerprint = hashlib.sha256(
        json.dumps(speaker, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return lock, f"{speaker['repo']}@{speaker['revision']}:{fingerprint}"
