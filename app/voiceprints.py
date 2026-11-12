from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import tempfile

import numpy as np

DIMENSION = 192
ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")


class StoreError(RuntimeError):
    pass


class VoiceprintExists(ValueError):
    pass


def validate_id(speaker_id: str) -> str:
    if not ID_PATTERN.fullmatch(speaker_id):
        raise ValueError("speaker_id must be 1-64 ASCII letters, digits, underscores or hyphens; start with a letter or digit")
    return speaker_id


def normalize(embedding) -> np.ndarray:
    vector = np.asarray(embedding, dtype=np.float64).reshape(-1)
    if vector.size != DIMENSION or not np.isfinite(vector).all():
        raise ValueError(f"Expected a finite {DIMENSION}-dimensional speaker embedding")
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError("Speaker embedding has zero norm")
    return (vector / norm).astype(np.float32)


def match_voiceprint(embedding, records: list[dict], threshold: float, top_k: int = 5) -> dict:
    if not -1 <= threshold <= 1:
        raise ValueError("threshold must be in [-1, 1]")
    vector = normalize(embedding)
    ranked = sorted(
        (
            {"speaker_id": item["speaker_id"], "name": item["name"],
             "score": float(np.clip(np.dot(vector, normalize(item["embedding"])), -1, 1))}
            for item in records
        ),
        key=lambda item: (-item["score"], item["speaker_id"]),
    )
    best = ranked[0] if ranked else None
    matched = bool(best and best["score"] >= threshold)
    return {
        "matched": matched,
        "speaker_id": best["speaker_id"] if matched else None,
        "name": best["name"] if matched else None,
        "score": round(best["score"], 6) if best else None,
        "threshold": threshold,
        "candidates": [{**item, "score": round(item["score"], 6)} for item in ranked[:top_k]],
    }


class VoiceprintStore:
    """Single-writer persistent gallery; atomic replacement also works on mounted NAS."""

    def __init__(self, directory: Path, model_id: str):
        self.directory = directory
        self.model_id = model_id
        directory.mkdir(parents=True, exist_ok=True)
        self.records()  # Fail startup on incompatible or corrupt stored data.

    @contextmanager
    def _locked(self, exclusive: bool = False):
        with (self.directory / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _read(self, path: Path) -> dict:
        try:
            record = json.loads(path.read_text())
            if record["schema_version"] != 1 or record["model_id"] != self.model_id:
                raise ValueError("Incompatible voiceprint model or schema")
            if validate_id(record["speaker_id"]) != path.stem:
                raise ValueError("Voiceprint filename does not match speaker_id")
            if not isinstance(record["name"], str) or len(record["name"]) > 128:
                raise ValueError("Invalid voiceprint name")
            normalize(record["embedding"])
            for field in ("created_at", "updated_at", "speech_seconds"):
                if field not in record:
                    raise ValueError(f"Missing {field}")
            return record
        except (ValueError, KeyError, TypeError, OSError) as error:
            raise StoreError(f"Cannot load voiceprint {path.name}: {error}") from error

    def records(self) -> list[dict]:
        with self._locked():
            return [self._read(path) for path in sorted(self.directory.glob("*.json"))]

    @staticmethod
    def public(record: dict) -> dict:
        return {key: value for key, value in record.items() if key != "embedding"}

    def put(self, speaker_id: str, name: str, embedding, speech_seconds: float, replace: bool = False) -> dict:
        validate_id(speaker_id)
        if len(name) > 128:
            raise ValueError("name must not exceed 128 characters")
        vector = normalize(embedding)
        path = self.directory / f"{speaker_id}.json"
        with self._locked(exclusive=True):
            previous = self._read(path) if path.exists() else None
            if previous and not replace:
                raise VoiceprintExists("Voiceprint exists; use replace=true to overwrite")
            now = datetime.now(timezone.utc).isoformat()
            record = {
                "schema_version": 1, "speaker_id": speaker_id, "name": name or speaker_id,
                "model_id": self.model_id, "embedding": vector.tolist(),
                "speech_seconds": round(speech_seconds, 3),
                "created_at": previous["created_at"] if previous else now, "updated_at": now,
            }
            descriptor, temporary = tempfile.mkstemp(prefix=".voiceprint-", suffix=".tmp", dir=self.directory)
            try:
                with os.fdopen(descriptor, "w") as stream:
                    json.dump(record, stream, ensure_ascii=False, allow_nan=False)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
                directory_fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            finally:
                Path(temporary).unlink(missing_ok=True)
        return self.public(record)

    def delete(self, speaker_id: str) -> None:
        validate_id(speaker_id)
        with self._locked(exclusive=True):
            path = self.directory / f"{speaker_id}.json"
            if not path.exists():
                raise KeyError(speaker_id)
            path.unlink()
