from contextlib import contextmanager
from importlib import import_module
from os import PathLike
from pathlib import Path
from threading import Lock, get_ident
from time import perf_counter


_loader_lock = Lock()


@contextmanager
def startup_phase(durations: dict[str, float], stage: str):
    started = perf_counter()
    try:
        yield
    finally:
        durations[stage] = perf_counter() - started


class _CheckpointTorch:
    def __init__(self, torch, load):
        self._torch = torch
        self.load = load

    def __getattr__(self, name):
        return getattr(self._torch, name)


def _is_zip_checkpoint(path: Path) -> bool:
    # PyTorch's mmap loader requires the ZIP serialization format.
    with path.open("rb") as checkpoint:
        return checkpoint.read(4) == b"PK\x03\x04"


@contextmanager
def profile_checkpoints(model_root: Path, durations: dict[str, float], modes: dict[str, str], mmap: bool):
    checkpoints = {
        (model_root / "asr" / "model.pt").resolve(): "asr",
        (model_root / "vad" / "model.pt").resolve(): "vad",
        (model_root / "punc" / "model.pt").resolve(): "punc",
        (model_root / "speaker" / "campplus_cn_common.bin").resolve(): "speaker",
    }
    owner = get_ident()
    loader = import_module("funasr.train_utils.load_pretrained_model")
    with _loader_lock:
        torch = loader.torch

        def load(source, *args, **kwargs):
            if get_ident() == owner and isinstance(source, (str, PathLike)):
                path = Path(source).resolve()
                role = checkpoints.get(path)
                if role is not None:
                    if mmap and role in {"asr", "punc"} and _is_zip_checkpoint(path):
                        kwargs["mmap"] = True
                    modes[role] = "mmap" if kwargs.get("mmap") else "standard"
                    with startup_phase(durations, f"read_{role}_checkpoint"):
                        return torch.load(source, *args, **kwargs)
            return torch.load(source, *args, **kwargs)

        # Scope the hook to the pinned FunASR loader without changing torch.load globally.
        loader.torch = _CheckpointTorch(torch, load)
        try:
            yield
        finally:
            loader.torch = torch
