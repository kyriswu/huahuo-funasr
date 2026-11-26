from contextlib import contextmanager
from importlib import import_module
from os import PathLike
from pathlib import Path
from threading import Lock, get_ident


_loader_lock = Lock()


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
def mapped_checkpoints(model_root: Path):
    checkpoints = {
        (model_root / "asr" / "model.pt").resolve(),
        (model_root / "punc" / "model.pt").resolve(),
    }
    owner = get_ident()
    loader = import_module("funasr.train_utils.load_pretrained_model")
    with _loader_lock:
        torch = loader.torch

        def load(source, *args, **kwargs):
            if get_ident() == owner and isinstance(source, (str, PathLike)):
                path = Path(source).resolve()
                if path in checkpoints and _is_zip_checkpoint(path):
                    kwargs["mmap"] = True
            return torch.load(source, *args, **kwargs)

        # Scope the hook to the pinned FunASR loader without changing torch.load globally.
        loader.torch = _CheckpointTorch(torch, load)
        try:
            yield
        finally:
            loader.torch = torch
