#!/usr/bin/env python3
"""Download the exact model files in model-lock.json; no hub SDK is required."""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote
from urllib.request import Request, urlopen


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(job: tuple[Path, dict, dict]) -> None:
    root, model, entry = job
    relative = PurePosixPath(entry["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Unsafe path in model lock")
    destination = root / str(relative)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and sha256(destination) == entry["sha256"]:
        return
    url = (
        f"https://modelscope.cn/models/{model['repo']}/resolve/"
        f"{quote(model['revision'], safe='')}/{quote(str(relative), safe='/')}"
    )
    partial = destination.with_name(destination.name + ".part")
    for attempt in range(4):
        try:
            print(f"Downloading {model['repo']}@{model['revision']}/{relative}", flush=True)
            with urlopen(Request(url, headers={"User-Agent": "funasr-eci-build/1"}), timeout=120) as response:
                with partial.open("wb") as target:
                    shutil.copyfileobj(response, target, length=1024 * 1024)
            if partial.stat().st_size != entry["size"] or sha256(partial) != entry["sha256"]:
                raise ValueError(f"Size or SHA-256 mismatch: {destination}")
            os.replace(partial, destination)
            return
        except Exception:
            partial.unlink(missing_ok=True)
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=Path("model-lock.json"))
    parser.add_argument("--output", type=Path, default=Path("models"))
    args = parser.parse_args()
    lock = json.loads(args.lock.read_text())
    if lock["schema_version"] != 1:
        raise ValueError("Unsupported model lock schema")
    jobs = [
        (args.output / role, model, entry)
        for role, model in lock["models"].items()
        for entry in model["files"]
    ]
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(download, jobs))
    (args.output / "manifest.json").write_text(json.dumps(lock, indent=2) + "\n")
    print(f"Verified {len(jobs)} files in {args.output}", flush=True)


if __name__ == "__main__":
    main()
