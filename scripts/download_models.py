#!/usr/bin/env python3
"""Download the exact model files in model-lock.json; no hub SDK is required."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
from http.client import HTTPException
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import time
from urllib.parse import quote
from urllib.request import Request, urlopen


Job = tuple[Path, dict, dict]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def matches(path: Path, entry: dict) -> bool:
    return path.is_file() and path.stat().st_size == entry["size"] and sha256(path) == entry["sha256"]


def transfer(url: str, partial: Path, offset: int, size: int) -> None:
    headers = {"User-Agent": "funasr-eci-build/1", "Accept-Encoding": "identity"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    with urlopen(Request(url, headers=headers), timeout=120) as response:
        if response.status == 206:
            expected_range = f"bytes {offset}-{size - 1}/{size}"
            if response.headers.get("Content-Range") != expected_range:
                raise ValueError(f"Invalid Content-Range for {url}; expected {expected_range}")
            mode = "ab"
        elif response.status == 200:
            offset = 0
            mode = "wb"
        else:
            raise ValueError(f"Unexpected HTTP status {response.status} for {url}")
        with partial.open(mode) as target:
            while chunk := response.read1(1024 * 1024):
                offset += len(chunk)
                if offset > size:
                    raise ValueError(f"Response exceeds locked size for {url}")
                target.write(chunk)


def fetch(path: Path, model: dict, entry: dict) -> None:
    relative = entry["path"]
    url = (
        f"https://modelscope.cn/models/{model['repo']}/resolve/"
        f"{quote(model['revision'], safe='')}/{quote(relative, safe='/')}"
    )
    partial = path.with_name(path.name + ".part")
    size = entry["size"]
    for attempt in range(4):
        try:
            offset = partial.stat().st_size if partial.is_file() else 0
            if offset < size or not partial.is_file():
                print(f"Downloading {model['repo']}@{model['revision']}/{relative} from byte {offset}", flush=True)
                transfer(url, partial, offset, size)
            if partial.stat().st_size < size:
                raise OSError(f"Incomplete download for {url}")
            if not matches(partial, entry):
                partial.unlink()
                raise ValueError(f"Size or SHA-256 mismatch for {url}")
            os.replace(partial, path)
            return
        except (OSError, HTTPException, ValueError):
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def download_group(jobs: list[Job], cache_dir: Path | None) -> None:
    for destination, model, entry in jobs:
        if matches(destination, entry):
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if cache_dir is None:
            fetch(destination, model, entry)
            continue
        cached = cache_dir / entry["sha256"]
        if not matches(cached, entry):
            cached.unlink(missing_ok=True)
            fetch(cached, model, entry)
        partial = destination.with_name(destination.name + ".part")
        shutil.copyfile(cached, partial)
        os.replace(partial, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=Path("model-lock.json"))
    parser.add_argument("--output", type=Path, default=Path("models"))
    parser.add_argument("--cache-dir", type=Path, help="Builder-local, SHA-256-keyed download cache")
    args = parser.parse_args()
    lock = json.loads(args.lock.read_text())
    if lock["schema_version"] != 1:
        raise ValueError("Unsupported model lock schema")
    groups: dict[str, list[Job]] = defaultdict(list)
    for role, model in lock["models"].items():
        if role in ("", ".", "..") or "/" in role or "\\" in role:
            raise ValueError(f"Unsafe role in model lock: {role}")
        for entry in model["files"]:
            relative = PurePosixPath(entry["path"])
            if not relative.parts or relative.is_absolute() or ".." in relative.parts or "\\" in entry["path"]:
                raise ValueError(f"Unsafe path in model lock: {entry['path']}")
            digest = entry["sha256"]
            if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise ValueError(f"Invalid SHA-256 in model lock: {entry['path']}")
            groups[digest].append((args.output / role / relative, model, entry))
    if args.cache_dir is not None:
        args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    # The build cache mount serializes writers across builds. Each digest also has only one worker here.
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda jobs: download_group(jobs, args.cache_dir), groups.values()))
    (args.output / "manifest.json").write_text(json.dumps(lock, indent=2) + "\n")
    print(f"Verified {sum(len(jobs) for jobs in groups.values())} files in {args.output}", flush=True)


if __name__ == "__main__":
    main()
