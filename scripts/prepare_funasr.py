#!/usr/bin/env python3
"""Package the pinned upstream Python source, excluding unrelated symlinks."""

import argparse
import hashlib
from http.client import IncompleteRead
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

COMMIT = "be264e9382b4fef735536f3d278ee14514098a1a"
SHA256 = "843cdc8b26d7a350a2367ccefb36f9c0b6b43efb787b360453df4b48b8e17caa"
URL = f"https://codeload.github.com/modelscope/FunASR/tar.gz/{COMMIT}"


def download_archive(archive):
    for attempt in range(3):
        archive.seek(0)
        archive.truncate()
        try:
            with urlopen(URL, timeout=120) as response:
                shutil.copyfileobj(response, archive, length=1024 * 1024)
            return
        except (ConnectionError, IncompleteRead, TimeoutError, URLError) as error:
            if isinstance(error, HTTPError) and not (error.code in {408, 429} or 500 <= error.code < 600):
                raise
            if attempt == 2:
                raise
            print(f"FunASR source download interrupted, retrying ({type(error).__name__})", flush=True)
            time.sleep(2 ** (attempt + 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", type=Path, help="Path to a local copy of the pinned upstream tar.gz")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    archive_file = tempfile.TemporaryFile() if args.archive is None else args.archive.open("rb")
    with archive_file as archive:
        if args.archive is None:
            download_archive(archive)
        archive.seek(0)
        if hashlib.file_digest(archive, "sha256").hexdigest() != SHA256:
            raise ValueError("FunASR source archive checksum mismatch")
        archive.seek(0)
        with tarfile.open(fileobj=archive, mode="r:gz") as source:
            for member in source:
                path = PurePosixPath(member.name)
                if path.parts[0] != f"FunASR-{COMMIT}" or ".." in path.parts or path.is_absolute():
                    raise ValueError("Unexpected path in upstream archive")
                relative = PurePosixPath(*path.parts[1:])
                if not member.isfile():
                    continue
                if not (str(relative).startswith("funasr/") or str(relative) in {"setup.py", "README.md", "LICENSE"}):
                    continue
                destination = args.output / str(relative)
                destination.parent.mkdir(parents=True, exist_ok=True)
                with source.extractfile(member) as stream, destination.open("wb") as target:
                    shutil.copyfileobj(stream, target)
    if not (args.output / "setup.py").is_file():
        raise ValueError("Upstream package is incomplete")
    (args.output / "UPSTREAM.json").write_text(json.dumps({"repository": "https://github.com/modelscope/FunASR", "commit": COMMIT, "archive_sha256": SHA256}, indent=2) + "\n")
    print(f"Prepared verified FunASR source at {COMMIT}")


if __name__ == "__main__":
    main()
