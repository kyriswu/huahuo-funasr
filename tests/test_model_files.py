import hashlib
import json

import pytest

from app.model_files import verify_models


def test_local_snapshot_integrity_and_missing_files(tmp_path):
    root = tmp_path / "models"
    lock = {"schema_version": 1, "models": {}}
    for role in ("asr", "vad", "punc", "speaker"):
        directory = root / role
        directory.mkdir(parents=True)
        content = role.encode()
        (directory / "model.pt").write_bytes(content)
        lock["models"][role] = {
            "repo": "test/" + role, "revision": "v1", "files": [
                {"path": "model.pt", "size": len(content), "sha256": hashlib.sha256(content).hexdigest()}
            ],
        }
    path = tmp_path / "lock.json"
    path.write_text(json.dumps(lock))
    (root / "manifest.json").write_text(json.dumps(lock))
    _, first_id = verify_models(root, path)
    _, second_id = verify_models(root, path)
    assert first_id == second_id and "test/speaker@v1:" in first_id
    (root / "asr" / "model.pt").write_bytes(b"bad")
    with pytest.raises(ValueError, match="checksum"):
        verify_models(root, path)
    (root / "asr" / "model.pt").unlink()
    with pytest.raises(ValueError, match="Missing"):
        verify_models(root, path)
