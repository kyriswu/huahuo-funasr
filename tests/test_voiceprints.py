import json

import numpy as np
import pytest

from app.voiceprints import StoreError, VoiceprintExists, VoiceprintStore, match_voiceprint, normalize


def test_identity_threshold_and_empty_gallery(vector):
    record = {"speaker_id": "alice", "name": "Alice", "embedding": vector * 10}
    assert match_voiceprint(vector, [record], 0.5)["speaker_id"] == "alice"
    other = np.roll(vector, 1)
    rejected = match_voiceprint(other, [record], 0.5)
    assert not rejected["matched"] and rejected["speaker_id"] is None
    assert rejected["score"] == 0
    assert match_voiceprint(vector, [], 0.5)["score"] is None


def test_persistence_replace_delete_and_no_embedding_in_metadata(tmp_path, vector):
    store = VoiceprintStore(tmp_path, "model-a")
    created = store.put("alice", "张三", vector, 4)
    assert "embedding" not in created
    with pytest.raises(VoiceprintExists):
        store.put("alice", "Alice", vector, 5)
    reloaded = VoiceprintStore(tmp_path, "model-a")
    assert reloaded.records()[0]["name"] == "张三"
    replaced = reloaded.put("alice", "Alice", np.roll(vector, 1), 5, replace=True)
    assert replaced["created_at"] == created["created_at"]
    assert not list(tmp_path.glob("*.tmp"))
    assert reloaded.records()[0]["embedding"][1] == 1
    reloaded.delete("alice")
    assert reloaded.records() == []
    with pytest.raises(KeyError):
        reloaded.delete("alice")


def test_model_mismatch_and_corruption_are_visible(tmp_path, vector):
    store = VoiceprintStore(tmp_path, "model-a")
    store.put("alice", "Alice", vector, 4)
    with pytest.raises(StoreError, match="Incompatible"):
        VoiceprintStore(tmp_path, "model-b")
    path = tmp_path / "alice.json"
    path.write_text('{"broken": true}')
    with pytest.raises(StoreError):
        store.records()
    assert json.loads(path.read_text()) == {"broken": True}


@pytest.mark.parametrize("identity", ["../escape", "a/b", ".hidden", "中文", "a" * 65])
def test_invalid_identity_never_creates_files(tmp_path, vector, identity):
    store = VoiceprintStore(tmp_path, "model-a")
    with pytest.raises(ValueError):
        store.put(identity, "test", vector, 4)
    assert list(tmp_path.glob("*.json")) == []


@pytest.mark.parametrize("embedding", [np.zeros(192), np.full(192, np.nan), np.ones(191)])
def test_invalid_embeddings_are_rejected(embedding):
    with pytest.raises(ValueError):
        normalize(embedding)
