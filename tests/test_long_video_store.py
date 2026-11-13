import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from nodes.long_video_store import ProjectStore, fingerprint
from nodes.long_video_plan import plan_windows


def spec(index=0, parent=None):
    return {"window": plan_windows(720)[index],
            "fingerprint": fingerprint({"index": index}), "parent": parent}


def native(store, request):
    path = store.path(f"takes/{request['take_id']}/clean.safetensors")
    path.write_bytes(b"clean av test data")
    preview = path.with_name("native.mp4")
    preview.write_bytes(b"preview")
    store.record_native(request["take_id"], {
        "artifacts": {"latent": store.artifact(path), "native": store.artifact(preview)},
        "sampling": {"selflift_highres_tiling_mode": "global_stream"}})


def test_native_review_binds_immutable_files_and_progress(tmp_path):
    store = ProjectStore(tmp_path, "project")
    take = store.create_take(spec())
    native(store, take)
    assert store.read()["completed_frames"] == 0
    store.review(take["take_id"], True, "right hand holds the plate", "frames 0-294, action, sound and free boundary checked")
    assert store.read()["completed_frames"] == 294
    assert store.read()["windows"][0]["status"] == "accepted"
    with pytest.raises(ValueError, match="immutable"):
        store.write_once(f"takes/{take['take_id']}/request.json", {})


def test_tampered_native_cannot_be_reviewed_or_resumed(tmp_path):
    store = ProjectStore(tmp_path, "project")
    take = store.create_take(spec())
    native(store, take)
    store.path(f"takes/{take['take_id']}/clean.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash"):
        store.review(take["take_id"], True, "observed", "reviewed")


def test_repair_invalidates_real_descendants_and_preserves_old_files(tmp_path):
    store = ProjectStore(tmp_path, "project")
    first = store.create_take(spec())
    native(store, first)
    store.review(first["take_id"], True, "observed", "reviewed")
    second = store.create_take(spec(1, {"take_id": first["take_id"], "history_sha256": "abc"}))
    native(store, second)
    store.review(second["take_id"], True, "observed 2", "reviewed")
    store.request_repair(0, [[2, 5]], True)
    state = store.read()
    assert state["completed_frames"] == 0
    assert all(w["status"] == "invalidated" for w in state["windows"])
    assert store.path(f"takes/{first['take_id']}/clean.safetensors").exists()
    assert state["repair"]["source_take"] == first["take_id"]


@pytest.mark.parametrize('seconds', [False, True])
def test_target_conditioned_repair_rejects_audio_changes_before_revoking_native_reviews(tmp_path, seconds):
    store = ProjectStore(tmp_path, 'project')
    take = store.create_take(dict(spec(), target_audio={'sha256': 'a' * 64}))
    native(store, take)
    store.review(take['take_id'], True, 'Actual state', 'Full actual AV review')
    before = store.read()
    with pytest.raises(ValueError, match='target|Target|complete audio'):
        if seconds:
            store.request_repair_seconds(0, 1, 2, stream='audio')
        else:
            store.request_repair(0, [[2, 5]], False, audio_ranges=[[10, 20]])
    assert store.read() == before


@pytest.mark.parametrize("video_ranges,lock_audio,index", [
    ([[1000, 1001]], True, 1),
    ([[11, 14]], True, 1),
    ([[20, 24], [22, 25]], True, 1),
    ([[20, 24]], False, 1),
    ([[20, 24]], 1, 1),
    ([[20, 24]], "true", 1),
    ([], True, 1),
    (None, True, 1),
    ([None], True, 1),
    ([20], True, 1),
    ([[True, 20]], True, 1),
    ([[20.0, 24]], True, 1),
    ([[20, 24]], True, True),
    ([[20, 24]], True, 1.0),
])
def test_malformed_repair_preserves_entire_accepted_journal(tmp_path, video_ranges, lock_audio, index):
    store = ProjectStore(tmp_path, "project")
    for i in (0, 1):
        take = store.create_take(spec(i))
        native(store, take)
        store.review(take["take_id"], True, f"observed {i}", "full native review")
    before = store.read()
    raw_state = store.path("state.json").read_bytes()
    with pytest.raises(ValueError):
        store.request_repair(index, video_ranges, lock_audio)
    assert store.read() == before
    assert store.path("state.json").read_bytes() == raw_state
    for record in before["windows"]:
        store.verify_accepted(record)


@pytest.mark.parametrize("failure_kind", ["serialization", "partial_io", "publication"])
def test_failed_receipt_write_never_publishes_partial_json_and_can_retry(tmp_path, monkeypatch, failure_kind):
    store = ProjectStore(tmp_path, "project")
    final = store.path("receipt.json")
    value = {"first": "complete", "last": "complete"}
    with monkeypatch.context() as failure:
        if failure_kind == "serialization":
            value["last"] = object()
            error = TypeError
        elif failure_kind == "partial_io":
            open_file = Path.open

            class PartialWriter:
                def __init__(self, stream):
                    self.stream = stream

                def __enter__(self):
                    self.stream.__enter__()
                    return self

                def __exit__(self, *args):
                    return self.stream.__exit__(*args)

                def __getattr__(self, name):
                    return getattr(self.stream, name)

                def write(self, text):
                    self.stream.write(text[:1])
                    self.stream.flush()
                    raise OSError("partial receipt write")

            def partial_open(path, mode="r", *args, **kwargs):
                stream = open_file(path, mode, *args, **kwargs)
                return PartialWriter(stream) if mode == "x" and path.parent == store.root else stream

            failure.setattr(Path, "open", partial_open)
            error = OSError
        else:
            def failed_publish(source, destination):
                raise OSError("receipt publication failed")

            failure.setattr(os, "link", failed_publish)
            error = OSError
        with pytest.raises(error):
            store.write_once("receipt.json", value)
    assert not final.exists()
    assert list(store.root.iterdir()) == []
    receipt = {"first": "complete", "last": "retry"}
    store.write_once("receipt.json", receipt)
    assert json.loads(final.read_text(encoding="utf-8")) == receipt
    with pytest.raises(ValueError, match="immutable"):
        store.write_once("receipt.json", {"changed": True})
    assert json.loads(final.read_text(encoding="utf-8")) == receipt


def test_receipt_publication_does_not_replace_a_concurrent_winner(tmp_path, monkeypatch):
    store = ProjectStore(tmp_path, "project")
    final = store.path("receipt.json")
    winner = {"published": "by another writer"}
    publish = os.link

    def racing_publish(source, destination):
        final.write_text(json.dumps(winner), encoding="utf-8")
        publish(source, destination)

    monkeypatch.setattr(os, "link", racing_publish)
    with pytest.raises(ValueError, match="immutable"):
        store.write_once("receipt.json", {"published": "by this writer"})
    assert json.loads(final.read_text(encoding="utf-8")) == winner
    assert list(store.root.iterdir()) == [final]


@pytest.mark.parametrize("old_lock", [b"", b'{"pid":'])
def test_empty_or_damaged_old_lock_metadata_does_not_block_project(tmp_path, old_lock):
    store = ProjectStore(tmp_path, "project")
    lock = store.path("state.lock")
    lock.write_bytes(old_lock)
    inode = lock.stat().st_ino
    with store.transaction() as state:
        state["status"] = "ready"
    assert store.read()["status"] == "ready"
    assert lock.exists() and lock.stat().st_ino == inode


def test_project_lock_stays_exclusive_until_owner_releases_it(tmp_path):
    first = ProjectStore(tmp_path, "project")
    second = ProjectStore(tmp_path, "project")
    probe = """
import sys, types
from pathlib import Path
root, path, expected = sys.argv[1:]
package = types.ModuleType('nodes')
package.__path__ = [str(Path(root) / 'nodes')]
sys.modules['nodes'] = package
from nodes.long_video_store import _process_lock
try:
    with _process_lock(Path(path)):
        pass
except RuntimeError:
    if expected != 'locked':
        raise
else:
    if expected == 'locked':
        raise AssertionError('another process acquired the active lock')
"""

    def probe_process(expected):
        result = subprocess.run([sys.executable, "-c", probe, str(Path(__file__).resolve().parents[1]),
                                 str(first.path("state.lock")), expected],
                                capture_output=True, text=True, timeout=10)
        assert result.returncode == 0, result.stderr

    with first.transaction() as state:
        lock = first.path("state.lock")
        inode = lock.stat().st_ino
        probe_process("locked")
        with pytest.raises(RuntimeError, match="another process"):
            with second.transaction():
                pytest.fail("a competing transaction acquired the active project lock")
        assert lock.exists() and lock.stat().st_ino == inode
        state["status"] = "first finished"
    with second.transaction() as state:
        assert state["status"] == "first finished"
        state["status"] = "second finished"
    assert first.read()["status"] == "second finished"
    assert lock.exists() and lock.stat().st_ino == inode
    probe_process("unlocked")


def test_partial_lock_metadata_write_releases_kernel_ownership_for_retry(tmp_path, monkeypatch):
    store = ProjectStore(tmp_path, "project")
    write_bytes = os.write
    with monkeypatch.context() as failure:
        def partial_write(fd, data):
            write_bytes(fd, data[:1])
            raise OSError("partial lock metadata write")
        failure.setattr(os, "write", partial_write)
        with pytest.raises(OSError, match="partial lock metadata"):
            with store.transaction():
                pytest.fail("failed lock metadata must stop this attempt")
    assert store.path("state.lock").read_bytes() == b"{"
    with store.transaction() as state:
        state["status"] = "retry succeeded"
    assert store.read()["status"] == "retry succeeded"


def test_export_change_does_not_revoke_native_review(tmp_path):
    store = ProjectStore(tmp_path, "project")
    take = store.create_take(spec())
    native(store, take)
    store.review(take["take_id"], True, "observed", "reviewed")
    store.configure({"total_frames": 1440, "export": {"width": 2560}}, [{"index": 0}])
    assert store.read()["windows"][0]["status"] == "accepted"
    assert store.read()["total_frames"] == 1440


@pytest.mark.parametrize("name", ["../x", "a/b", "a\\b", "", ".", "CON"])
def test_project_name_cannot_escape_root(tmp_path, name):
    with pytest.raises(ValueError):
        ProjectStore(tmp_path, name)
