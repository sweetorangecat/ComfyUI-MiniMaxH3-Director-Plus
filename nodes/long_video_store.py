"""Disk project journal. Native takes and review receipts are immutable.

This module has no ComfyUI dependency; it never stores RGB batches in state.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import errno
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import uuid

from .longfilm_media import atomic_write_json, sha256_file, safe_project_dir
from .long_video_av import validate_repair

_LOCKS = {}
_LOCKS_GUARD = threading.Lock()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@contextmanager
def _process_lock(path):
    """Keep one persistent lock inode; kernel ownership survives torn metadata."""
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    acquired = False
    try:
        try:
            if os.name == "nt":
                import msvcrt
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                raise RuntimeError("Long-video project is being updated by another process") from exc
            raise
        acquired = True
        # PID text is diagnostic only. Ownership is held by this descriptor.
        os.lseek(fd, 0, os.SEEK_SET)
        os.ftruncate(fd, 0)
        os.write(fd, json.dumps({"pid": os.getpid()}).encode("utf-8"))
        yield
    finally:
        try:
            if acquired:
                if os.name == "nt":
                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


class ProjectStore:
    def __init__(self, output_root, name):
        # Reuse the reviewed path rules, with a distinct native-AV namespace.
        self.root = safe_project_dir(output_root, name, namespace='h3_native_av')
        self.output_root = Path(output_root).resolve()
        if not self.root.resolve().is_relative_to(self.output_root) or self.root.is_symlink():
            raise ValueError("Project directory escapes output root")
        self.root = self.root.resolve()
        self.name = name
        with _LOCKS_GUARD:
            self._lock = _LOCKS.setdefault(str(self.root), threading.RLock())

    def path(self, relative):
        path = self.root / str(relative)
        if Path(str(relative)).is_absolute() or not path.resolve().is_relative_to(self.root):
            raise ValueError("Artifact path escapes project")
        for part in [path, *path.parents]:
            if part == self.root.parent:
                break
            if part.is_symlink() or (hasattr(part, "is_junction") and part.is_junction()):
                raise ValueError("Project artifacts cannot be symlinks or junctions")
        return path

    @contextmanager
    def transaction(self):
        with self._lock, _process_lock(self.path("state.lock")):
            state = self.read()
            yield state
            state["updated_at"] = time.time()
            atomic_write_json(self.path("state.json"), state)

    def read(self):
        path = self.path("state.json")
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return {"schema_version": 1, "project_name": self.name, "status": "new",
                "windows": [], "completed_frames": 0, "generated_frames": 0,
                "total_frames": 0, "stop_requested": False, "jobs": [], "output": ""}

    def write_once(self, relative, value):
        path = self.path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise ValueError("Refusing to overwrite an immutable receipt")
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        created = False
        try:
            with temporary.open("x", encoding="utf-8") as f:
                created = True
                json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
                f.flush()
                os.fsync(f.fileno())
            os.link(temporary, path)
        except FileExistsError as exc:
            raise ValueError("Refusing to overwrite an immutable receipt") from exc
        finally:
            if created:
                temporary.unlink(missing_ok=True)
        return path

    def artifact(self, path):
        path = self.path(Path(path).relative_to(self.root))
        return {"name": path.relative_to(self.root).as_posix(), "sha256": sha256_file(path),
                "bytes": path.stat().st_size}

    def verify(self, artifacts):
        for item in artifacts.values():
            path = self.path(item["name"])
            if not path.is_file() or sha256_file(path) != item["sha256"]:
                raise ValueError("Artifact missing or hash changed: " + item["name"])

    def configure(self, config, plan):
        with self.transaction() as state:
            if state.get('active_job'):
                raise ValueError('Wait for the current window before changing project settings')
            previous = state.get('config') or {}
            state.update(config=deepcopy(config), plan=deepcopy(plan),
                         total_frames=config["total_frames"])
            if len(state['windows']) > len(plan):
                state.setdefault('superseded', []).extend(deepcopy(state['windows'][len(plan):]))
                del state['windows'][len(plan):]
                self.progress(state)
            if (previous.get('total_frames'), previous.get('export_fingerprint')) != (
                    config['total_frames'], config.get('export_fingerprint')):
                state['output'] = ''
            state.pop('error', None)
            # Changes are compared per window by the coordinator. A new export
            # size or an appended target does not revoke native reviews here.

    def create_take(self, spec):
        take_id = uuid.uuid4().hex
        request = {**deepcopy(spec), "take_id": take_id, "project_name": self.name}
        self.write_once(f"takes/{take_id}/request.json", request)
        index = int(spec["window"]["index"])
        with self.transaction() as state:
            if index > len(state["windows"]):
                raise ValueError("A take cannot skip a window")
            record = {**request, "status": "queued", "artifacts": {}, "exports": {}}
            if index == len(state["windows"]):
                state["windows"].append(record)
            else:
                state.setdefault("superseded", []).append(deepcopy(state["windows"][index]))
                state["windows"][index] = record
            state["status"] = "queued"
        return request

    @staticmethod
    def active(state, take_id):
        record = next((w for w in state["windows"] if w["take_id"] == take_id), None)
        if record is None or record["status"] == "invalidated":
            raise ValueError("This take no longer belongs to the current source chain")
        return record

    @staticmethod
    def progress(state):
        cursor = 0
        for w in state["windows"]:
            if w["status"] != "accepted" or w["window"]["new_start"] != cursor:
                break
            cursor = w["window"]["new_end"]
        state["completed_frames"] = cursor
        state["generated_frames"] = max((w["window"]["new_end"] for w in state["windows"]
                                         if w["status"] in {"awaiting_review", "accepted"}), default=0)

    def record_clean(self, take_id, receipt):
        self.verify(receipt["artifacts"])
        with self.transaction() as state:
            record = self.active(state, take_id)
            record.update(deepcopy(receipt), status="sampled")
            state["status"] = "decoding"

    def record_native(self, take_id, receipt):
        self.verify(receipt["artifacts"])
        with self.transaction() as state:
            record = self.active(state, take_id)
            record["artifacts"].update(deepcopy(receipt["artifacts"]))
            record.update({k: deepcopy(v) for k, v in receipt.items() if k != "artifacts"})
            record["status"] = "awaiting_review"
            state["status"] = "awaiting_review"
            self.progress(state)

    def review(self, take_id, accepted, observed_outgoing_state, scope):
        if type(accepted) is not bool or not str(scope).strip():
            raise ValueError("Record the actual native review scope")
        if accepted and not str(observed_outgoing_state).strip():
            raise ValueError("Continuation needs the observed actual outgoing state")
        with self.transaction() as state:
            record = self.active(state, take_id)
            if record["status"] != "awaiting_review":
                raise ValueError("Only the current decoded native take can be reviewed")
            self.verify(record["artifacts"])
            review = {"take_id": take_id, "accepted": accepted, "scope": str(scope),
                      "observed_outgoing_state": str(observed_outgoing_state),
                      "source_artifacts": record["artifacts"], "time": time.time()}
            path = self.write_once(f"reviews/{uuid.uuid4().hex}.json", review)
            record.update(status="accepted" if accepted else "rejected",
                          observed_outgoing_state=str(observed_outgoing_state),
                          review=self.artifact(path))
            state["status"] = "ready" if accepted else "rejected"
            self.progress(state)

    def invalidate_from(self, index, reason):
        with self.transaction() as state:
            self._invalidate(state, index, reason)

    @staticmethod
    def _invalidate(state, index, reason):
        for record in state["windows"][index:]:
            record.update(status="invalidated", invalidation_reason=reason)
        state["output"] = ""
        state["status"] = "ready"
        ProjectStore.progress(state)

    def request_repair(self, index, video_ranges, lock_audio=True):
        if type(index) is not int:
            raise ValueError("Repair window index must be an integer")
        with self.transaction() as state:
            if state.get('active_job'):
                raise ValueError('Wait for the running window before repairing its source chain')
            if not 0 <= index < len(state["windows"]):
                raise ValueError("Unknown repair window")
            record = state["windows"][index]
            ranges = validate_repair(record["window"], {"video_ranges": video_ranges, "lock_audio": lock_audio})
            self.verify(record["artifacts"])
            state["repair"] = {"index": index, "source_take": record["take_id"],
                               "video_ranges": ranges, "lock_audio": True}
            self._invalidate(state, index, "Native repair replaces an actual parent source")

    def set_status(self, status, **fields):
        with self.transaction() as state:
            state.update(status=status, **fields)

    def stop(self):
        with self.transaction() as state:
            state["stop_requested"] = True
            state["status"] = "stopping" if state.get("active_job") else "stopped"

    def resume(self):
        with self.transaction() as state:
            state["stop_requested"] = False
            state.pop("error", None)

    def verify_accepted(self, record):
        self.verify(record['artifacts'])
        self.verify({'review': record['review']})
        review = json.loads(self.path(record['review']['name']).read_text(encoding='utf-8'))
        if (review.get('accepted') is not True or review.get('take_id') != record['take_id']
                or review.get('source_artifacts') != record['artifacts']):
            raise ValueError('Native acceptance does not bind these actual artifacts')
