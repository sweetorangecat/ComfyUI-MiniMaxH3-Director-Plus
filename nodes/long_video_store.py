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
from .long_video_av import conditioning_geometry, normalize_repair, repair_seconds, slice_fingerprint

_LOCKS = {}
_LOCKS_GUARD = threading.Lock()


class StaleCoordination(ValueError):
    """A legitimate newer journal input superseded a coordinator reservation."""


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

    def validate_parent_binding(self, state, request):
        """Check the source chain while the caller holds this journal's kernel lock."""
        window, parent = request['window'], request.get('parent')
        if window.get('kind') == 'independent_shots':
            index = window.get('index')
            if (state.get('config', {}).get('entry_mode') != 'independent_shots'
                    or type(index) is not int or not 0 <= index < len(state.get('plan', []))
                    or window != state['plan'][index] or parent is not None
                    or window.get('context_frames') != 0 or request.get('forward_frames', 0)):
                raise ValueError('Independent clip must use its exact planned short request without a parent')
            return
        forward = request.get('forward_frames', 0)
        conditioning_geometry(window, forward)
        if not window['context_frames']:
            if parent is not None:
                raise ValueError('A first native window cannot have a continuation parent')
            return
        if not isinstance(parent, dict) or not parent.get('take_id') or not parent.get('history_sha256'):
            raise ValueError('Continuation requires its current accepted parent and consumed AV hash')
        try:
            source = self.active(state, parent['take_id'])
        except ValueError as exc:
            raise StaleCoordination('Current parent source chain changed') from exc
        previous = source['window']
        if (source['status'] != 'accepted' or previous['index'] != window['index'] - 1
                or previous['new_end'] != window['new_start']
                or previous['local_new_end'] != window['source_endpoint']):
            raise StaleCoordination('Continuation parent is no longer the current accepted adjacent source')
        self.verify_accepted(source)
        if forward:
            if not parent.get('forward_review'):
                raise ValueError('Forward AV needs its explicit actual source-tail review')
            if source.get('forward_review') != parent['forward_review']:
                raise StaleCoordination('Actual forward source review changed')
            proof = self.verify_forward_review(source, parent['forward_review'])
            if (proof['requirement']['next_window'] != window
                    or proof['requirement']['forward_frames'] != forward
                    or proof['consumed_av_sha256'] != parent['history_sha256']):
                raise ValueError('Forward review or consumed AV differs from the immutable child binding')

    @staticmethod
    def coordination_snapshot(state, index):
        """Reservation identity, separate from the reusable model recipe.

        Include the selected target and consumed source records, durable retry
        attempts, config/plan, pending repair and stop/job controls. Omit status
        messages, timestamps and the reservation token itself so normal driver
        bookkeeping does not invalidate its own reservation.
        """
        records = [{k: v for k, v in record.items() if k != '_coordination_snapshot'}
                   for record in state['windows'][:index + 1]]
        return fingerprint({'index': index, 'records': records, 'config': state.get('config'),
                            'plan': state.get('plan'), 'attempts': state.get('content_attempts', {}),
                            'repair': state.get('repair'), 'repair_base': state.get('repair_base_fingerprint'),
                            'stop': state['stop_requested'], 'active_job': state.get('active_job')})

    @classmethod
    def check_coordination(cls, state, index, expected_snapshot):
        if expected_snapshot is None or cls.coordination_snapshot(state, index) != expected_snapshot:
            raise StaleCoordination('Coordinator inputs or source chain changed; re-evaluate current project state')

    def coordinator_status(self, status, index, expected_snapshot, **fields):
        with self.transaction() as state:
            self.check_coordination(state, index, expected_snapshot)
            state.update(status=status, **fields)

    def create_take(self, spec, *, require_current_parent=False, expected_snapshot=None):
        take_id = uuid.uuid4().hex
        request = {**deepcopy(spec), "take_id": take_id, "project_name": self.name}
        index = int(spec["window"]["index"])
        with self.transaction() as state:
            if require_current_parent:
                self.check_coordination(state, index, expected_snapshot)
                if state.get('active_job'):
                    raise StaleCoordination('A project already has a queued or running window')
                self.validate_parent_binding(state, request)
            from .long_video_parent import verify_parent_reference
            verify_parent_reference(self, request, state=state)
            if (request.get('repair') or {}).get('video_reference') is not None:
                from .long_video_repair import verify_video_reference
                recipe = normalize_repair(request['window'], request['repair'], request.get('forward_frames', 0))
                verify_video_reference(self, recipe['video_reference'], request['window'], state=state)
            if index > len(state["windows"]):
                raise ValueError("A take cannot skip a window")
            record = {**request, "status": "queued", "artifacts": {}, "exports": {}}
            if index == len(state["windows"]):
                state["windows"].append(record)
            else:
                state.setdefault("superseded", []).append(deepcopy(state["windows"][index]))
                state["windows"][index] = record
            state["status"] = "queued"
            if require_current_parent:
                if request.get('repair') and state.get('repair') == request['repair']:
                    state.pop('repair', None)
                    state.pop('repair_base_fingerprint', None)
                token = self.coordination_snapshot(state, index)
                request['_coordination_snapshot'] = record['_coordination_snapshot'] = token
            self.write_once(f"takes/{take_id}/request.json", request)
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
                                         if (w["status"] in {"awaiting_review", "accepted"}
                                             or w["status"] == "rejected" and
                                             {'native', 'preview'} & set(w.get('artifacts', {})))), default=0)

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
        if type(accepted) is not bool or not isinstance(scope, str) or not scope.strip():
            raise ValueError("Record the actual native review scope")
        if accepted and (not isinstance(observed_outgoing_state, str) or not observed_outgoing_state.strip()):
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

    def invalidate_from(self, index, reason, *, expected_snapshot=None):
        with self.transaction() as state:
            if expected_snapshot is not None:
                self.check_coordination(state, index, expected_snapshot)
            self._invalidate(state, index, reason)
            return self.coordination_snapshot(state, index)

    @staticmethod
    def _invalidate(state, index, reason):
        independent = state.get('config', {}).get('entry_mode') == 'independent_shots'
        for record in state["windows"][index:index+1] if independent else state["windows"][index:]:
            record.update(status="invalidated", invalidation_reason=reason)
        state["output"] = ""
        state["status"] = "ready"
        ProjectStore.progress(state)

    def _repair_source(self, state, index, take_id=None):
        if state.get('config', {}).get('entry_mode') == 'independent_shots':
            raise ValueError('分镜式长视频请使用“只重做本段”；局部原生 AV 返修仅用于连续镜头。')
        if type(index) is not int:
            raise ValueError("Repair window index must be an integer")
        if state.get('active_job'):
            raise ValueError('Wait for the running window before repairing its source chain')
        if not 0 <= index < len(state['windows']):
            raise ValueError('Unknown repair window')
        record = state['windows'][index]
        if take_id is not None and take_id != record['take_id']:
            raise ValueError('Selected repair source is stale; select the current decoded take')
        artifacts = record.get('artifacts', {})
        if (record['status'] not in {'accepted', 'rejected'}
                or not {'native', 'preview'} & set(artifacts)
                or not {'clean_video', 'latent'} & set(artifacts)):
            raise ValueError('Repair needs the current decoded accepted or rejected native source')
        self.verify(artifacts)
        from .long_video_parent import verify_parent_reference
        verify_parent_reference(self, record, state=state)
        if record['status'] == 'accepted':
            self.verify_accepted(record)
        elif record.get('review'):
            self.verify({'review': record['review']})
        return record

    def _queue_repair(self, state, record, recipe):
        if record.get('target_audio') is not None and (recipe.get('lock_audio') is not True or recipe.get('audio_ranges')):
            raise ValueError('Target-conditioned repair inherits complete audio; changing audio requires a new base generation')
        index = record['window']['index']
        state['repair'] = {'index': index, 'source_take': record['take_id'], **recipe}
        state['repair_base_fingerprint'] = record.get('base_fingerprint', record['fingerprint'])
        self._invalidate(state, index, 'Native repair replaces an actual parent source')

    def request_repair(self, index, video_ranges=None, lock_audio=True, *, audio_ranges=None,
                       video_weights=None, take_id=None, video_reference=None):
        """Legacy video ranges plus explicit audio ranges / temporal video weights."""
        with self.transaction() as state:
            record = self._repair_source(state, index, take_id)
            raw = {'video_ranges': ([] if video_ranges is None and
                    (audio_ranges is not None or video_weights is not None) else video_ranges),
                   'lock_audio': lock_audio}
            if audio_ranges is not None:
                raw['audio_ranges'] = audio_ranges
            if video_weights is not None:
                raw['video_weights'] = video_weights
            if video_reference is not None:
                raw['video_reference'] = video_reference
            recipe = normalize_repair(record['window'], raw, record.get('forward_frames', 0))
            if video_reference is not None:
                from .long_video_repair import verify_video_reference
                verify_video_reference(self, recipe['video_reference'], record['window'], state=state)
            self._queue_repair(state, record, recipe)

    def request_repair_seconds(self, index, start_seconds, end_seconds, stream='video', *, take_id=None):
        """Atomic normal-user repair: film seconds mapped against the current take."""
        with self.transaction() as state:
            record = self._repair_source(state, index, take_id)
            recipe = repair_seconds(record['window'], start_seconds, end_seconds, stream,
                                    record.get('forward_frames', 0))
            self._queue_repair(state, record, recipe)

    def request_retry(self, index, *, replace_accepted=False, take_id=None):
        """A content retry changes its durable attempt; automatic recovery does not."""
        if type(replace_accepted) is not bool:
            raise ValueError('replace_accepted must be an explicit boolean')
        with self.transaction() as state:
            if state.get('active_job'):
                raise ValueError('Wait for the running window before replacing a source take')
            if type(index) is not int or not 0 <= index < len(state['windows']):
                raise ValueError('Unknown retry window')
            record = state['windows'][index]
            if take_id is not None and take_id != record['take_id']:
                raise ValueError('Selected retry source is stale')
            if record['status'] == 'accepted':
                if not replace_accepted or take_id != record['take_id']:
                    raise ValueError('Replacing an accepted parent requires replace_accepted:true and its current take_id')
                self.verify_accepted(record)
            elif record['status'] not in {'rejected', 'invalidated'}:
                raise ValueError('Only a rejected or invalidated window needs a new full take')
            attempts = state.setdefault('content_attempts', {})
            attempts[str(index)] = attempts.get(str(index), record.get('content_attempt', 0)) + 1
            if state.get('repair', {}).get('index') == index:
                state.pop('repair', None)
                state.pop('repair_base_fingerprint', None)
            self._invalidate(state, index, 'Explicitly retry native content without bounded repair')

    @staticmethod
    def _forward_requirement(state, record):
        index = record['window']['index'] + 1
        plan = state.get('plan', [])
        if index >= len(plan):
            return None
        window = plan[index]
        contract = state.get('config', {}).get('scene_contract', {})
        overrides = contract.get('window_overrides', [])
        f = next((item.get('forward_frames', 0) for item in overrides
                  if (item.get('new_start'), item.get('new_end')) ==
                  (window['new_start'], window['new_end'])), 0)
        if not f:
            return None
        previous = record['window']
        if (window['index'] != index or previous['new_end'] != window['new_start']
                or previous['local_new_end'] != window['source_endpoint']):
            raise ValueError('Forward source must be the actual adjacent delivery parent')
        geometry = conditioning_geometry(window, f, previous['sample_frames'])
        return {'next_index': index, 'next_window': deepcopy(window), **geometry}

    def required_forward_review(self, record):
        """Return the next planned actual tail requirement; users never enter indices."""
        state = self.read()
        current = self.active(state, record['take_id'])
        return self._forward_requirement(state, current)

    def verify_forward_review(self, record, expected=None):
        """Revalidate receipt, actual artifacts, geometry and all consumed AV bytes."""
        from .long_video_runtime import load_clean
        state = self.read()
        current = self.active(state, record['take_id'])
        if current['status'] != 'accepted':
            raise ValueError('Forward source must be the current accepted adjacent parent')
        requirement = self._forward_requirement(state, current)
        artifact = current.get('forward_review')
        if requirement is None or not artifact or expected is not None and expected != artifact:
            raise ValueError('Actual source tail needs an explicit matching forward review')
        self.verify_accepted(current)
        self.verify({'forward_review': artifact})
        receipt = json.loads(self.path(artifact['name']).read_text(encoding='utf-8'))
        if (receipt.get('take_id') != current['take_id'] or receipt.get('requirement') != requirement
                or receipt.get('source_artifacts') != current['artifacts']
                or receipt.get('source_review') != current['review']
                or not isinstance(receipt.get('scope'), str) or not receipt['scope'].strip()
                or not isinstance(receipt.get('observed_state'), str) or not receipt['observed_state'].strip()):
            raise ValueError('Forward review does not bind this actual source, range and observed free-phase state')
        window = requirement['next_window']
        actual = slice_fingerprint(load_clean(self, current['take_id']), window['source_endpoint'],
                                   window['context_frames'], requirement['forward_frames'])
        if receipt.get('consumed_av_sha256') != actual:
            raise ValueError('Forward review consumed AV hash differs from the actual current parent')
        return receipt

    def review_forward(self, take_id, scope, observed_state):
        """Explicit immutable review of generated surplus, separate from delivery approval."""
        from .long_video_runtime import load_clean
        if not isinstance(scope, str) or not scope.strip():
            raise ValueError('Record the actual source-tail review scope')
        if not isinstance(observed_state, str) or not observed_state.strip():
            raise ValueError('Record the observed actual forward endpoint state, not an expected state object')
        with self.transaction() as state:
            if state.get('active_job'):
                raise ValueError('Wait for the running window before reviewing its source tail')
            record = self.active(state, take_id)
            if record['status'] != 'accepted':
                raise ValueError('Forward review requires the current accepted source delivery')
            requirement = self._forward_requirement(state, record)
            if requirement is None:
                raise ValueError('The actual next planned window does not request a source tail')
            if not {'clean_video', 'clean_audio', 'native', 'request'}.issubset(record['artifacts']):
                raise ValueError('Forward review requires actual clean AV and its decoded native preview')
            self.verify_accepted(record)
            window = requirement['next_window']
            receipt = {'take_id': take_id, 'requirement': requirement, 'scope': scope,
                       'observed_state': observed_state, 'source_artifacts': deepcopy(record['artifacts']),
                       'source_review': deepcopy(record['review']),
                       'consumed_av_sha256': slice_fingerprint(load_clean(self, take_id),
                           window['source_endpoint'], window['context_frames'], requirement['forward_frames'])}
            if record.get('forward_review'):
                try:
                    previous = self.verify_forward_review(record)
                    if {k: v for k, v in previous.items() if k != 'time'} == receipt:
                        return
                except ValueError:
                    pass  # A fresh explicit review may replace an unusable old proof.
            receipt['time'] = time.time()
            path = self.write_once(f'reviews/forward-{uuid.uuid4().hex}.json', receipt)
            record['forward_review'] = self.artifact(path)
            self._invalidate(state, window['index'], 'Actual reviewed forward source proof changed')

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
                or review.get('source_artifacts') != record['artifacts']
                or review.get('observed_outgoing_state') != record.get('observed_outgoing_state')):
            raise ValueError('Native acceptance does not bind these actual artifacts')
