"""Disk-backed native AV coordinator using ComfyUI's ordinary prompt queue.

The output node only starts this driver. It never waits for another queued job
inside Comfy's execution thread. Native review is a persisted explicit action.
"""
from __future__ import annotations

import asyncio
from copy import deepcopy
import json
from pathlib import Path
import threading
import time
import uuid

from .long_video_graph import (build_window_graph, generation_settings, roles,
                               output_canvas, freeze_assets, apply_window_reference_policy)
from .long_video_plan import (target_frame_count, validate_contract, protected_intervals, plan_windows,
                              compile_window, validate_window_overrides)
from .long_video_store import ProjectStore, StaleCoordination, fingerprint
from .long_video_media import assemble, resolve_encoding, encoder_capabilities
from .long_video_assets import (freeze_native_assets, freeze_export_assets,
                                native_asset_choices, export_asset_choices)

_DRIVERS = {}
_DRIVER_LOCK = threading.Lock()
_SAMPLER_INCOMPLETE = 'Sampler did not publish a clean checkpoint'


def _code_identity(names):
    from .longfilm_media import sha256_file
    root = Path(__file__).parent
    return {name: sha256_file(root / (name + '.py')) for name in names}


def current_assets(graph):
    import folder_paths
    records = freeze_assets(graph, folder_paths)
    # SelfLift's automatic upscaler / matched LoRAs do not have explicit UI
    # loader nodes. Their concrete files are part of generation provenance too.
    from .director import _selflift_dependency_report
    from .longfilm_media import sha256_file
    report = _selflift_dependency_report()
    if not report.get('ready'):
        raise ValueError('SelfLift dependencies missing: ' + ', '.join(report.get('missing', [])))
    for item in report.get('required_assets', []):
        path = ((item.get('path') or item.get('full_path')) if isinstance(item, dict) else
                (folder_paths.get_full_path('loras', item) or folder_paths.get_full_path('latent_upscale_models', item)))
        if path and Path(path).is_file():
            file = Path(path).resolve()
            records['selflift:' + str(file)] = {'path': str(file), 'sha256': sha256_file(file),
                                              'bytes': file.stat().st_size}
    implicit = freeze_native_assets(graph, folder_paths)
    records.update(implicit['records'])
    return {'records': records, 'choices': {
        'automatic': implicit['choices'],
        'selflift': {key: report.get(key) for key in ('ready', 'required_assets')}}}


def current_native_choices(graph):
    import folder_paths
    from .director import _selflift_dependency_report
    report = _selflift_dependency_report()
    return {'automatic': native_asset_choices(graph, folder_paths),
            'selflift': {key: report.get(key) for key in ('ready', 'required_assets')}}


def _export_source_size(graph):
    from .sampling_experiments import plan_selflift_dimensions
    _, director, _, _, _ = roles(graph)
    width, height = output_canvas(graph[director]['inputs'])
    canvas = plan_selflift_dimensions(width, height, 13, 32, 28)
    return canvas['second_stage_width'], canvas['second_stage_height']


def current_export_assets(graph):
    import folder_paths
    return freeze_export_assets(graph, folder_paths, source_size=_export_source_size(graph))


def current_export_choices(graph):
    import folder_paths
    return export_asset_choices(graph, folder_paths, source_size=_export_source_size(graph))


def _asset_bundle(value):
    # Explicit injectors may provide just file records; production freezers
    # also record selection/absence so a newly installed adapter is detected.
    return value if 'records' in value else {'records': value, 'choices': None}


def _asset_stats(records):
    return {value['path']: [Path(value['path']).stat().st_size, Path(value['path']).stat().st_mtime_ns]
            for value in records.values()}


def configure_project(store, base_graph, raw, asset_freezer=current_assets,
                      export_asset_freezer=current_export_assets, encoder_checker=encoder_capabilities):
    config = deepcopy(raw)
    total = target_frame_count(config.get('target_seconds', 30))
    if config.get('project_name', store.name) != store.name:
        raise ValueError('Project identity differs from its disk journal')
    if config.get('stop_condition', 'target') not in {'target', 'manual'}:
        raise ValueError('Choose a finite target or explicit manual continuation')
    contract = validate_contract(config.get('scene_contract'), total)
    plan = plan_windows(total, protected_intervals=protected_intervals(contract))
    validate_window_overrides(contract, plan)
    graph, director, _, _, output = roles(base_graph)
    output_settings = graph[output]['inputs']
    # Fail before asset hashing, take creation or any model/queue work.
    encoding = resolve_encoding(output_settings)
    encoder_checker()
    if (float(output_settings.get('frame_rate', 24)) != 24
            or output_settings.get('pingpong', False) or output_settings.get('crop_to_audio', False)):
        raise ValueError('Long-video delivery uses 24 fps, with no pingpong or audio-driven frame removal')
    if not output_settings.get('save_output', True):
        raise ValueError('Long-video checkpoints require saved output')
    # A config node is not a sampling ancestor. Omit it from the retained
    # per-window graph to avoid recursively launching another coordinator.
    graph[output]['inputs'].pop('long_video', None)
    graph = roles(graph)[0]
    native_assets = _asset_bundle(asset_freezer(graph))
    assets = native_assets['records']
    export_assets = _asset_bundle(export_asset_freezer(graph))
    native_code = _code_identity(['long_video_plan','long_video_av','long_video_runtime','long_video_graph',
                                 'long_video_references',
                                 'director','guide','schema','prompting','voice_guard','resolution',
                                 'sampling_experiments','two_stage','performance','two_stage_assets','vram_budget',
                                 'additional_loras','long_video_assets','long_video_entry'])
    generation = fingerprint({'graph': generation_settings(graph), 'assets': native_assets, 'code': native_code})
    export_settings = {'target': list(output_canvas(graph[director]['inputs'])), 'encoding': encoding,
                       'output': {k:v for k,v in output_settings.items()
                                  if k not in {'images','guide','audio','audio_override','long_video'}},
                       'director': {k:graph[director]['inputs'].get(k) for k in
                                    ('postprocess_mode','vosr2_profile','audio_loudness','rtx_quality','ai_upscale_model')},
                       'assets': export_assets,
                       'code': _code_identity(['stream_output','vosr2_profiles','vosr2_compat','video_sr',
                                              'long_video_media','long_video_assets','color_guard'])}
    config.update(total_frames=total, actual_seconds=total / 24, scene_contract=contract,
                  base_graph=graph, assets=assets, generation_key=generation,
                  export=export_settings, export_fingerprint=fingerprint(export_settings),
                  seed=int(graph[director]['inputs'].get('seed', 0)), native_code=native_code,
                  asset_stats=_asset_stats(assets), native_asset_choices=native_assets['choices'],
                  export_asset_stats=_asset_stats(export_assets['records']),
                  export_asset_choices=export_assets['choices'])
    store.configure(config, plan)
    return config


def consumed_history(store, parent, window):
    from .long_video_runtime import load_clean
    from .long_video_av import slice_fingerprint
    required = store.required_forward_review(parent)
    forward = required['forward_frames'] if required and required['next_window'] == window else 0
    return slice_fingerprint(load_clean(store, parent['take_id']),
                             window['source_endpoint'], window['context_frames'], forward)


def delivery_identity(config, records):
    sources = []
    for record in records:
        export = record.get('exports', {}).get(config['export_fingerprint'])
        if record['status'] != 'accepted' or export is None:
            return None
        sources.append({key: record[key] for key in ('take_id', 'fingerprint', 'window', 'review', 'artifacts')})
        sources[-1]['export'] = export
    return fingerprint({'generation': config['generation_key'], 'export': config['export_fingerprint'],
                        'frames': config['total_frames'], 'sources': sources})


class ComfyQueue:
    def __init__(self, server=None):
        if server is None:
            from server import PromptServer
            server = PromptServer.instance
        self.server = server

    def submit(self, prompt_id, graph):
        async def submit():
            import execution
            self.server.node_replace_manager.apply_replacements(graph)
            valid = await execution.validate_prompt(prompt_id, graph, None)
            if not valid[0]:
                raise ValueError('Long-video window validation failed: ' + json.dumps(
                    {'error': valid[1], 'node_errors': valid[3]}, ensure_ascii=False))
            number = self.server.number
            self.server.number += 1
            self.server.prompt_queue.put((number, prompt_id, graph,
                {'create_time': int(time.time() * 1000)}, valid[2], {}))
        # This waits in the background driver only. A timeout could report a
        # failed submission while the coroutine later enqueues the same job.
        asyncio.run_coroutine_threadsafe(submit(), self.server.loop).result()

    def status(self, prompt_id):
        queue = self.server.prompt_queue
        running, queued = queue.get_current_queue()
        if any(item[1] == prompt_id for item in running):
            return 'running'
        if any(item[1] == prompt_id for item in queued):
            return 'queued'
        entry = queue.get_history(prompt_id=prompt_id).get(prompt_id)
        if entry is None:
            return 'missing'
        return 'completed' if entry.get('status', {}).get('status_str') == 'success' else 'failed'


class Coordinator:
    def __init__(self, store, backend, history_provider=consumed_history, assembler=assemble):
        self.store, self.backend = store, backend
        self.history_provider, self.assembler = history_provider, assembler

    def reconcile_job(self):
        state = self.store.read()
        job = state.get('active_job')
        if not job:
            return False
        status = self.backend.status(job['prompt_id'])
        if status in {'queued', 'running'}:
            return True
        with self.store.transaction() as current:
            if current.get('active_job') != job:
                raise StaleCoordination('Queue ownership changed before reconciliation')
            take = deepcopy(self.store.active(current, job['take_id']))
            current.pop('active_job', None)
            current['jobs'].append({**job, 'queue_status': status, 'finished_at': time.time()})
            index = take['window']['index']
            expected_snapshot = self.store.coordination_snapshot(current, index)
        stage = job['stage']
        if (stage == 'native' and take['status'] in {'queued','running'}
                and self.store.path(f"takes/{take['take_id']}/clean_receipt.json").is_file()):
            from .long_video_runtime import load_clean
            try:
                load_clean(self.store, take['take_id'])
            except ValueError:
                # Only a changed reservation converts this known recovery
                # failure into stale work. Unchanged corrupt evidence still fails.
                self.store.check_coordination(self.store.read(), index, expected_snapshot)
                raise
            recovered = self.store.read()
            try:
                take = deepcopy(self.store.active(recovered, job['take_id']))
            except ValueError as exc:
                raise StaleCoordination('Source chain changed during clean recovery') from exc
            expected_snapshot = self.store.coordination_snapshot(recovered, index)
        # A completed decode may already have a negative content review before
        # its queue entry is reconciled. Rejection retains the technical source;
        # only an explicit retry or repair may replace that reviewed take.
        usable = (take['status'] in {'sampled','awaiting_review','accepted','rejected'} if stage in {'native','preview'}
                  else self.store.read()['config']['export_fingerprint'] in take['exports'])
        if not usable:
            if stage == 'native':
                # Partial prepared files are diagnostics. A new take on resume
                # avoids overwriting them after an interrupted sampler.
                expected_snapshot = self.store.invalidate_from(index, _SAMPLER_INCOMPLETE,
                                                                 expected_snapshot=expected_snapshot)
            if status == 'missing':
                # Server restart removed the queue entry. Explicit resume can
                # retry this step immediately; it cannot duplicate a live job.
                return False
            self.store.coordinator_status('failed', index, expected_snapshot,
                error=f'{stage} job {status}; checkpoints retained, resume to retry')
            return None
        return False

    def schedule(self, request, stage, *, expected_snapshot=None):
        prompt_id = str(uuid.uuid4())
        # A resumed journal record also contains status, media and reviews.
        # Only the immutable generation spec belongs in the runtime payload.
        request = json.loads(self.store.path(f"takes/{request['take_id']}/request.json").read_text(encoding='utf-8'))
        state = self.store.read()
        index = request['window']['index']
        if expected_snapshot is None:
            expected_snapshot = self.store.coordination_snapshot(state, index)
        request['export_fingerprint'] = state['config']['export_fingerprint']
        graph = build_window_graph(state['config']['base_graph'], request, stage)
        graph_file = self.store.write_once(f'jobs/{prompt_id}.json', graph)
        job = {'prompt_id': prompt_id, 'take_id': request['take_id'], 'stage': stage,
               'graph': self.store.artifact(graph_file), 'submitted_at': time.time()}
        with self.store.transaction() as current:
            self.store.check_coordination(current, index, expected_snapshot)
            if current.get('active_job'):
                raise StaleCoordination('A project already has a queued or running window')
            if current['stop_requested']:
                return False
            ProjectStore.active(current, request['take_id'])
            self.store.validate_parent_binding(current, request)
            current['active_job'] = job
            current['status'] = 'enhancing' if stage == 'finish' else 'generating'
        try:
            self.backend.submit(prompt_id, graph)
        except Exception:
            with self.store.transaction() as current:
                current.pop('active_job', None)
            raise
        return True

    def step(self):
        try:
            return self._step()
        except StaleCoordination:
            # A newer control/config/proof is already durable. Re-read it on
            # the next driver iteration; do not overwrite its status or error.
            return True

    def _step(self):
        busy = self.reconcile_job()
        if busy is True:
            return True
        if busy is None:
            return False
        state = self.store.read()
        if state['stop_requested']:
            self.store.coordinator_status('stopped', 0, self.store.coordination_snapshot(state, 0))
            return False
        config = state['config']
        validate_window_overrides(config['scene_contract'], state['plan'])
        if config.get('native_code') != _code_identity(config['native_code']):
            raise ValueError('Generation code changed; re-queue the original U11 to validate this project against the new version')
        export_code = config.get('export', {}).get('code', {})
        if (not config.get('export', {}).get('encoding')
                or not export_code or export_code != _code_identity(export_code)):
            raise ValueError('Export code changed; re-queue the original U11 to refresh the export fingerprint before continuing')
        for name, stat in config.get('asset_stats', {}).items():
            file = Path(name)
            if not file.is_file() or [file.stat().st_size, file.stat().st_mtime_ns] != stat:
                raise ValueError('Referenced asset changed; re-queue the project to fingerprint the new file: ' + name)
        if (config.get('native_asset_choices') is not None
                and config['native_asset_choices'] != current_native_choices(config['base_graph'])):
            raise ValueError('Resolved native model selection changed; re-queue the original U11 to fingerprint the new stack')
        for name, stat in config.get('export_asset_stats', {}).items():
            file = Path(name)
            if not file.is_file() or [file.stat().st_size, file.stat().st_mtime_ns] != stat:
                raise ValueError('Enhancement asset changed; re-queue the original U11 to refresh exports: ' + name)
        if (config.get('export_asset_choices') is not None
                and config['export_asset_choices'] != current_export_choices(config['base_graph'])):
            raise ValueError('Resolved enhancement weights changed; re-queue the original U11 to refresh exports')
        parts, accepted_records = [], []
        parent = None
        for window in state['plan']:
            index = window['index']
            expected_snapshot = self.store.coordination_snapshot(state, index)
            context = '' if parent is None else parent['observed_outgoing_state']
            forward = next((item.get('forward_frames', 0) for item in config['scene_contract'].get('window_overrides', [])
                            if (item['new_start'], item['new_end']) == (window['new_start'], window['new_end'])), 0)
            proof = None
            if forward:
                try:
                    if parent is None:
                        raise ValueError('Forward AV requires an actual accepted adjacent parent')
                    proof = self.store.verify_forward_review(parent)
                    if (proof['requirement']['next_window'] != window
                            or proof['requirement']['forward_frames'] != forward):
                        raise ValueError('Actual forward review does not match this next window')
                except ValueError as exc:
                    self.store.coordinator_status('awaiting_forward_review', index, expected_snapshot,
                                                  forward_review_error=str(exc))
                    return False
                context = proof['observed_state']
            compiled = compile_window(config['scene_contract'], window, context)
            active, director, *_ = roles(config['base_graph'])
            settings = active[director]['inputs']
            settings['prompt'] = compiled['prompt']
            bindings = apply_window_reference_policy(settings, compiled, active)
            history = None
            if parent is not None:
                try:
                    history_hash = self.history_provider(self.store, parent, window)
                except ValueError:
                    # The production loader may fail while a legitimate retry
                    # removes this parent. Convert only a changed reservation;
                    # unchanged corrupt/missing clean evidence must still fail.
                    self.store.check_coordination(self.store.read(), index, expected_snapshot)
                    raise
                history = {'take_id': parent['take_id'], 'history_sha256': history_hash}
            if proof is not None:
                history['forward_review'] = deepcopy(parent['forward_review'])
            current = state
            attempt = current.get('content_attempts', {}).get(str(index), 0)
            seed = (config['seed'] + index * 1009 + attempt * 1000003) % (2**64)
            # Freeze the whole compiler result in request.json, but only model
            # inputs affect reuse. Audit/mapped metadata can describe a beat's
            # later endpoint when a target is extended; it is not sampled here.
            model_compiled = {key: compiled[key] for key in
                              ('prompt', 'dialogue_ids', 'reference_selection', 'voice_policy', 'forward_frames')}
            base_key = fingerprint({'generation': config['generation_key'], 'window': window,
                                    'compiled': model_compiled, 'reference_bindings': bindings, 'seed': seed,
                                    'parent': history})
            records = current['windows']
            record = records[index] if index < len(records) else None
            repair = current.get('repair')
            repair = repair if repair and repair['index'] == index else None
            if repair and current.get('repair_base_fingerprint', base_key) != base_key:
                raise ValueError('Repair generation inputs or current parent changed; select the current source and request repair again')
            if (not repair and record and record.get('base_fingerprint') == base_key
                    and record.get('repair') and (record['status'] != 'invalidated'
                    or record.get('invalidation_reason') == _SAMPLER_INCOMPLETE)):
                # Recovery and accepted repaired takes keep the same immutable
                # bounds/locks. The base recipe is separate from the repair key.
                repair = record['repair']
            key = fingerprint({'base': base_key, 'repair': repair}) if repair else base_key
            if record and record['fingerprint'] != key:
                expected_snapshot = self.store.invalidate_from(index,
                    'Actual generation inputs or consumed history changed', expected_snapshot=expected_snapshot)
                record = {**record, 'status': 'invalidated',
                          'invalidation_reason': 'Actual generation inputs or consumed history changed'}
            if record is None or record['status'] == 'invalidated':
                spec = {'window': window, **deepcopy(compiled), 'reference_bindings': bindings, 'seed': seed,
                        'content_attempt': attempt, 'base_fingerprint': base_key,
                        'fingerprint': key, 'parent': history,
                        'asset_revision': config['generation_key']}
                if repair and repair['index'] == index:
                    spec['repair'] = repair
                request = self.store.create_take(spec, require_current_parent=True,
                                                 expected_snapshot=expected_snapshot)
                return self.schedule(request, 'native', expected_snapshot=request['_coordination_snapshot'])
            if record['status'] in {'queued','running'}:
                return self.schedule(record, 'native', expected_snapshot=expected_snapshot)
            if record['status'] == 'sampled':
                self.store.verify(record['artifacts'])
                return self.schedule(record, 'preview', expected_snapshot=expected_snapshot)
            if record['status'] in {'awaiting_review', 'rejected'}:
                self.store.coordinator_status(record['status'], index, expected_snapshot)
                return False
            if record['status'] != 'accepted':
                raise ValueError('Unrecognized native take state: ' + record['status'])
            self.store.verify_accepted(record)
            try:
                if self.store.required_forward_review(record) is not None:
                    self.store.verify_forward_review(record)
            except ValueError as exc:
                # Review surplus before spending VOSR2 work: a bad source tail
                # may require replacing this still-accepted parent delivery.
                self.store.coordinator_status('awaiting_forward_review', index, expected_snapshot,
                                              forward_review_error=str(exc))
                return False
            export = record['exports'].get(config['export_fingerprint'])
            if export is None:
                return self.schedule(record, 'finish', expected_snapshot=expected_snapshot)
            self.store.verify({k:export[k] for k in ('video','audio')})
            if export.get('receipt'):
                self.store.verify({'receipt': export['receipt']})
            parts.append({'video': str(self.store.path(export['video']['name'])),
                          'audio': str(self.store.path(export['audio']['name'])),
                          'frames': window['new_end'] - window['new_start']})
            accepted_records.append(record)
            parent = record
        source_identity = delivery_identity(config, accepted_records)
        reuse = (state.get('output') and state.get('final_artifacts')
                 and state.get('final_source') == source_identity)
        if reuse:
            self.store.verify(state['final_artifacts'])
        else:
            with self.store.transaction() as current:
                if current['stop_requested']:
                    current['status'] = 'stopped'
                    return False
                if delivery_identity(current['config'], current['windows']) != source_identity:
                    return True
                current['status'] = 'assembling'
            final = self.store.path(f"final/seed{config['seed']}_f{config['total_frames']}_{uuid.uuid4().hex}.mp4")
            report = self.assembler(parts, final, config['total_frames'], config['export']['target'],
                                    encoding=config['export']['encoding'])
        with self.store.transaction() as current:
            # A review/repair can legally revoke sources while assembly runs.
            # Publish only against the exact accepted source chain, atomically.
            if delivery_identity(current['config'], current['windows']) != source_identity:
                return True
            if not reuse:
                current.update(output=str(final), assembly=report, final_source=source_identity,
                               final_artifacts={'video': self.store.artifact(final)})
            current['status'] = ('stopped' if current['stop_requested'] else
                                 'awaiting_extension' if config.get('stop_condition') == 'manual' else 'complete')
        return False


def _drive(store, wake, setup=None):
    try:
        if setup:
            graph, config = setup
            store.set_status('preparing')
            configure_project(store, graph, config)
        driver = Coordinator(store, ComfyQueue())
        while True:
            wake.clear()
            if driver.step():
                # Wait only in the daemon, never the GPU executor.
                wake.wait(1)
                continue
            with _DRIVER_LOCK:
                if wake.is_set():
                    continue
                _DRIVERS.pop(str(store.root), None)
                return
    except Exception as exc:
        self_message = f'{type(exc).__name__}: {exc}'
        store.set_status('failed', error=self_message)
    finally:
        with _DRIVER_LOCK:
            entry = _DRIVERS.get(str(store.root))
            if entry and entry[0] is threading.current_thread():
                _DRIVERS.pop(str(store.root), None)


def launch(base_graph, config, output_root=None):
    if output_root is None:
        import folder_paths
        output_root = folder_paths.get_output_directory()
    store = ProjectStore(output_root, config['project_name'])
    _start(store, (deepcopy(base_graph), deepcopy(config)))
    return store.read()


def _start(store, setup=None):
    with _DRIVER_LOCK:
        key = str(store.root)
        if key in _DRIVERS and _DRIVERS[key][0].is_alive():
            if setup is not None:
                raise ValueError('Project is already running; use its progress controls')
            _DRIVERS[key][1].set()
            return
        if setup is not None:
            # Clear an old stop synchronously at launch. A newer stop during
            # asset hashing/configuration must survive the preparation phase.
            store.resume()
            store.set_status('preparing')
        wake = threading.Event()
        thread = threading.Thread(target=_drive, args=(store, wake, setup), daemon=True,
                                  name='h3-native-av-' + store.name)
        _DRIVERS[key] = (thread, wake)
        thread.start()


def control(store, action, data):
    if not isinstance(data, dict):
        raise ValueError('Long-video control data must be an object')
    if action == 'stop':
        store.stop()
    elif action == 'resume':
        if 'config' not in store.read():
            raise ValueError('Queue the original U11 with long video enabled before resuming this project')
        store.resume()
        _start(store)
    elif action == 'review':
        store.review(data['take_id'], data['accepted'], data.get('observed_outgoing_state',''), data.get('scope',''))
        if data['accepted']:
            store.resume()
            _start(store)
    elif action == 'review_forward':
        if set(data) - {'take_id', 'scope', 'observed_state'} or 'take_id' not in data:
            raise ValueError('Forward review accepts take_id, actual scope and observed_state only')
        store.review_forward(data['take_id'], data.get('scope'), data.get('observed_state'))
        store.resume()
        _start(store)
    elif action == 'repair':
        normal = {'index', 'take_id', 'start_seconds', 'end_seconds', 'stream'}
        advanced = {'index', 'take_id', 'video_ranges', 'audio_ranges', 'video_weights', 'lock_audio'}
        seconds = bool({'start_seconds', 'end_seconds', 'stream'} & set(data))
        allowed = normal if seconds else advanced
        if set(data) - allowed or 'index' not in data:
            raise ValueError('Unknown or mixed repair fields; use film seconds and stream, or bounded native ranges')
        if seconds:
            if not {'start_seconds', 'end_seconds'}.issubset(data):
                raise ValueError('Seconds repair requires start_seconds and end_seconds')
            store.request_repair_seconds(data['index'], data['start_seconds'], data['end_seconds'],
                                        data.get('stream', 'video'), take_id=data.get('take_id'))
        else:
            store.request_repair(data['index'], data.get('video_ranges'), data.get('lock_audio', True),
                                 audio_ranges=data.get('audio_ranges'), video_weights=data.get('video_weights'),
                                 take_id=data.get('take_id'))
        store.resume()
        _start(store)
    elif action == 'retry':
        if set(data) - {'index', 'replace_accepted', 'take_id'} or 'index' not in data:
            raise ValueError('Unknown retry fields')
        store.request_retry(data['index'], replace_accepted=data.get('replace_accepted', False), take_id=data.get('take_id'))
        store.resume()
        _start(store)
    else:
        raise ValueError('Unknown long-video control action')
    return store.read()
