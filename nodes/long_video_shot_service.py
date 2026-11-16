"""Disk-backed editorial clips sharing the native service's queue and journal."""
from copy import deepcopy
import json
from pathlib import Path
import uuid

from .long_video_service import Coordinator, delivery_identity
from .long_video_shots import ENTRY_MODE, plan_shots
from .long_video_graph import freeze_assets, output_canvas, OUTPUT_TYPES
from .long_video_store import ProjectStore, StaleCoordination, fingerprint
from .long_video_media import encoder_capabilities, resolve_encoding
from .longfilm_media import sha256_file


def _code():
    root = Path(__file__).parents[1]
    files = [root / '__init__.py', *sorted((root / 'nodes').rglob('*.py'))]
    return {str(file.relative_to(root)): sha256_file(file) for file in files}


def _assets(graph):
    import folder_paths
    from .long_video_assets import freeze_native_assets, freeze_export_assets
    settings = next(n['inputs'] for n in graph.values() if n['class_type'] == 'MiniMaxH3DirectorPlus')
    if settings.get('sampling_method') == 'selflift':
        from .long_video_service import current_assets
        native = current_assets(graph)
    else:
        implicit = freeze_native_assets(graph, folder_paths)
        native = {'records': {**freeze_assets(graph, folder_paths), **implicit['records']},
                  'choices': implicit['choices']}
    export = freeze_export_assets(graph, folder_paths)
    return {'records': {**native['records'], **export['records']},
            'choices': {'native': native['choices'], 'export': export['choices']}}


def _choices(graph):
    """Recheck implicit selection/absence without rehashing large checkpoints."""
    import folder_paths
    from .long_video_assets import native_asset_choices, export_asset_choices
    settings = next(n['inputs'] for n in graph.values() if n['class_type'] == 'MiniMaxH3DirectorPlus')
    if settings.get('sampling_method') == 'selflift':
        from .long_video_service import current_native_choices
        native = current_native_choices(graph)
    else:
        native = native_asset_choices(graph, folder_paths)
    return {'native': native, 'export': export_asset_choices(graph, folder_paths)}


def configure_shots(store, base, raw, *, asset_freezer=_assets, encoder_checker=encoder_capabilities):
    graph, config = deepcopy(base), deepcopy(raw)
    settings = next(n['inputs'] for n in graph.values() if n['class_type'] == 'MiniMaxH3DirectorPlus')
    total = int(config['target_seconds'] * 24)
    plan = plan_shots(settings['prompt'], config['target_seconds'], settings)
    if config.get('entry_mode') != ENTRY_MODE or config['shot_plan'] != plan:
        raise ValueError('Independent clip plan differs from the original Director prompt')
    if config['project_name'] != store.name:
        raise ValueError('Independent clip project identity differs from its journal')
    output = next(n['inputs'] for n in graph.values() if n['class_type'] in OUTPUT_TYPES)
    encoding = resolve_encoding(output)
    encoder_checker()
    bundle = asset_freezer(graph)
    bundle = bundle if 'records' in bundle else {'records': bundle, 'choices': None}
    code = _code()
    recipe = deepcopy(graph)
    for node in recipe.values():
        node.pop('_meta', None)
        if node['class_type'] == 'MiniMaxH3DirectorPlus':
            node['inputs'].update(prompt='', duration=15, seed=0)
        if node['class_type'] == 'RandomNoise':
            node['inputs']['noise_seed'] = 0
        if node['class_type'] in OUTPUT_TYPES:
            node['inputs'].pop('filename_prefix', None)
            node['inputs'].pop('long_video', None)
    config.update(base_graph=graph, total_frames=total, actual_seconds=total / 24,
        seed=int(settings.get('seed', 0)), code=code, assets=bundle['records'],
        asset_choices=bundle['choices'],
        asset_stats={item['path']: [Path(item['path']).stat().st_size, Path(item['path']).stat().st_mtime_ns]
                     for item in bundle['records'].values()},
        generation_key=fingerprint({'recipe': recipe, 'assets': bundle, 'code': code}),
        export={'target': list(output_canvas(settings)), 'encoding': encoding})
    config['export_fingerprint'] = fingerprint(config['export'])
    store.configure(config, plan)
    return config


def recover_shot(store, take_id):
    path = store.path(f'takes/{take_id}/shot_receipt.json')
    if not path.is_file():
        return False
    receipt = json.loads(path.read_text(encoding='utf-8'))
    store.verify(receipt['artifacts'])
    saved = json.loads(store.path(f'takes/{take_id}/request.json').read_text(encoding='utf-8'))
    if receipt['request_sha256'] != fingerprint(saved):
        raise ValueError('Independent clip receipt differs from its immutable request')
    receipt['artifacts']['shot_receipt'] = store.artifact(path)
    store.record_native(take_id, receipt)
    return True


def _delivery_canvas(config, records):
    """Keep the actual original short output, including harmless SR rounding."""
    sizes = {(r['technical']['width'], r['technical']['height']) for r in records}
    if len(sizes) != 1:
        raise ValueError('Independent clip output sizes differ; re-generate the mismatched clip')
    width, height = sizes.pop()
    minimum = config['export']['target']
    if width < minimum[0] or height < minimum[1]:
        raise ValueError('Original short output is below the selected delivery resolution')
    return width, height


class ShotCoordinator(Coordinator):
    def _step(self):
        state = self.store.read()
        job = state.get('active_job')
        if job and self.backend.status(job['prompt_id']) not in {'running', 'queued'}:
            record = self.store.active(state, job['take_id'])
            if record['status'] in {'queued', 'running'}:
                recover_shot(self.store, record['take_id'])
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
        if config['code'] != _code():
            raise ValueError('Short-video code changed; re-queue the original Director before resuming')
        if config['asset_choices'] is not None and config['asset_choices'] != _choices(config['base_graph']):
            raise ValueError('Automatic model selection changed; re-queue the original Director before resuming')
        for name, stat in config['asset_stats'].items():
            file = Path(name)
            if not file.is_file() or [file.stat().st_size, file.stat().st_mtime_ns] != stat:
                raise ValueError('Original reference/model changed; re-queue the original Director: ' + name)
        parts, accepted = [], []
        for window in state['plan']:
            index = window['index']
            snapshot = self.store.coordination_snapshot(state, index)
            attempt = state.get('content_attempts', {}).get(str(index), 0)
            seed = (config['seed'] + index * 1009 + attempt * 1000003) % (2**64)
            key = fingerprint({'generation': config['generation_key'], 'window': window, 'seed': seed})
            record = state['windows'][index] if index < len(state['windows']) else None
            if record and record['fingerprint'] != key and record['status'] != 'invalidated':
                self.store.invalidate_from(index, 'Independent clip inputs changed', expected_snapshot=snapshot)
                return True
            if record is None or record['status'] == 'invalidated':
                request = self.store.create_take({'window': window, 'prompt': window['prompt'],
                    'seed': seed, 'fingerprint': key, 'parent': None, 'content_attempt': attempt},
                    require_current_parent=True, expected_snapshot=snapshot)
                return self.schedule(request, 'native', expected_snapshot=request['_coordination_snapshot'])
            if record['status'] in {'queued', 'running'}:
                return self.schedule(record, 'native', expected_snapshot=snapshot)
            if record['status'] in {'awaiting_review', 'rejected'}:
                self.store.coordinator_status(record['status'], index, snapshot)
                return False
            if record['status'] != 'accepted':
                raise ValueError('Unknown independent clip state: ' + record['status'])
            self.store.verify_accepted(record)
            export = {'video': record['artifacts']['preview'], 'audio': record['artifacts']['delivery_audio'],
                      'receipt': record['artifacts']['shot_receipt']}
            if record['exports'].get(config['export_fingerprint']) != export:
                with self.store.transaction() as current:
                    self.store.check_coordination(current, index, snapshot)
                    self.store.active(current, record['take_id'])['exports'][config['export_fingerprint']] = export
                record['exports'][config['export_fingerprint']] = export
                # Publication changed the coordination snapshot for later clips.
                state = self.store.read()
            parts.append({'video': str(self.store.path(export['video']['name'])),
                          'audio': str(self.store.path(export['audio']['name'])),
                          'frames': window['new_end'] - window['new_start']})
            accepted.append(record)
        identity = delivery_identity(config, accepted)
        if state.get('output') and state.get('final_source') == identity:
            self.store.verify(state['final_artifacts'])
            self.store.set_status('complete')
            return False
        with self.store.transaction() as current:
            if delivery_identity(current['config'], current['windows']) != identity:
                raise StaleCoordination('Accepted independent clips changed before assembly')
            if current['stop_requested']:
                current['status'] = 'stopped'
                return False
            current['status'] = 'assembling'
        final = self.store.path(f"final/seed{config['seed']}_f{config['total_frames']}_{uuid.uuid4().hex}.mp4")
        canvas = _delivery_canvas(config, accepted)
        report = self.assembler(parts, final, config['total_frames'], canvas,
                                encoding=config['export']['encoding'])
        with self.store.transaction() as current:
            if delivery_identity(current['config'], current['windows']) != identity:
                raise StaleCoordination('Independent clips changed during assembly')
            current.update(output=str(final), assembly=report, final_source=identity,
                final_artifacts={'video': self.store.artifact(final)},
                status='stopped' if current['stop_requested'] else 'complete')
        return False
