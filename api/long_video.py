"""Project progress and explicit native content review for the U11 entry."""
from pathlib import Path
from urllib.parse import urlencode
import asyncio
from copy import deepcopy
import json

try:
    from ..nodes.long_video_store import ProjectStore
    from ..nodes.long_video_service import control
except ImportError:
    from nodes.long_video_store import ProjectStore
    from nodes.long_video_service import control

_REGISTERED = False


def media_url(output_root, path):
    relative = Path(path).resolve().relative_to(Path(output_root).resolve())
    return '/view?' + urlencode({'filename': relative.name,
                               'subfolder': relative.parent.as_posix(), 'type': 'output'})


def _preview(store, record):
    artifacts = record.get('artifacts', {})
    asset = artifacts.get('preview') or artifacts.get('native')
    return ({'url': media_url(store.output_root, store.path(asset['name'])),
             'artifact': deepcopy(asset), 'take_id': record['take_id']} if asset else None)


def seam_evidence(store, state, record):
    """Describe the actual delivery join without hashing/loading clean tensors."""
    parent_id = (record.get('parent') or {}).get('take_id')
    if not parent_id:
        return None
    parent = next((w for w in state['windows'] + state.get('superseded', [])
                   if w['take_id'] == parent_id), None)
    if parent is None:
        return {'error': '实际 request.parent 来源不存在；不能用当前相邻窗口替代。'}
    left, right = _preview(store, parent), _preview(store, record)
    if left is None or right is None:
        return {'error': '实际父窗或子窗预览尚未生成。'}
    p, w = parent['window'], record['window']
    e, c = p['local_new_end'], w['local_new_start']
    count = min(4, p['new_end'] - p['new_start'], w['new_end'] - w['new_start'])
    left.update(local_frames=[e - count, e], local_seconds=[(e-count)/24, e/24],
                film_seconds=[(p['new_end']-count)/24, p['new_end']/24])
    right.update(local_frames=[c, c+count], local_seconds=[c/24, (c+count)/24],
                 film_seconds=[w['new_start']/24, (w['new_start']+count)/24])
    evidence = {'parent': left, 'child': right,
                'note': '实际父窗最后交付帧 → 子窗首批净新增帧；相等的历史 latent 不构成可见接缝审批。'}
    f = record.get('forward_frames', 0)
    if f:
        a, b = max(c, c+f-2), min(w['local_new_end'], c+f+2)
        boundary = deepcopy(right)
        boundary.update(local_frames=[a,b], local_seconds=[a/24,b/24],
                        film_seconds=[(w['new_start']+a-c)/24, (w['new_start']+b-c)/24])
        evidence['free_boundary'] = boundary
    return evidence


def public_status(store):
    state = store.read()
    result = {key:state.get(key) for key in ('project_name','status','completed_frames','generated_frames',
                                           'total_frames','stop_requested','error','forward_review_error')}
    result['actual_seconds'] = state['total_frames'] / 24
    result['completed_seconds'] = state['completed_frames'] / 24
    result['generated_seconds'] = state['generated_frames'] / 24
    result['target_seconds'] = state.get('config', {}).get('target_seconds', result['actual_seconds'])
    result['encoding'] = state.get('config', {}).get('export', {}).get('encoding')
    result['busy'] = bool(state.get('active_job'))
    result['windows'] = []
    for record in state['windows']:
        window = {key:record.get(key) for key in ('take_id','status','observed_outgoing_state')}
        window.update(index=record['window']['index'], start=record['window']['new_start']/24,
                      end=record['window']['new_end']/24)
        window['can_retry'] = record['status'] in {'rejected', 'invalidated', 'accepted'}
        pending = state.get('repair') or {}
        window['repair'] = deepcopy(pending if pending.get('index') == record['window']['index'] else record.get('repair'))
        window['local_new_start'] = record['window'].get('local_new_start', 0) / 24
        window['local_new_end'] = record['window'].get('local_new_end', record['window']['new_end']) / 24
        window['free_start_seconds'] = (record['window']['new_start'] + record.get('forward_frames', 0)) / 24
        window['seam'] = seam_evidence(store, state, record)
        if record['status'] == 'accepted':
            try:
                requirement = store.required_forward_review(record)
                if requirement:
                    window['forward_review'] = {**requirement,
                        'review_receipt_exists': bool(record.get('forward_review')),
                        'source_seconds': [n / 24 for n in requirement['source_local_range']],
                        'consumed_seconds': [n / 24 for n in requirement['consumed_local_range']]}
            except ValueError as exc:
                window['forward_review_error'] = str(exc)
        artifacts = record.get('artifacts', {})
        preview = artifacts.get('preview') or artifacts.get('native')
        if preview:
            window['preview_url'] = media_url(store.output_root, store.path(preview['name']))
        audio = artifacts.get('delivery_audio')
        if audio:
            window['audio_url'] = media_url(store.output_root, store.path(audio['name']))
        result['windows'].append(window)
    if state.get('output'):
        result['output_url'] = media_url(store.output_root, state['output'])
    return result


def register_long_video_routes():
    global _REGISTERED
    if _REGISTERED:
        return
    try:
        from aiohttp import web
        from server import PromptServer
        import folder_paths
    except ImportError:
        return
    server = PromptServer.instance
    if server is None:
        return

    async def progress(request):
        try:
            store = ProjectStore(folder_paths.get_output_directory(), request.match_info['project'])
            return web.json_response(public_status(store))
        except (ValueError, OSError) as exc:
            return web.json_response({'error': str(exc)}, status=400)

    async def update(request):
        try:
            store = ProjectStore(folder_paths.get_output_directory(), request.match_info['project'])
            data = await request.json()
            if not isinstance(data, dict) or not isinstance(data.get('action'), str):
                raise ValueError('control requires a JSON object with a string action')
            action = data.pop('action')
            result = await asyncio.to_thread(control, store, action, data)
            status = public_status(store)
            if action == 'review_video_reference':
                status['video_reference_review'] = result['video_reference_review']
            return web.json_response(status)
        except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
            return web.json_response({'error': str(exc)}, status=400)

    async def check(request):
        try:
            from ..nodes.long_video_preflight import preflight
        except ImportError:
            from nodes.long_video_preflight import preflight
        try:
            data = await request.json()
            if not isinstance(data, dict):
                raise ValueError('preflight requires a JSON object with prompt and config')
            result = await asyncio.to_thread(preflight, data.get('prompt'), data.get('config'))
            return web.json_response(result)
        except (ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
            return web.json_response({'error': str(exc)}, status=400)

    async def example(request):
        # This canonical exported contract retains its unapproved source facts.
        path = Path(__file__).parents[1] / 'examples/u11_long_video_scene_contract.en.json'
        return web.json_response(json.loads(path.read_text(encoding='utf-8')))

    async def seam(request):
        try:
            try:
                from ..nodes.long_video_media import seam_contact_sheet
            except ImportError:
                from nodes.long_video_media import seam_contact_sheet
            store = ProjectStore(folder_paths.get_output_directory(), request.match_info['project'])
            state = store.read()
            record = next((w for w in state['windows'] + state.get('superseded', [])
                           if w['take_id'] == request.match_info['take']), None)
            if record is None:
                raise ValueError('Unknown actual take')
            evidence = seam_evidence(store, state, record)
            if not evidence or evidence.get('error'):
                raise ValueError((evidence or {}).get('error', 'First window has no preceding delivery seam'))
            segments = []
            for role in ('parent', 'child', 'free_boundary'):
                if role in evidence:
                    item = evidence[role]
                    segments.append({**item, 'path': str(store.path(item['artifact']['name'])), 'role': role})
            path = await asyncio.to_thread(seam_contact_sheet, segments, store.path('seam_previews'))
            return web.json_response({'image_url': media_url(store.output_root, path), 'evidence': evidence})
        except (ValueError, TypeError, KeyError, OSError, RuntimeError, ImportError) as exc:
            return web.json_response({'error': str(exc)}, status=400)

    # The extra namespace leaves every legal one-segment project name usable.
    server.routes.post('/h3-director-plus/long-video/contract/preflight')(check)
    server.routes.get('/h3-director-plus/long-video/contract/example')(example)
    server.routes.get('/h3-director-plus/long-video/{project}')(progress)
    server.routes.post('/h3-director-plus/long-video/{project}/control')(update)
    server.routes.get('/h3-director-plus/long-video/{project}/seam/{take}')(seam)
    _REGISTERED = True
