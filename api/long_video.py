"""Project progress and explicit native content review for the U11 entry."""
from pathlib import Path
from urllib.parse import urlencode
import asyncio

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


def public_status(store):
    state = store.read()
    result = {key:state.get(key) for key in ('project_name','status','completed_frames','generated_frames',
                                           'total_frames','stop_requested','error')}
    result['actual_seconds'] = state['total_frames'] / 24
    result['completed_seconds'] = state['completed_frames'] / 24
    result['windows'] = []
    for record in state['windows']:
        window = {key:record.get(key) for key in ('take_id','status','observed_outgoing_state')}
        window.update(index=record['window']['index'], start=record['window']['new_start']/24,
                      end=record['window']['new_end']/24)
        window['can_retry'] = record['status'] == 'invalidated' and bool(record.get('repair'))
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
            await asyncio.to_thread(control, store, data['action'], data)
            return web.json_response(public_status(store))
        except (ValueError, KeyError, OSError, RuntimeError) as exc:
            return web.json_response({'error': str(exc)}, status=400)

    server.routes.get('/h3-director-plus/long-video/{project}')(progress)
    server.routes.post('/h3-director-plus/long-video/{project}/control')(update)
    _REGISTERED = True
