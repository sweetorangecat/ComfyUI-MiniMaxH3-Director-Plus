import asyncio
import sys
from types import SimpleNamespace

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

import api.long_video as routes
import nodes.long_video_service as service
from nodes.long_video_store import ProjectStore


def test_progress_and_review_routes_bind_actual_project_and_receipts(tmp_path, monkeypatch):
    registry = web.RouteTableDef()
    monkeypatch.setattr(routes, '_REGISTERED', False)
    monkeypatch.setitem(sys.modules, 'server', SimpleNamespace(
        PromptServer=SimpleNamespace(instance=SimpleNamespace(routes=registry))))
    monkeypatch.setitem(sys.modules, 'folder_paths', SimpleNamespace(
        get_output_directory=lambda: str(tmp_path)))
    started = []
    monkeypatch.setattr(service, '_start', lambda store: started.append(store.name))
    routes.register_long_video_routes()
    # Registration is idempotent in the real plugin import path.
    routes.register_long_video_routes()
    assert len(registry) == 2
    store = ProjectStore(tmp_path, 'scene')
    store.set_status('new', total_frames=720)
    take = store.create_take({'window': {'index': 0, 'new_start': 0, 'new_end': 294}})
    preview = store.path(f"takes/{take['take_id']}/原生 preview.mp4")
    preview.write_bytes(b'preview fixture')
    store.record_native(take['take_id'], {'artifacts': {'preview': store.artifact(preview)}})

    async def exercise():
        app = web.Application()
        app.add_routes(registry)
        async with TestClient(TestServer(app)) as client:
            response = await client.get('/h3-director-plus/long-video/scene')
            assert response.status == 200
            data = await response.json()
            assert data['actual_seconds'] == 30
            assert data['windows'][0]['preview_url'].startswith('/view?filename=')
            response = await client.post('/h3-director-plus/long-video/scene/control', json={
                'action': 'review', 'take_id': take['take_id'], 'accepted': True,
                'scope': 'Native frames checked', 'observed_outgoing_state': 'Same cup in right hand'})
            assert response.status == 200
            data = await response.json()
            assert data['completed_frames'] == 294
            assert data['windows'][0]['status'] == 'accepted'
            assert started == ['scene']
            response = await client.post('/h3-director-plus/long-video/scene/control', json={'action': 'stop'})
            assert response.status == 200
            assert (await response.json())['stop_requested'] is True
            response = await client.post('/h3-director-plus/long-video/scene/control', json={'action': 'unknown'})
            assert response.status == 400
            response = await client.get('/h3-director-plus/long-video/CON')
            assert response.status == 400

    asyncio.run(exercise())
    store.verify_accepted(store.read()['windows'][0])
