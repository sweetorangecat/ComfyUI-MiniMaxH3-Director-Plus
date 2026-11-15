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
    assert len(registry) == 5
    store = ProjectStore(tmp_path, 'scene')
    store.set_status('new', total_frames=720)
    take = store.create_take({'window': {'index': 0, 'new_start': 0, 'new_end': 294}})
    # Real Coordinator requests persist explicit parent:None on the first take.
    with store.transaction() as state:
        state['windows'][0]['parent'] = None
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
            for malformed in ([], None, 42, 'stop'):
                response = await client.post('/h3-director-plus/long-video/scene/control', json=malformed)
                assert response.status == 400
            response = await client.post('/h3-director-plus/long-video/scene/control', data='{bad',
                                         headers={'Content-Type': 'application/json'})
            assert response.status == 400
            response = await client.get('/h3-director-plus/long-video/CON')
            assert response.status == 400

    asyncio.run(exercise())
    store.verify_accepted(store.read()['windows'][0])


def test_seam_uses_immutable_request_parent_and_first_net_new_not_context(tmp_path):
    store = ProjectStore(tmp_path, 'seam')
    def record(take, index, a, b, c, e):
        preview = store.path(f'{take}.mp4')
        preview.write_bytes(b'fixture')
        return {'take_id': take, 'status': 'accepted',
                'window': {'index': index, 'new_start': a, 'new_end': b,
                           'local_new_start': c, 'local_new_end': e, 'sample_frames': 328},
                'artifacts': {'native': store.artifact(preview)}}
    parent = record('actual-parent', 0, 0, 294, 0, 294)
    replacement = record('replacement-parent', 0, 0, 294, 0, 294)
    child = record('child', 1, 294, 549, 39, 294)
    child.update(parent={'take_id': 'actual-parent'}, forward_frames=17)
    with store.transaction() as state:
        state.update(windows=[replacement, child], superseded=[parent])
    result = routes.public_status(store)
    seam = result['windows'][1]['seam']
    assert seam['parent']['take_id'] == 'actual-parent'
    assert seam['parent']['local_frames'] == [290, 294]
    assert seam['parent']['film_seconds'] == [290 / 24, 294 / 24]
    assert seam['child']['local_frames'] == [39, 43]
    assert seam['child']['film_seconds'] == [294 / 24, 298 / 24]
    assert seam['free_boundary']['local_frames'] == [54, 58]
    assert 'actual-parent.mp4' in seam['parent']['url']


def test_control_route_strips_action_before_strict_seconds_repair(tmp_path, monkeypatch):
    registry = web.RouteTableDef()
    monkeypatch.setattr(routes, '_REGISTERED', False)
    monkeypatch.setitem(sys.modules, 'server', SimpleNamespace(
        PromptServer=SimpleNamespace(instance=SimpleNamespace(routes=registry))))
    monkeypatch.setitem(sys.modules, 'folder_paths', SimpleNamespace(get_output_directory=lambda: str(tmp_path)))
    received = []
    monkeypatch.setattr(routes, 'control', lambda store, action, data: received.append((action, data)))
    routes.register_long_video_routes()
    async def exercise():
        app = web.Application()
        app.add_routes(registry)
        async with TestClient(TestServer(app)) as client:
            data = {'action': 'repair', 'index': 1, 'take_id': 'take',
                    'start_seconds': 13.2, 'end_seconds': 13.8, 'stream': 'video'}
            assert (await client.post('/h3-director-plus/long-video/test/control', json=data)).status == 200
            assert received == [('repair', {k:v for k,v in data.items() if k != 'action'})]
            artifact = {'name': 'reviews/video-reference-test.json', 'sha256': 'a' * 64, 'bytes': 123}
            monkeypatch.setattr(routes, 'control', lambda *_: {'video_reference_review': artifact})
            response = await client.post('/h3-director-plus/long-video/test/control', json={
                'action': 'review_video_reference', 'source_take': 'reviewed', 'source_steps': [87, 97],
                'scope': 'Actual unused frames', 'observed_state': 'Open laptop'})
            assert response.status == 200
            assert (await response.json())['video_reference_review'] == artifact
            assert (await client.post('/h3-director-plus/long-video/contract/preflight', json=[])).status == 400
            assert (await client.post('/h3-director-plus/long-video/contract/preflight', json={'prompt': []})).status == 400
            response = await client.get('/h3-director-plus/long-video/contract/example')
            assert response.status == 200
            example = await response.json()
            assert example['not_ready_for_video'] is True
            response = await client.get('/h3-director-plus/long-video/example')
            assert (await response.json())['project_name'] == 'example'
            from test_long_video_preflight import fixture
            graph,cfg = fixture()
            for meta in (None,[],{'h3_long_video_reference_source_files':[]}):
                graph['10']['_meta'] = meta
                response = await client.post('/h3-director-plus/long-video/contract/preflight',json={'prompt':graph,'config':cfg})
                assert response.status == 400
            graph,cfg = fixture()
            for key,value in (('enabled','yes'),('project_name',[])):
                wrong = {**cfg,key:value}
                response = await client.post('/h3-director-plus/long-video/contract/preflight',json={'prompt':graph,'config':wrong})
                assert response.status == 400
            assert not (tmp_path / 'h3_native_av' / 'test' / 'state.json').exists()
            monkeypatch.setattr(routes,'ProjectStore',lambda *_: pytest.fail('preflight touched the journal'))
            graph,cfg = fixture()
            response = await client.post('/h3-director-plus/long-video/contract/preflight',json={'prompt':graph,'config':cfg})
            assert response.status == 200
            result = await response.json()
            assert result['queued'] is False and result['assets_approved'] is False
            assert result['source_facts']['not_ready_for_video'] is True
            assert result['window_count'] == 3
    asyncio.run(exercise())
