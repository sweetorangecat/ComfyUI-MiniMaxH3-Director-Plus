"""No-queue preflight, immutable policy propagation and content retry semantics."""
from copy import deepcopy
import json

import pytest

from nodes.long_video_plan import compile_window, plan_windows, validate_contract
from nodes.long_video_store import ProjectStore
from nodes.long_video_service import Coordinator
import nodes.long_video_service as service
from nodes.long_video_graph import build_window_graph
from test_long_video_service import graph, config, configure_project, setup, complete, approve, Backend
from test_long_video_runtime import environment, runtime, latent, guide, model, sample, sampled_guide, VideoVAE


def test_later_stale_override_is_rejected_before_any_assets_or_take(tmp_path):
    store = ProjectStore(tmp_path, 'test')
    cfg = config()
    cfg['scene_contract']['window_overrides'] = [{'new_start': 549, 'new_end': 719, 'voice_policy': 'inherit'}]
    frozen = []
    with pytest.raises(ValueError, match='exact planned'):
        configure_project(store, graph(), cfg, asset_freezer=lambda _: frozen.append('called') or {})
    assert frozen == [] and store.read()['windows'] == []


def test_compiled_policies_and_actual_bindings_are_immutable_and_reach_every_stage(tmp_path):
    store = ProjectStore(tmp_path, 'test')
    cfg = config()
    cfg['scene_contract'].update(voice_policy='dialogue_only', reference_selection={'pictures': [1], 'audios': []})
    configure_project(store, graph(), cfg, asset_freezer=lambda _: {})
    backend = Backend()
    runner = Coordinator(store, backend)
    assert runner.step()
    record = store.read()['windows'][0]
    saved = json.loads(store.path(f"takes/{record['take_id']}/request.json").read_text(encoding='utf-8'))
    compiled = compile_window(store.read()['config']['scene_contract'], saved['window'])
    assert all(saved[key] == value for key, value in compiled.items())
    assert saved['reference_bindings']['preprocessing_validation'] == 'director_policy_checked'
    assert 'long_video_references' in store.read()['config']['native_code']
    before = deepcopy(saved)
    for stage in ('native', 'preview', 'finish'):
        payload_graph = build_window_graph(store.read()['config']['base_graph'], saved, stage)
        director = payload_graph['10']
        assert director['_meta']['h3_long_video_reference_bindings'] == saved['reference_bindings']
        requests = [json.loads(node['inputs']['request_json']) for node in payload_graph.values()
                    if 'request_json' in node.get('inputs', {})]
        assert requests and all(all(req[key] == value for key, value in before.items()) for req in requests)
    assert saved == before


def test_explicit_rejected_content_retry_changes_seed_but_recovery_keeps_it(tmp_path, monkeypatch):
    store, backend, runner = setup(tmp_path)
    runner.step()
    original = store.read()['windows'][0]
    backend.jobs[store.read()['active_job']['prompt_id']] = 'failed'
    assert runner.step() is False
    store.resume()
    runner.step()
    recovered = store.read()['windows'][0]
    assert recovered['seed'] == original['seed']
    complete(store, backend)
    runner.step()
    store.review(recovered['take_id'], False, '', 'Observed defect needs another content attempt')
    monkeypatch.setattr(service, '_start', lambda *args: None)
    service.control(store, 'retry', {'index': 0})
    runner.step()
    retried = store.read()['windows'][0]
    assert retried['seed'] != recovered['seed']
    assert retried['content_attempt'] == 1
    backend.jobs[store.read()['active_job']['prompt_id']] = 'failed'
    runner.step()
    store.resume()
    runner.step()
    assert store.read()['windows'][0]['seed'] == retried['seed']


def test_accepted_parent_full_regeneration_requires_explicit_current_identity(tmp_path, monkeypatch):
    store, backend, runner = setup(tmp_path)
    runner.step()
    complete(store, backend)
    runner.step()
    approve(store)
    source = store.read()['windows'][0]
    monkeypatch.setattr(service, '_start', lambda *args: None)
    before = store.path('state.json').read_bytes()
    for payload in ({'index': 0}, {'index': 0, 'replace_accepted': True, 'take_id': 'stale'}):
        with pytest.raises(ValueError):
            service.control(store, 'retry', payload)
        assert store.path('state.json').read_bytes() == before
    service.control(store, 'retry', {'index': 0, 'replace_accepted': True, 'take_id': source['take_id']})
    runner.step()
    assert store.read()['windows'][0]['seed'] != source['seed']


@pytest.mark.parametrize('payload', [
    {'index': 0, 'video_ranges': [[20, 24]], 'anchors': []},
    {'index': 0, 'video_ranges': [[20, 24]], 'source_path': 'anything'},
    {'index': 0, 'start_seconds': 1, 'end_seconds': 2, 'video_ranges': [[20, 24]]},
    {'index': 0, 'start_seconds': 1, 'end_seconds': 2, 'stream': 'video', 'quiet_source': {}},
])
def test_unknown_or_mixed_repair_control_preserves_accepted_state(tmp_path, monkeypatch, payload):
    store, backend, runner = setup(tmp_path)
    runner.step()
    complete(store, backend)
    runner.step()
    approve(store)
    monkeypatch.setattr(service, '_start', lambda *args: pytest.fail('Invalid request started the coordinator'))
    before = store.path('state.json').read_bytes()
    with pytest.raises(ValueError):
        service.control(store, 'repair', payload)
    assert store.path('state.json').read_bytes() == before


def test_normal_seconds_control_records_requested_and_snapped_extent(tmp_path, monkeypatch):
    store, backend, runner = setup(tmp_path)
    runner.step()
    complete(store, backend)
    runner.step()
    approve(store)
    monkeypatch.setattr(service, '_start', lambda *args: None)
    result = service.control(store, 'repair', {'index': 0, 'start_seconds': 2.1, 'end_seconds': 2.6, 'stream': 'joint'})
    assert result['repair']['conditioning_extent']['requested_seconds'] == [2.1, 2.6]
    assert result['repair']['audio_ranges'] == [[84, 104]]
    assert result['repair']['lock_audio'] is False
    assert runner.step()
    repair = store.read()['windows'][0]
    backend.jobs[store.read()['active_job']['prompt_id']] = 'failed'
    runner.step()
    store.resume()
    assert runner.step()
    recovered = store.read()['windows'][0]
    assert recovered['repair'] == repair['repair']
    assert recovered['seed'] == repair['seed']
    assert recovered['base_fingerprint'] == repair['base_fingerprint']


def test_model_affecting_policy_changes_invalidate_cached_take(tmp_path):
    store, backend, runner = setup(tmp_path)
    runner.step()
    complete(store, backend)
    runner.step()
    approve(store)
    first = store.read()['windows'][0]['take_id']
    cfg = config()
    cfg['scene_contract']['voice_policy'] = 'dialogue_only'
    configure_project(store, graph(), cfg, asset_freezer=lambda _: {})
    assert runner.step()
    assert store.read()['windows'][0]['take_id'] != first
    assert store.read()['active_job']['stage'] == 'native'


@pytest.mark.parametrize('proof', ['missing', 'delivery_only', 'unavailable'])
def test_unreviewed_forward_tail_blocks_expensive_finish_and_child_queue(tmp_path, proof):
    store = ProjectStore(tmp_path, 'test')
    cfg = config()
    cfg['scene_contract']['window_overrides'] = [
        {'new_start': 294, 'new_end': 549, 'forward_frames': 51 if proof == 'unavailable' else 17}]
    configure_project(store, graph(), cfg, asset_freezer=lambda _: {})
    backend = Backend()
    runner = Coordinator(store, backend)
    runner.step()
    complete(store, backend)
    runner.step()
    approve(store)
    if proof == 'delivery_only':
        with store.transaction() as state:
            state['windows'][0]['forward_review'] = state['windows'][0]['review']
    before_jobs = len(backend.submitted)
    assert runner.step() is False
    assert store.read()['status'] == 'awaiting_forward_review'
    assert store.read()['windows'][0]['status'] == 'accepted'
    assert len(store.read()['windows']) == 1
    assert len(backend.submitted) == before_jobs
    assert not store.read().get('active_job')


def test_reviewed_forward_observation_drives_child_prompt_and_actual_runtime_prefix(environment, monkeypatch):
    cfg = config()
    cfg['project_name'] = environment.name
    cfg['scene_contract']['window_overrides'] = [{'new_start': 294, 'new_end': 549, 'forward_frames': 17}]
    configure_project(environment, graph(), cfg, asset_freezer=lambda _: {})
    backend = Backend()
    runner = Coordinator(environment, backend)
    runner.step()
    req = {**environment.read()['windows'][0], 'stage': 'native'}
    req = json.loads(environment.path(f"takes/{req['take_id']}/request.json").read_text(encoding='utf-8'))
    req['stage'] = 'native'
    prepared = runtime().MiniMaxH3LongVideoContext().prepare(latent(), guide(), json.dumps(req), model())[0]
    runtime().MiniMaxH3LongVideoCheckpoint().save(sample(prepared), prepared, sampled_guide(), VideoVAE(), object(), json.dumps(req))
    backend.jobs[environment.read()['active_job']['prompt_id']] = 'completed'
    assert runner.step() is False
    environment.review(req['take_id'], True, 'DELIVERY_STATE_HAND_ON_CUP', 'Actual delivery checked')
    assert runner.step() is False
    monkeypatch.setattr(service, '_start', lambda *args: None)
    service.control(environment, 'review_forward', {
        'take_id': req['take_id'], 'scope': 'Actual tail video, contact and audio checked',
        'observed_state': 'FORWARD_STATE_HAND_RELEASED_CUP'})
    assert runner.step()
    assert environment.read()['active_job']['stage'] == 'finish'
    complete(environment, backend, 'finish')
    assert runner.step()
    child = environment.read()['windows'][1]
    assert 'FORWARD_STATE_HAND_RELEASED_CUP' in child['prompt']
    assert 'DELIVERY_STATE_HAND_ON_CUP' not in child['prompt']
    assert child['parent']['forward_review'] == environment.read()['windows'][0]['forward_review']
    child_req = json.loads(environment.path(f"takes/{child['take_id']}/request.json").read_text(encoding='utf-8'))
    child_req['stage'] = 'native'
    baseline = runtime().MiniMaxH3LongVideoContext().prepare(latent(), guide(), json.dumps(child_req), model())[0]
    assert baseline['_h3_long_video_context']['video_steps'] == 17
    assert baseline['_h3_long_video_context']['audio_ticks'] == 93


def test_take_invalidated_between_graph_build_and_queue_claim_cannot_be_submitted(tmp_path, monkeypatch):
    store, backend, runner = setup(tmp_path)
    runner.step()
    complete(store, backend)
    runner.step()
    record = store.read()['windows'][0]
    write = store.write_once

    def concurrent_replacement(relative, value):
        result = write(relative, value)
        if str(relative).startswith('jobs/'):
            store.invalidate_from(0, 'Source changed before queue claim')
        return result

    monkeypatch.setattr(store, 'write_once', concurrent_replacement)
    count = len(backend.submitted)
    with pytest.raises(ValueError, match='source chain'):
        runner.schedule(record, 'preview')
    assert len(backend.submitted) == count
    assert not store.read().get('active_job')


def rejected_first(store, backend, runner):
    runner.step()
    complete(store, backend)
    runner.step()
    record = store.read()['windows'][0]
    store.review(record['take_id'], False, '', 'Actual content defect')
    return record


@pytest.mark.parametrize('change', ['retry', 'config', 'repair'])
def test_bound_creation_cas_preserves_newer_target_attempt_config_or_repair(tmp_path, monkeypatch, change):
    store, backend, runner = setup(tmp_path)
    original = rejected_first(store, backend, runner)
    if change == 'repair':
        store.request_repair(0, [[20, 24]])
    else:
        store.request_retry(0)
    create = store.create_take
    injected = False

    def concurrent_control(spec, **options):
        nonlocal injected
        if not injected:
            injected = True
            if change == 'retry':
                store.request_retry(0)
                store.resume()
            elif change == 'config':
                changed = graph()
                changed['10']['inputs']['seed'] = 77
                configure_project(store, changed, config(), asset_freezer=lambda _: {})
            else:
                with store.transaction() as state:
                    state['repair']['video_ranges'] = [[30, 32]]
        return create(spec, **options)

    monkeypatch.setattr(store, 'create_take', concurrent_control)
    count = len(backend.submitted)
    assert runner.step() is True  # Recoverable stale work requests a fresh read.
    state = store.read()
    assert len(backend.submitted) == count and not state.get('active_job')
    assert state['windows'][0]['take_id'] == original['take_id']
    assert state['windows'][0]['status'] == 'invalidated'
    assert state['status'] == 'ready' and 'error' not in state
    if change == 'retry':
        assert state['content_attempts']['0'] == 2
    elif change == 'config':
        assert state['config']['seed'] == 77
    else:
        assert state['repair']['video_ranges'] == [[30, 32]]
    assert runner.step()
    current = store.read()['windows'][0]
    assert current['content_attempt'] == (2 if change == 'retry' else 0 if change == 'repair' else 1)
    if change == 'repair':
        assert current['repair']['video_ranges'] == [[30, 32]]


def test_queue_claim_cas_rejects_config_changed_after_take_creation(tmp_path, monkeypatch):
    store, backend, runner = setup(tmp_path)
    write = store.write_once
    injected = False

    def concurrent_config(relative, value):
        nonlocal injected
        result = write(relative, value)
        if str(relative).startswith('jobs/') and not injected:
            injected = True
            changed = graph()
            changed['10']['inputs']['seed'] = 99
            configure_project(store, changed, config(), asset_freezer=lambda _: {})
        return result

    monkeypatch.setattr(store, 'write_once', concurrent_config)
    assert runner.step() is True
    assert backend.submitted == [] and not store.read().get('active_job')
    assert store.read()['config']['seed'] == 99 and 'error' not in store.read()
    assert runner.step()
    assert store.read()['windows'][0]['seed'] == 99


def test_resumed_queued_take_refreshes_reservation_after_harmless_extension_and_export_change(tmp_path, monkeypatch):
    store, backend, runner = setup(tmp_path)
    schedule = runner.schedule
    monkeypatch.setattr(runner, 'schedule', lambda *args, **kwargs: False)
    assert runner.step() is False
    saved = store.read()['windows'][0]
    changed = graph()
    changed['10']['inputs']['resolution_preset'] = '4K UHD'
    configure_project(store, changed, config(60), asset_freezer=lambda _: {})
    monkeypatch.setattr(runner, 'schedule', schedule)
    assert runner.step()
    current = store.read()['windows'][0]
    assert current['take_id'] == saved['take_id'] and current['seed'] == saved['seed']
    assert current['fingerprint'] == saved['fingerprint']
    assert store.read()['active_job']['stage'] == 'native'


@pytest.mark.parametrize('production_helper', [False, True])
def test_drive_recovers_parent_retry_during_history_without_clobbering_new_ready_state(tmp_path, monkeypatch, production_helper):
    import threading
    store, backend, runner = setup(tmp_path)
    runner.step()
    complete(store, backend)
    runner.step()
    approve(store)
    parent = store.read()['windows'][0]
    runner.step()
    complete(store, backend, 'finish')
    runner.reconcile_job()
    wake = threading.Event()
    original_history = service.consumed_history if production_helper else runner.history_provider
    injected = False
    saw_new_retry = []

    def concurrent_parent_retry(s, p, w):
        nonlocal injected
        if not injected:
            injected = True
            s.request_retry(0, replace_accepted=True, take_id=p['take_id'])
            s.resume()
            wake.set()
        return original_history(s, p, w)

    runner.history_provider = concurrent_parent_retry
    step = runner.step

    def bounded_step():
        result = step()
        state = store.read()
        if state.get('active_job') and state['windows'][0]['take_id'] != parent['take_id']:
            saw_new_retry.append(state)
            complete(store, backend)
            store.stop()
        return result

    monkeypatch.setattr(runner, 'step', bounded_step)
    monkeypatch.setattr(service, 'Coordinator', lambda *args: runner)
    monkeypatch.setattr(service, 'ComfyQueue', lambda: backend)
    service._drive(store, wake)
    assert injected and len(saw_new_retry) == 1
    assert saw_new_retry[0]['content_attempts']['0'] == 1
    assert saw_new_retry[0]['windows'][0]['seed'] != parent['seed']
    assert 'error' not in saw_new_retry[0]
    assert store.read()['status'] == 'stopped' and 'error' not in store.read()
    assert len(store.read()['windows']) == 1


def test_unchanged_corrupt_evidence_from_real_history_helper_still_fails_drive(tmp_path, monkeypatch):
    import threading
    store, backend, runner = setup(tmp_path)
    runner.step()
    complete(store, backend)
    runner.step()
    approve(store)
    runner.step()
    complete(store, backend, 'finish')
    runner.reconcile_job()
    # Lifecycle fixtures are intentionally not real safetensors/clean receipts.
    # The production helper must report this unchanged missing clean evidence.
    runner.history_provider = service.consumed_history
    before = store.read()['windows'][0]
    count = len(backend.submitted)
    monkeypatch.setattr(service, 'Coordinator', lambda *args: runner)
    monkeypatch.setattr(service, 'ComfyQueue', lambda: backend)
    service._drive(store, threading.Event())
    state = store.read()
    assert state['status'] == 'failed'
    assert state['error'].startswith('ValueError:')
    assert 'recoverable clean AV checkpoint' in state['error']
    assert state['windows'][0] == before and len(backend.submitted) == count


@pytest.mark.parametrize('stage', ['native', 'preview'])
@pytest.mark.parametrize('queue_status', ['completed', 'failed', 'missing'])
def test_decoded_take_rejected_before_reconciliation_remains_rejected_until_explicit_action(tmp_path, stage, queue_status):
    store, backend, runner = setup(tmp_path)
    runner.step()
    if stage == 'preview':
        complete(store, backend, clean_only=True)
        assert runner.step()
        assert store.read()['active_job']['stage'] == 'preview'
    complete(store, backend)
    record = store.read()['windows'][0]
    store.review(record['take_id'], False, '', 'Reviewed decoded native AV and rejected actual content')
    assert store.read()['generated_frames'] == 294
    job = store.read()['active_job']['prompt_id']
    if queue_status == 'missing':
        backend.jobs.pop(job)
    else:
        backend.jobs[job] = queue_status
    count = len(backend.submitted)
    assert runner.step() is False
    state = store.read()
    assert state['status'] == state['windows'][0]['status'] == 'rejected'
    assert state['windows'][0]['take_id'] == record['take_id']
    assert state['generated_frames'] == 294 and state['completed_frames'] == 0
    assert not state.get('active_job') and 'error' not in state
    store.verify(record['artifacts'])
    store.resume()
    assert runner.step() is False
    assert store.read()['windows'][0]['take_id'] == record['take_id']
    assert len(backend.submitted) == count


def test_finished_job_reconciliation_preserves_retry_after_queue_ownership_releases(tmp_path, monkeypatch):
    from contextlib import contextmanager
    store, backend, runner = setup(tmp_path)
    runner.step()
    complete(store, backend)
    runner.step()
    approve(store)
    source = store.read()['windows'][0]
    runner.step()
    job_id = store.read()['active_job']['prompt_id']
    backend.jobs[job_id] = 'failed'
    transaction = store.transaction
    injected = False

    @contextmanager
    def newer_retry_after_release():
        nonlocal injected
        with transaction() as state:
            had_job = state.get('active_job', {}).get('prompt_id') == job_id
            yield state
            released = had_job and not state.get('active_job')
        if released and not injected:
            injected = True
            store.request_retry(0, replace_accepted=True, take_id=source['take_id'])
            store.resume()

    monkeypatch.setattr(store, 'transaction', newer_retry_after_release)
    assert runner.step() is True
    assert injected and store.read()['status'] == 'ready'
    assert store.read()['content_attempts']['0'] == 1
    assert 'error' not in store.read() and not store.read().get('active_job')
