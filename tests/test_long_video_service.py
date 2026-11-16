import json
import asyncio
import sys
import threading
import types
from copy import deepcopy
from functools import partial
from pathlib import Path

import pytest

from nodes.long_video_store import ProjectStore
from nodes.long_video_service import Coordinator, ComfyQueue, configure_project as _configure_project


configure_project = partial(_configure_project, export_asset_freezer=lambda _: {}, encoder_checker=lambda: {})


def graph():
    data = json.loads((Path(__file__).parents[1] / 'templates/u11_api.json').read_text(encoding='utf-8'))
    data['10']['inputs'].update(mode='REF2VA', sampling_method='selflift',
        performance_preset='quality_two_stage', postprocess_mode='vosr2',
        vosr2_profile='manual_baseline', resolution_preset='2K QHD', seed=42)
    return data


def config(seconds=30):
    total = int(seconds * 24)
    return {'enabled': True, 'project_name': 'test', 'target_seconds': seconds,
            'stop_condition': 'target', 'scene_contract': {
                'schema_version': 1, 'fps': 24,
                'subject_definitions': '<Subject 1> adult, <Picture 1> identity.',
                'summary': 'One continuous tracking shot.', 'retention_analysis': 'Walk continuously.',
                'overall_soundscape': 'Room tone', 'non_diegetic_music': 'N/A',
                'scale_contract': 'Adult 170 cm; cup 10 cm.',
                'beats': [{'id': 'walk', 'start_frame': 0, 'end_frame': total,
                           'action': 'Walk naturally carrying the same cup.'}], 'dialogues': []}}


def test_unsupported_complete_encoding_rejects_before_asset_freeze_or_take(tmp_path):
    store = ProjectStore(tmp_path,'test')
    base = graph()
    base['24']['inputs']['bit_depth'] = '10-bit'
    with pytest.raises(ValueError,match='8-bit'):
        configure_project(store,base,config(),asset_freezer=lambda _: pytest.fail('hashed before rejecting encoding'))
    assert store.read()['windows'] == []


def test_resolved_encoding_is_fingerprinted_and_audio_quality_are_not_lost(tmp_path):
    store = ProjectStore(tmp_path,'test')
    base = graph(); base['24']['inputs'].update(quality=11,audio_bitrate='96k')
    first = configure_project(store,base,config(),asset_freezer=lambda _: {})
    assert first['export']['encoding']['crf'] == 11
    assert first['export']['encoding']['audio_bitrate'] == 96000
    base['24']['inputs']['audio_bitrate'] = '128k'
    second = configure_project(store,base,config(),asset_freezer=lambda _: {})
    assert second['export_fingerprint'] != first['export_fingerprint']
    assert second['generation_key'] == first['generation_key']


def test_export_code_change_stops_before_new_queue_work(tmp_path,monkeypatch):
    import nodes.long_video_service as service
    store,backend,runner = setup(tmp_path)
    original = service._code_identity
    def changed(names):
        result = original(names)
        if 'long_video_media' in result:
            result['long_video_media'] = 'new-encoder-version'
        return result
    monkeypatch.setattr(service,'_code_identity',changed)
    with pytest.raises(ValueError,match='original U11.*export|export.*original U11'):
        runner.step()
    assert backend.submitted == []


@pytest.mark.parametrize('name', ['selflift_lifter', 'vendor/minimax_h3_latent_upscaler_2d'])
def test_native_lifter_code_change_stops_before_new_queue_work(tmp_path, monkeypatch, name):
    import nodes.long_video_service as service
    store, backend, runner = setup(tmp_path)
    original = service._code_identity
    def changed(names):
        result = original(names)
        if name in result:
            result[name] = 'different-lifter-code'
        return result
    monkeypatch.setattr(service, '_code_identity', changed)
    with pytest.raises(ValueError, match='original U11'):
        runner.step()
    assert backend.submitted == []


def test_original_requeue_refreshes_export_code_while_preserving_valid_native(tmp_path,monkeypatch):
    import nodes.long_video_service as service
    store,backend,runner = setup(tmp_path,seconds=5)
    runner.step(); complete(store,backend); runner.step(); approve(store)
    native_id = store.read()['windows'][0]['take_id']
    original = service._code_identity
    def changed(names):
        result = original(names)
        if 'long_video_media' in result:
            result['long_video_media'] = 'new-encoder-version'
        return result
    monkeypatch.setattr(service,'_code_identity',changed)
    with pytest.raises(ValueError,match='original U11'):
        runner.step()
    configure_project(store,graph(),config(5),asset_freezer=lambda _: {})
    assert runner.step()
    assert store.read()['active_job']['stage'] == 'finish'
    assert store.read()['windows'][0]['take_id'] == native_id


class Backend:
    def __init__(self):
        self.jobs = {}
        self.submitted = []

    def submit(self, prompt_id, prompt):
        self.jobs[prompt_id] = 'queued'
        self.submitted.append((prompt_id, prompt))

    def status(self, prompt_id):
        return self.jobs.get(prompt_id, 'missing')


def setup(tmp_path, seconds=30):
    store = ProjectStore(tmp_path, 'test')
    configure_project(store, graph(), config(seconds), asset_freezer=lambda _: {})
    backend = Backend()

    def assemble(parts, output, total_frames, target, *, encoding=None):
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_bytes(b'assembled fixture')
        return {'frames': total_frames, 'source_parts': len(parts), 'content_review': 'technical_only', 'encoding': encoding}

    runner = Coordinator(store, backend, history_provider=lambda s, p, w: 'history-' + p['take_id'],
                         assembler=assemble)
    return store, backend, runner


@pytest.mark.parametrize('selected,policy', [([1], 'inherit'), ([2], 'inherit'),
                                            ([], 'inherit'), ([1], 'dialogue_only')])
def test_selected_reference_jobs_keep_required_empty_voice_names(tmp_path, selected, policy):
    store = ProjectStore(tmp_path, 'test')
    base = graph()
    base['10']['inputs'].update(voice_mode='h3_reference',
        voice_reference_audio_file='first.wav', voice_reference_audio_2_file='second.wav',
        voice_reference_name_1='First speaker', voice_reference_name_2='Second speaker',
        voice_reference_name_3='Unused speaker')
    raw = config(5)
    raw['scene_contract']['window_overrides'] = [{
        'new_start': 0, 'new_end': 120,
        'reference_selection': {'pictures': [1], 'audios': selected},
        'voice_policy': policy}]
    configure_project(store, base, raw, asset_freezer=lambda _: {})
    backend = Backend()
    runner = Coordinator(store, backend)
    assert runner.step()
    prompt_id, actual = backend.submitted[0]
    settings = actual['10']['inputs']
    expected = (f'{"First" if selected == [1] else "Second"} speaker'
                if selected and policy == 'inherit' else '')
    # Comfy validates required inputs before calling Python, so the callable's
    # default arguments do not substitute for missing empty widget values.
    assert {key: settings[key] for key in ('voice_reference_name_1',
            'voice_reference_name_2', 'voice_reference_name_3')} == {
        'voice_reference_name_1': expected, 'voice_reference_name_2': '',
        'voice_reference_name_3': ''}
    assert not any(key.startswith(('voice_reference_audio_2', 'voice_reference_audio_3'))
                   for key in settings)
    recorded = json.loads(store.path(f'jobs/{prompt_id}.json').read_text(encoding='utf-8'))
    assert recorded == actual


def complete(store, backend, stage='native', clean_only=False):
    state = store.read()
    job = state['active_job']
    take = next(w for w in state['windows'] if w['take_id'] == job['take_id'])
    folder = store.path('takes/' + take['take_id'])
    if stage == 'native':
        path = folder / 'clean_video.safetensors'
        path.write_bytes(b'fake tensor bytes for lifecycle tests')
        store.record_clean(take['take_id'], {'artifacts': {'clean_video': store.artifact(path)}})
        if not clean_only:
            path = folder / 'native.mp4'
            path.write_bytes(b'fake preview for lifecycle tests')
            audio = folder / 'delivery.wav'
            audio.write_bytes(b'fake exact delivery audio')
            store.record_native(take['take_id'], {'artifacts': {
                'preview': store.artifact(path), 'delivery_audio': store.artifact(audio)}})
    else:
        path = folder / 'enhanced.mp4'
        path.write_bytes(b'fake enhanced video for lifecycle tests')
        with store.transaction() as current:
            record = store.active(current, take['take_id'])
            record['exports'][current['config']['export_fingerprint']] = {
                'video': store.artifact(path), 'audio': record['artifacts']['delivery_audio'],
                'frames': take['window']['new_end'] - take['window']['new_start']}
    backend.jobs[job['prompt_id']] = 'completed'


def approve(store):
    take = store.read()['windows'][-1]
    store.review(take['take_id'], True, 'Walk continues; right hand holds the cup.',
                 'Native whole window including first free frames, dialogue, props and outgoing state.')


def test_window_waits_for_native_review_before_expensive_export_or_continuation(tmp_path):
    store, backend, runner = setup(tmp_path)
    assert runner.step()
    assert len(backend.submitted) == 1
    assert not any(n['class_type'] == 'MiniMaxH3LongVideoExport' for n in backend.submitted[0][1].values())
    complete(store, backend)
    assert runner.step() is False
    assert store.read()['status'] == 'awaiting_review'
    approve(store)
    assert runner.step()
    assert any(n['class_type'] == 'MiniMaxH3LongVideoExport' for n in backend.submitted[-1][1].values())
    complete(store, backend, 'finish')
    assert runner.step()
    assert len(store.read()['windows']) == 2


def test_completed_project_assembles_exact_target_and_can_resume_without_new_h3(tmp_path):
    store, backend, runner = setup(tmp_path)
    for _ in range(3):
        assert runner.step()
        complete(store, backend)
        assert runner.step() is False
        approve(store)
        assert runner.step()
        complete(store, backend, 'finish')
        # Complete active export bookkeeping without beginning the next one.
        runner.reconcile_job()
    assert runner.step() is False
    assert store.read()['status'] == 'complete'
    assert store.read()['assembly']['frames'] == 720
    assert store.read()['assembly']['encoding'] == store.read()['config']['export']['encoding']
    count = len(backend.submitted)
    configure_project(store, graph(), config(), asset_freezer=lambda _: {})
    assert runner.step() is False
    assert len(backend.submitted) == count


def test_stop_keeps_running_job_and_saves_clean_before_stopping(tmp_path):
    store, backend, runner = setup(tmp_path)
    runner.step()
    store.stop()
    assert runner.step()
    complete(store, backend)
    assert runner.step() is False
    assert store.read()['status'] == 'stopped'
    assert store.read()['windows'][0]['status'] == 'awaiting_review'
    assert len(backend.submitted) == 1


def test_missing_queue_after_restart_retries_preview_from_clean_without_sampling(tmp_path):
    store, backend, runner = setup(tmp_path)
    runner.step()
    complete(store, backend, clean_only=True)
    backend.jobs.clear()
    assert runner.step()
    assert any(n['class_type'] == 'MiniMaxH3LongVideoLoadAV' for n in backend.submitted[-1][1].values())
    assert not any('Sampler' in n['class_type'] for n in backend.submitted[-1][1].values())


def test_extension_keeps_accepted_full_prefix_but_regenerates_partial_tail(tmp_path):
    store, backend, runner = setup(tmp_path)
    runner.step()
    complete(store, backend)
    runner.step()
    approve(store)
    first_id = store.read()['windows'][0]['take_id']
    configure_project(store, graph(), config(60), asset_freezer=lambda _: {})
    assert runner.step()
    assert store.read()['windows'][0]['take_id'] == first_id
    assert store.read()['windows'][0]['status'] == 'accepted'


def test_export_change_reuses_native_but_model_change_invalidates_it(tmp_path):
    store, backend, runner = setup(tmp_path)
    runner.step()
    complete(store, backend)
    runner.step()
    approve(store)
    first_id = store.read()['windows'][0]['take_id']
    changed = graph()
    changed['10']['inputs']['resolution_preset'] = '4K UHD'
    configure_project(store, changed, config(), asset_freezer=lambda _: {})
    runner.step()
    assert store.read()['windows'][0]['take_id'] == first_id
    assert store.read()['active_job']['stage'] == 'finish'
    complete(store, backend, 'finish')
    runner.reconcile_job()
    changed['2']['inputs']['unet_name'] = 'different-model.safetensors'
    configure_project(store, changed, config(), asset_freezer=lambda _: {})
    runner.step()
    assert store.read()['windows'][0]['take_id'] != first_id
    assert store.read()['active_job']['stage'] == 'native'


def test_rejected_window_requires_repair_or_explicit_retry(tmp_path):
    store, backend, runner = setup(tmp_path)
    runner.step()
    complete(store, backend)
    runner.step()
    take = store.read()['windows'][0]
    store.review(take['take_id'], False, '', 'Wrong hand and extra speech.')
    assert runner.step() is False
    assert store.read()['status'] == 'rejected'
    store.request_repair(0, [[20,30]], lock_audio=True)
    assert runner.step()
    request = store.read()['windows'][0]
    assert request['repair']['source_take'] == take['take_id']
    assert request['repair']['lock_audio'] is True


def test_native_review_receipt_tampering_cannot_resume(tmp_path):
    store, backend, runner = setup(tmp_path)
    runner.step()
    complete(store, backend)
    runner.step()
    approve(store)
    review = store.read()['windows'][0]['review']
    store.path(review['name']).write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='hash'):
        runner.step()


def test_resumed_runtime_payload_contains_only_immutable_spec_and_stage(tmp_path):
    store, backend, runner = setup(tmp_path)
    runner.step()
    complete(store, backend)
    runner.step()
    approve(store)
    runner.step()
    record = store.read()['windows'][0]
    saved = json.loads(store.path(f"takes/{record['take_id']}/request.json").read_text(encoding='utf-8'))
    payload = json.loads(backend.submitted[-1][1]['24']['inputs']['request_json'])
    assert {k:v for k,v in payload.items() if k not in {'stage','export_fingerprint'}} == saved
    assert payload['stage'] == 'finish'
    assert 'artifacts' not in payload and 'status' not in payload


def test_extending_completed_project_reuses_two_full_windows_and_replaces_partial_tail(tmp_path):
    store, backend, runner = setup(tmp_path)
    for _ in range(3):
        runner.step()
        complete(store, backend)
        runner.step()
        approve(store)
        runner.step()
        complete(store, backend, 'finish')
        runner.reconcile_job()
    old = [w['take_id'] for w in store.read()['windows']]
    runner.step()
    configure_project(store, graph(), config(60), asset_freezer=lambda _: {})
    runner.step()
    new = [w['take_id'] for w in store.read()['windows']]
    assert new[:2] == old[:2] and new[2] != old[2]
    assert store.read()['active_job']['stage'] == 'native'


def test_failed_sampler_retries_as_a_new_take_and_preserves_partial_files(tmp_path):
    store, backend, runner = setup(tmp_path)
    runner.step()
    old = store.read()['windows'][0]['take_id']
    partial = store.path(f'takes/{old}/prepared_video.safetensors')
    partial.write_bytes(b'partial diagnosis')
    backend.jobs[store.read()['active_job']['prompt_id']] = 'failed'
    assert runner.step() is False
    assert store.read()['status'] == 'failed'
    store.resume()
    runner.step()
    assert store.read()['windows'][0]['take_id'] != old
    assert partial.read_bytes() == b'partial diagnosis'


@pytest.mark.parametrize('queue_status', ['failed', 'missing'])
def test_failed_bounded_repair_retains_ranges_and_audio_lock_on_retry(tmp_path, queue_status):
    store, backend, runner = setup(tmp_path)
    runner.step()
    complete(store, backend)
    runner.step()
    source = store.read()['windows'][0]['take_id']
    store.review(source, False, '', 'Video defect, native audio remains usable.')
    store.request_repair(0, [[20, 30]], lock_audio=True)
    runner.step()
    failed = store.read()['windows'][0]
    job = store.read()['active_job']['prompt_id']
    if queue_status == 'missing':
        backend.jobs.pop(job)
    else:
        backend.jobs[job] = 'failed'
    runner.step()
    store.resume()
    runner.step()
    retried = store.read()['windows'][0]
    assert retried['take_id'] != failed['take_id']
    assert retried['repair'] == failed['repair'] == {
        'index': 0, 'source_take': source, 'video_ranges': [[20, 30]], 'lock_audio': True}


def test_stop_during_asset_hashing_survives_configure(tmp_path):
    store = ProjectStore(tmp_path, 'test')

    def freeze(_):
        store.stop()
        return {}

    configure_project(store, graph(), config(), asset_freezer=freeze)
    backend = Backend()
    runner = Coordinator(store, backend)
    assert runner.step() is False
    assert store.read()['stop_requested'] is True
    assert store.read()['status'] == 'stopped'
    assert backend.submitted == []


def test_newly_installed_implicit_adapter_cannot_silently_resume(tmp_path, monkeypatch):
    import nodes.long_video_service as service
    store = ProjectStore(tmp_path, 'test')
    configure_project(store, graph(), config(), asset_freezer=lambda _: {
        'records': {}, 'choices': {'motion': 'missing'}})
    monkeypatch.setattr(service, 'current_native_choices', lambda _: {'motion': 'Motion_Repair.safetensors'})
    backend = Backend()
    with pytest.raises(ValueError, match='native model selection changed'):
        Coordinator(store, backend).step()
    assert backend.submitted == []


def test_replacing_sr_weights_only_changes_export_fingerprint(tmp_path):
    from nodes.long_video_assets import freeze_export_assets
    weights = tmp_path / 'dit.safetensors'
    weights.write_bytes(b'original reconstruction weights')
    freezer = lambda g: freeze_export_assets(g, object(), bypass=False,
        bundle_resolver=lambda registry, bundle: {'dit': [weights]})
    store = ProjectStore(tmp_path, 'test')
    before = configure_project(store, graph(), config(), asset_freezer=lambda _: {},
                               export_asset_freezer=freezer)
    weights.write_bytes(b'new reconstruction weights')
    with pytest.raises(ValueError, match='Enhancement asset changed'):
        Coordinator(store, Backend()).step()
    after = configure_project(store, graph(), config(), asset_freezer=lambda _: {},
                              export_asset_freezer=freezer)
    assert before['generation_key'] == after['generation_key']
    assert before['export_fingerprint'] != after['export_fingerprint']


def test_repair_during_assembly_cannot_publish_or_reuse_revoked_film(tmp_path):
    store, backend, runner = setup(tmp_path, seconds=5)
    runner.step()
    complete(store, backend)
    runner.step()
    approve(store)
    runner.step()
    complete(store, backend, 'finish')
    runner.reconcile_job()
    original_assembler = runner.assembler
    assembled = []

    def concurrent_repair(parts, output, frames, target, *, encoding=None):
        result = original_assembler(parts, output, frames, target, encoding=encoding)
        assembled.append(output)
        if len(assembled) == 1:
            store.request_repair(0, [[20, 30]], lock_audio=True)
        return result

    runner.assembler = concurrent_repair
    assert runner.step() is True
    assert store.read()['output'] == ''
    assert store.read()['windows'][0]['status'] == 'invalidated'
    runner.step()
    complete(store, backend)
    runner.step()
    approve(store)
    runner.step()
    complete(store, backend, 'finish')
    assert runner.step() is False
    assert len(assembled) == 2
    assert store.read()['output'] == str(assembled[1])
    assert store.read()['final_source']


def test_failed_repair_can_be_explicitly_abandoned_for_full_regeneration(tmp_path, monkeypatch):
    import nodes.long_video_service as service
    store, backend, runner = setup(tmp_path, seconds=5)
    runner.step()
    complete(store, backend)
    runner.step()
    store.review(store.read()['windows'][0]['take_id'], False, '', 'Rejected video only')
    store.request_repair(0, [[20, 30]], lock_audio=True)
    runner.step()
    backend.jobs[store.read()['active_job']['prompt_id']] = 'failed'
    runner.step()
    assert store.read()['windows'][0]['status'] == 'invalidated'
    monkeypatch.setattr(service, '_start', lambda store: None)
    service.control(store, 'retry', {'index': 0})
    assert runner.step() is True
    assert 'repair' not in store.read()['windows'][0]


def test_standard_queue_submission_uses_six_field_tuple_and_native_validation(monkeypatch):
    submitted = []
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def serve():
        asyncio.set_event_loop(loop)
        ready.set()
        loop.run_forever()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    assert ready.wait(5)
    validations = []

    async def validate(prompt_id, prompt, targets):
        validations.append((prompt_id, prompt, targets))
        return True, None, ['out'], {}

    monkeypatch.setitem(sys.modules, 'execution', types.SimpleNamespace(validate_prompt=validate))
    server = types.SimpleNamespace(loop=loop, number=7,
        node_replace_manager=types.SimpleNamespace(apply_replacements=lambda graph: None),
        prompt_queue=types.SimpleNamespace(put=submitted.append))
    try:
        ComfyQueue(server).submit('prompt-123', {'out': {'class_type':'Output','inputs':{}}})
        assert validations == [('prompt-123', submitted[0][2], None)]
        assert len(submitted[0]) == 6 and submitted[0][0:2] == (7,'prompt-123')
        assert submitted[0][4] == ['out'] and server.number == 8
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(5)
        loop.close()
