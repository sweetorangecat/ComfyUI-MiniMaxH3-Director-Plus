"""Disk-backed reference provenance and the real runtime preparation boundary."""
from copy import deepcopy
import json

import pytest
import torch

from nodes.long_video_av import prepare_av, slice_fingerprint, streams
from nodes.long_video_plan import plan_windows
from nodes import long_video_repair as repair
from test_long_video_runtime import environment, guide, latent, model, request, runtime, Nested


def scene(store):
    previous = None
    for window in plan_windows(2160)[:3]:
        parent = None if previous is None else {
            'take_id': previous[0]['take_id'],
            'history_sha256': slice_fingerprint(previous[1], window['source_endpoint'], window['context_frames'])}
        take = request(store, window, parent=parent)
        clean = prepare_av(latent(), window, source=None if previous is None else previous[1], nested_factory=Nested)
        for tensor, mask in zip(streams(clean), clean['noise_mask'].unbind()):
            tensor.add_(mask.expand_as(tensor) * .25)
        artifacts = {'request': store.artifact(store.path(f"takes/{take['take_id']}/request.json"))}
        for name, tensor in zip(('video', 'audio'), streams(clean)):
            path = store.path(f"takes/{take['take_id']}/clean_{name}.safetensors")
            runtime()._save_tensors(path, {'samples': tensor})
            artifacts['clean_' + name] = store.artifact(path)
        path = store.write_once(f"takes/{take['take_id']}/clean_receipt.json", {'artifacts': deepcopy(artifacts)})
        artifacts['clean_receipt'] = store.artifact(path)
        preview = store.path(f"takes/{take['take_id']}/native.mp4")
        preview.write_bytes(b'native media fixture: not an actual motion review')
        artifacts['native'] = store.artifact(preview)
        path = store.write_once(f"takes/{take['take_id']}/native_receipt.json", {'artifacts': {'native': artifacts['native']}})
        artifacts['native_receipt'] = store.artifact(path)
        store.record_native(take['take_id'], {'artifacts': artifacts})
        store.review(take['take_id'], window['index'] != 2, 'Open laptop; hands by paper.', 'Actual native preview.')
        previous = take, clean
    state = store.read()
    return state['windows'][0], state['windows'][2]


def reference(store, source):
    review = repair.review_video_reference(store, source['take_id'], [87, 97],
        'Actual native frames 294..328, camera, hands, cards and laptop checked.',
        'Open laptop at the same angle; empty hands rest by paper; same card state.')
    return {'source_take': source['take_id'], 'source_steps': [87, 97], 'target_steps': [77, 87], 'review': review}


def test_review_freezes_exact_current_native_source_and_unused_range(environment):
    source, failed = scene(environment)
    ref = reference(environment, source)
    receipt = repair.verify_video_reference(environment, ref, failed['window'])
    assert receipt['source_review'] == source['review']
    assert receipt['source_artifacts'] == source['artifacts']
    assert receipt['source_native_frames'] == [294, 328]
    assert receipt['observed_state']
    with pytest.raises(ValueError, match='earlier'):
        repair.verify_video_reference(environment, ref, source['window'])


@pytest.mark.parametrize('steps', [[82, 92], [92, 102], [88, 98]])
def test_review_rejects_delivered_outside_or_partial_codec_groups(environment, steps):
    source, _ = scene(environment)
    before = environment.read()
    with pytest.raises(ValueError):
        repair.review_video_reference(environment, source['take_id'], steps, 'Actual', 'Observed')
    assert environment.read() == before


@pytest.mark.parametrize('change', ['rejected', 'superseded', 'changed_file', 'missing_review', 'changed_range', 'wrong_size'])
def test_bad_reference_fails_before_revoking_existing_source_chain(environment, change):
    source, failed = scene(environment)
    ref = reference(environment, source)
    if change in {'rejected', 'superseded'}:
        with environment.transaction() as state:
            state['windows'][0]['status'] = 'rejected' if change == 'rejected' else 'invalidated'
    elif change == 'changed_file':
        environment.path(source['artifacts']['clean_video']['name']).write_bytes(b'changed')
    elif change == 'missing_review':
        environment.path(ref['review']['name']).unlink()
    elif change == 'changed_range':
        ref['source_steps'] = [92, 102]
    elif change == 'wrong_size':
        ref['review']['bytes'] += 1
    before = environment.path('state.json').read_bytes()
    with pytest.raises(ValueError):
        environment.request_repair(2, [[22, 97]], True, take_id=failed['take_id'], video_reference=ref)
    assert environment.path('state.json').read_bytes() == before


def test_reference_is_rechecked_inside_take_creation_lock(environment):
    source, failed = scene(environment)
    ref = reference(environment, source)
    environment.request_repair(2, [[22, 97]], True, take_id=failed['take_id'], video_reference=ref)
    spec = json.loads(environment.path(f"takes/{failed['take_id']}/request.json").read_text(encoding='utf-8'))
    spec['repair'] = environment.read()['repair']
    for key in ('take_id', 'project_name'):
        spec.pop(key, None)
    with environment.transaction() as state:
        state['windows'][0]['status'] = 'rejected'
    before = environment.path('state.json').read_bytes()
    with pytest.raises(ValueError, match='accepted'):
        environment.create_take(spec)
    assert environment.path('state.json').read_bytes() == before


def test_runtime_loads_reviewed_source_and_keeps_full_failed_audio(environment):
    source, failed = scene(environment)
    ref = reference(environment, source)
    original = runtime().load_clean(environment, failed['take_id'])
    expected = runtime().load_clean(environment, source['take_id'])
    environment.request_repair(2, [[22, 97]], True, take_id=failed['take_id'], video_reference=ref)
    spec = json.loads(environment.path(f"takes/{failed['take_id']}/request.json").read_text(encoding='utf-8'))
    spec['repair'] = environment.read()['repair']
    for key in ('take_id', 'project_name'):
        spec.pop(key, None)
    request = environment.create_take(spec)
    prepared = runtime().MiniMaxH3LongVideoContext().prepare(latent(), guide(), json.dumps(request), model())[0]
    video, audio = streams(prepared)
    vm, am = prepared['noise_mask'].unbind()
    assert torch.equal(video[:, :, 77:87], streams(expected)[0][:, :, 87:97])
    assert torch.equal(audio, streams(original)[1])
    assert torch.all(vm[:, :, 77:87] == 0) and torch.all(am == 0)
    assert prepared['_h3_long_video_context']['video_reference'] == ref
    with environment.transaction() as state:
        state['windows'][0]['status'] = 'rejected'
    with pytest.raises(ValueError, match='accepted'):
        runtime()._bound_store(request)


def test_expert_service_reviews_then_passes_exact_reference_to_existing_repair(environment, monkeypatch):
    from nodes import long_video_service as service
    source, failed = scene(environment)
    started = []
    monkeypatch.setattr(service, '_start', lambda store: started.append(store.name))
    result = service.control(environment, 'review_video_reference', {
        'source_take': source['take_id'], 'source_steps': [87, 97],
        'scope': 'Actual unused native tail checked.', 'observed_state': 'Open laptop, empty hands on desk.'})
    assert not started and 'video_reference_review' not in environment.read()
    ref = {'source_take': source['take_id'], 'source_steps': [87, 97], 'target_steps': [77, 87],
           'review': result['video_reference_review']}
    service.control(environment, 'repair', {'index': 2, 'take_id': failed['take_id'],
        'video_ranges': [[22, 97]], 'lock_audio': True, 'video_reference': ref})
    assert started == [environment.name]
    assert environment.read()['repair']['video_reference'] == ref


def test_accepted_forward_delivery_is_not_an_unused_reference(environment):
    source, _ = scene(environment)
    with environment.transaction() as state:
        state['windows'][1]['forward_frames'] = 17
    with pytest.raises(ValueError, match='unused'):
        reference(environment, source)
