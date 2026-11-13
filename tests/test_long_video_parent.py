"""Actual CPU native-preview frames, immutable lineage and the private loader."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
import sys
import types

import av
import numpy as np
from PIL import Image
import pytest

from nodes.long_video_plan import plan_windows
from nodes.long_video_store import ProjectStore, StaleCoordination


def parent_module():
    assert (Path(__file__).parents[1] / 'nodes/long_video_parent.py').is_file(), 'Actual parent-frame resolver is missing'
    return importlib.import_module('nodes.long_video_parent')


def movie(path, frames=328, size=(8, 6)):
    """Lossless tiny video with distinct delivery, forward and safety-tail pixels."""
    with av.open(str(path), 'w') as dest:
        stream = dest.add_stream('libx264rgb', rate=24)
        stream.width, stream.height = size
        stream.pix_fmt = 'rgb24'
        stream.options = {'crf': '0', 'preset': 'ultrafast'}
        for index in range(frames):
            color = [211, 31, 9] if index == 293 else [13, 199, 41] if index == 310 else [5, 17, 229]
            pixels = np.full((size[1], size[0], 3), color, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format='rgb24')
            for packet in stream.encode(frame):
                dest.mux(packet)
        for packet in stream.encode():
            dest.mux(packet)


def accepted_parent(tmp_path, *, frames=328):
    store = ProjectStore(tmp_path, 'scene')
    windows = plan_windows(720)
    req = store.create_take({'window': windows[0], 'parent': None, 'fingerprint': 'parent'})
    base = 'takes/' + req['take_id']
    native = store.path(base + '/native.mp4')
    movie(native, frames)
    artifacts = {'native': store.artifact(native),
                 'request': store.artifact(store.path(base + '/request.json'))}
    receipt = store.write_once(base + '/native_receipt.json', {'artifacts': deepcopy(artifacts), 'content_review': 'pending'})
    artifacts['native_receipt'] = store.artifact(receipt)
    store.record_native(req['take_id'], {'artifacts': artifacts})
    store.review(req['take_id'], True, 'The actual adult still holds the cup.', 'Actual native delivery and outgoing state.')
    return store, deepcopy(store.read()['windows'][0]), windows[1]


def child(store, parent, window, frame_policy='delivery_last', **changes):
    return {'project_name': store.name, 'window': window, 'parent': {'take_id': parent['take_id'], 'history_sha256': 'history'},
            'parent_reference': {'mode': 'replace', 'picture_id': 2, 'frame_policy': frame_policy},
            'forward_frames': 0, **changes}


def freeze(store, request, **kwargs):
    return parent_module().resolve_parent_reference(store, request,
        expected_snapshot=store.coordination_snapshot(store.read(), request['window']['index']), **kwargs)


def test_delivery_last_freezes_exact_native_frame_pixels_and_complete_receipt(tmp_path):
    store, parent, window = accepted_parent(tmp_path)
    request = child(store, parent, window)
    frozen = freeze(store, request)
    assert frozen['frame_index'] == window['source_endpoint'] - 1 == 293
    assert frozen['frame_policy'] == 'delivery_last'
    assert frozen['parent_take_id'] == parent['take_id'] and frozen['source_native'] == parent['artifacts']['native']
    assert frozen['source_review'] == parent['review'] and frozen['window'] == window
    assert (frozen['width'], frozen['height']) == (8, 6)
    assert np.array_equal(np.asarray(Image.open(store.path(frozen['png']['name']))),
                          np.full((6, 8, 3), [211, 31, 9], dtype=np.uint8))
    receipt = json.loads(store.path(frozen['receipt']['name']).read_text(encoding='utf-8'))
    assert receipt == {key: value for key, value in frozen.items() if key != 'receipt'}
    parent_module().verify_parent_reference(store, {**request, 'parent_reference_image': frozen})


@pytest.mark.parametrize('value', [
    {}, {'mode': 'latest', 'frame_policy': 'delivery_last'},
    {'mode': 'append', 'frame_policy': 'delivery_last', 'picture_id': 2},
    {'mode': 'replace', 'frame_policy': 'delivery_last'},
    {'mode': 'replace', 'frame_policy': 'delivery_last', 'picture_id': True},
    {'mode': 'replace', 'frame_policy': 'delivery_last', 'picture_id': 12},
    {'mode': 'replace', 'frame_policy': 'safety_last', 'picture_id': 2},
    {'mode': 'append', 'frame_policy': 'delivery_last', 'fallback': 'static'},
])
def test_parent_policy_rejects_unknown_fields_modes_and_ids(value):
    with pytest.raises(ValueError, match='parent_reference'):
        parent_module().validate_parent_reference(value)


@pytest.mark.parametrize('change', ['rejected', 'nonadjacent', 'review', 'native_receipt', 'native_hash', 'native_range'])
def test_resolver_rejects_wrong_or_unreviewed_actual_source(tmp_path, change):
    store, parent, window = accepted_parent(tmp_path, frames=200 if change == 'native_range' else 328)
    if change == 'nonadjacent':
        window['source_endpoint'] = 243
    elif change == 'review':
        store.path(parent['review']['name']).write_text('{}', encoding='utf-8')
    elif change == 'native_hash':
        with store.path(parent['artifacts']['native']['name']).open('ab') as source:
            source.write(b'changed')
    elif change in {'rejected', 'native_receipt'}:
        with store.transaction() as state:
            record = state['windows'][0]
            if change == 'rejected':
                record['status'] = 'rejected'
            else:
                del record['artifacts']['native_receipt']
    with pytest.raises(ValueError):
        freeze(store, child(store, parent, window))
    assert not list(store.root.glob('references/parent-*/receipt.json'))


@pytest.mark.parametrize('forward_frames', [17, 34])
def test_forward_last_requires_separate_review_for_exact_child_and_f(tmp_path, monkeypatch, forward_frames):
    store, parent, window = accepted_parent(tmp_path)
    request = child(store, parent, window, 'forward_last', forward_frames=forward_frames)
    with pytest.raises(ValueError, match='forward.*review|forward.*proof'):
        freeze(store, request)
    proof = {'requirement': {'next_window': window, 'forward_frames': forward_frames}, 'consumed_av_sha256': 'history'}
    forward = store.write_once('reviews/test-forward.json', proof)
    with store.transaction() as state:
        state['windows'][0]['forward_review'] = store.artifact(forward)
    request['parent']['forward_review'] = store.artifact(forward)
    monkeypatch.setattr(store, 'verify_forward_review', lambda current, expected=None: deepcopy(proof))
    frozen = freeze(store, request)
    assert frozen['frame_index'] == (310 if forward_frames == 17 else 327)
    assert frozen['source_forward_review'] == request['parent']['forward_review']
    assert np.array_equal(np.asarray(Image.open(store.path(frozen['png']['name'])))[0, 0],
                          [13, 199, 41] if forward_frames == 17 else [5, 17, 229])
    proof['requirement']['forward_frames'] = 34 if forward_frames == 17 else 17
    with pytest.raises(ValueError, match='forward.*match|forward.*scope'):
        freeze(store, request)


def test_delivery_policy_does_not_implicitly_select_reviewed_surplus(tmp_path, monkeypatch):
    store, parent, window = accepted_parent(tmp_path)
    request = child(store, parent, window, forward_frames=17)
    monkeypatch.setattr(store, 'verify_forward_review', lambda *_: pytest.fail('delivery policy promoted forward review'))
    assert freeze(store, request)['frame_index'] == 293


@pytest.mark.parametrize('mutation', ['native_bytes', 'parent_retry', 'stop'])
def test_source_or_coordination_change_during_decode_cannot_publish(tmp_path, monkeypatch, mutation):
    store, parent, window = accepted_parent(tmp_path)
    module = parent_module()
    decode = module.decode_native_frame
    def racing(*args, **kwargs):
        image = decode(*args, **kwargs)
        if mutation == 'native_bytes':
            with store.path(parent['artifacts']['native']['name']).open('ab') as source:
                source.write(b'raced')
        elif mutation == 'parent_retry':
            store.request_retry(0, replace_accepted=True, take_id=parent['take_id'])
        else:
            store.stop()
        return image
    monkeypatch.setattr(module, 'decode_native_frame', racing)
    with pytest.raises(ValueError):
        freeze(store, child(store, parent, window))
    assert not list(store.root.glob('references/parent-*/receipt.json'))


def test_partial_png_write_never_publishes_a_usable_receipt(tmp_path, monkeypatch):
    store, parent, window = accepted_parent(tmp_path)
    save = Image.Image.save
    def interrupted(self, path, *args, **kwargs):
        if kwargs.get('format') == 'PNG':
            Path(path).write_bytes(b'partial PNG')
            raise OSError('interrupted PNG')
        return save(self, path, *args, **kwargs)
    monkeypatch.setattr(Image.Image, 'save', interrupted)
    with pytest.raises(OSError, match='interrupted PNG'):
        freeze(store, child(store, parent, window))
    assert not list(store.root.glob('references/parent-*/receipt.json'))


@pytest.mark.parametrize('damage', ['png', 'source', 'receipt', 'parent', 'path'])
def test_frozen_reference_verification_never_reextracts_or_chooses_new_parent(tmp_path, monkeypatch, damage):
    store, parent, window = accepted_parent(tmp_path)
    request = child(store, parent, window)
    frozen = freeze(store, request)
    if damage == 'parent':
        store.request_retry(0, replace_accepted=True, take_id=parent['take_id'])
    elif damage == 'path':
        frozen['png']['name'] = '../elsewhere.png'
    else:
        artifact = frozen[{'png': 'png', 'source': 'source_native', 'receipt': 'receipt'}[damage]]
        store.path(artifact['name']).write_bytes(b'corrupt')
    monkeypatch.setattr(parent_module(), 'decode_native_frame', lambda *_: pytest.fail('frozen input silently regenerated'))
    with pytest.raises(ValueError):
        parent_module().verify_parent_reference(store, {**request, 'parent_reference_image': frozen})


def test_private_image_loader_requires_bound_project_request_and_rechecks_bytes(tmp_path, monkeypatch):
    store, parent, window = accepted_parent(tmp_path)
    request = child(store, parent, window)
    request.update(stage='native', fingerprint='child', parent_reference_image=freeze(store, request))
    take = store.create_take(request)
    monkeypatch.setitem(sys.modules, 'folder_paths', types.SimpleNamespace(get_output_directory=lambda: str(tmp_path)))
    node = parent_module().MiniMaxH3LongVideoParentImage()
    actual = node.load(json.dumps(take))[0]
    assert tuple(actual.shape) == (1, 6, 8, 3)
    assert np.array_equal(np.rint(actual.numpy()[0] * 255).astype(np.uint8),
                          np.full((6, 8, 3), [211, 31, 9], dtype=np.uint8))
    changed = deepcopy(take)
    changed['parent_reference_image']['frame_index'] -= 1
    with pytest.raises(ValueError, match='immutable'):
        node.load(json.dumps(changed))
    store.path(take['parent_reference_image']['png']['name']).unlink()
    with pytest.raises(ValueError):
        node.load(json.dumps(take))


def base_graph():
    graph = json.loads((Path(__file__).parents[1] / 'templates/u11_api.json').read_text(encoding='utf-8'))
    graph['10']['inputs'].update(mode='REF2VA', sampling_method='selflift', voice_mode='none',
        performance_preset='quality_two_stage', postprocess_mode='vosr2', vosr2_profile='manual_baseline')
    return graph


def contract():
    return {'schema_version': 1, 'fps': 24, 'total_frames': 720,
        'subject_definitions': '<Subject 8> (S2) uses <Picture 1> identity; <Picture 2> is the scene phase.',
        'summary': 'Carry the cup.', 'retention_analysis': 'Preserve the adult and cup.',
        'scale_contract': 'Adult 170 cm, cup 10 cm, one cup held in the right hand.',
        'overall_soundscape': 'Room tone.', 'non_diegetic_music': 'None.',
        'beats': [{'id': 'carry', 'start_frame': 0, 'end_frame': 720, 'action': 'Continue carrying the same cup.'}],
        'dialogues': [{'id': 'line', 'start_frame': 320, 'end_frame': 400,
            'speaker': '<Subject 8> (S2)', 'text': 'Literal <Picture 2> and <Audio 1>.', 'language': 'English'}]}


@pytest.mark.parametrize('level', ['root', 'override'])
@pytest.mark.parametrize('invalid', [False, True])
def test_canonical_parent_policy_schema_and_runtime_agree(level, invalid):
    from nodes.long_video_plan import validate_contract, compile_window
    from jsonschema import Draft202012Validator
    scene = contract()
    policy = {'mode': 'replace', 'picture_id': 2, 'frame_policy': 'delivery_last'}
    if invalid:
        policy['fallback'] = 'static'
    window = plan_windows(720)[1]
    if level == 'root':
        scene['parent_reference'] = policy
    else:
        scene['window_overrides'] = [{'new_start': window['new_start'], 'new_end': window['new_end'], 'parent_reference': policy}]
    schema = json.loads((Path(__file__).parents[1] / 'templates/u11_long_video_contract.schema.json').read_text(encoding='utf-8'))
    errors = list(Draft202012Validator(schema).iter_errors(scene))
    if invalid:
        assert errors, 'Schema accepted unknown parent_reference fields'
        with pytest.raises(ValueError, match='parent_reference'):
            validate_contract(scene, 720)
    else:
        assert not errors, errors
        valid = validate_contract(scene, 720)
        assert compile_window(valid, window, 'Actual outgoing state.')['parent_reference'] == policy


@pytest.mark.parametrize('stage', ['native', 'preview', 'finish'])
def test_real_eleven_id_picture2_replacement_binds_same_frozen_image_at_both_handshakes(tmp_path, stage, monkeypatch):
    from nodes.long_video_graph import roles, apply_window_reference_policy, build_window_graph
    from nodes.long_video_plan import compile_window, validate_contract
    store, parent, window = accepted_parent(tmp_path)
    scene = contract()
    scene.update(parent_reference={'mode': 'replace', 'picture_id': 2, 'frame_policy': 'delivery_last'},
                 reference_selection={'pictures': [1, 2, 6, 9], 'audios': []})
    compiled = compile_window(validate_contract(scene, 720), window, parent['observed_outgoing_state'])
    request = {**child(store, parent, window), **compiled, 'seed': 4, 'fingerprint': 'child'}
    request['parent_reference_image'] = freeze(store, request)
    original = base_graph()
    original['10']['inputs']['last_image_file'] = 'old-P01-storyboard.png'
    saved = deepcopy(original)
    active, director, *_ = roles(original)
    active[director]['inputs']['prompt'] = request['prompt']
    trace = apply_window_reference_policy(active[director]['inputs'], request, active, parent_store=store)
    assert len(trace['original']['pictures']) == 11
    assert trace['maps']['pictures'] == {'1': 1, '2': 2, '6': 3, '9': 4}
    assert trace['original']['pictures'][1]['binding'] == {'link': ['h3long_parent_reference', 0]}
    assert trace['original']['pictures'][1]['preprocessing_source'] == ''
    assert trace['parent_reference']['local_picture_id'] == 2
    assert trace['parent_reference']['image'] == request['parent_reference_image']
    request['prompt'] = active[director]['inputs']['prompt']
    request['reference_bindings'] = trace
    take = store.create_take(request)
    monkeypatch.setitem(sys.modules, 'folder_paths', types.SimpleNamespace(get_output_directory=lambda: str(tmp_path)))
    physical = build_window_graph(original, take, stage)
    settings = physical['10']['inputs']
    assert settings['last_image'] == ['h3long_parent_reference', 0]
    assert not settings.get('last_image_file')
    assert '<Picture 2> is the observed accepted native parent' in settings['prompt']
    assert '<Subject 8> (S2)' in settings['prompt']
    assert '<d>[English] Literal <Picture 2> and <Audio 1>.</d>' in settings['prompt']
    assert settings['prompt'].count('is the observed accepted native parent') == 1
    assert 'adult 170 cm' in settings['prompt'].lower() and 'cup 10 cm' in settings['prompt'].lower()
    assert physical['h3long_parent_reference']['class_type'] == 'MiniMaxH3LongVideoParentImage'
    assert json.loads(physical['h3long_parent_reference']['inputs']['request_json'])['parent_reference_image'] == request['parent_reference_image']
    assert physical['10']['_meta']['h3_long_video_reference_bindings'] == trace
    assert original == saved
    if stage == 'native':
        assert json.loads(physical['h3long_context']['inputs']['request_json'])['parent'] == request['parent']


@pytest.mark.parametrize('selected', [[1, 6], [1, 2]])
def test_replacement_requires_designated_original_selected_picture(tmp_path, selected):
    from nodes.long_video_graph import roles, apply_window_reference_policy
    store, parent, window = accepted_parent(tmp_path)
    request = child(store, parent, window)
    request.update(prompt='retention_analysis:\nPreserve the scene.', reference_selection={'pictures': selected, 'audios': []})
    graph, director, *_ = roles(base_graph())
    if selected == [1, 2]:
        for field in ('last_image', *[f'reference_image_{i}' for i in range(1, 10)]):
            graph[director]['inputs'].pop(field, None)
    with pytest.raises(ValueError, match='parent_reference.*selected|parent_reference.*missing'):
        apply_window_reference_policy(graph[director]['inputs'], request, graph, preflight=True)


def test_preflight_uses_unobserved_placeholder_and_real_overlay_without_media_or_publication(monkeypatch):
    from nodes.long_video_preflight import preflight
    from nodes.long_video_references import PICTURE_SLOTS
    scene = contract()
    scene['parent_reference'] = {'mode': 'append', 'frame_policy': 'delivery_last'}
    graph = base_graph()
    for slot in PICTURE_SLOTS[4:]:
        graph['10']['inputs'].pop(slot, None)
    monkeypatch.setattr(parent_module(), 'decode_native_frame', lambda *_: pytest.fail('preflight decoded parent media'))
    monkeypatch.setattr(parent_module(), 'resolve_parent_reference', lambda *_: pytest.fail('preflight froze real source'))
    before = deepcopy(graph)
    report = preflight(graph, {'target_seconds': 30, 'scene_contract': scene, 'project_name': 'scene'}, check_encoder=False)
    assert [len(window['reference_bindings']['selection']['pictures']) for window in report['windows']] == [4, 5, 5]
    for window in report['windows'][1:]:
        assert window['reference_bindings']['parent_reference']['image']['observation'] == 'UNOBSERVED_STATIC_PREFLIGHT_ONLY_UNAPPROVED'
    assert graph == before and report['queued'] is False and report['media_approved'] is False


def test_original_append_continuity_slot_quota_rejects_before_hash_encoder_or_take(tmp_path):
    from nodes.long_video_service import configure_project
    store = ProjectStore(tmp_path, 'scene')
    scene = contract()
    scene['parent_reference'] = {'mode': 'append', 'frame_policy': 'delivery_last'}
    graph = base_graph()
    # Exactly nine ordinary static references are already the physical maximum.
    for field in ('first_image', 'last_image'):
        graph['10']['inputs'].pop(field, None)
    def expensive(*args):
        pytest.fail('expensive work happened before continuity-slot rejection')
    with pytest.raises(ValueError, match='continuity.*slot|slot.*continuity|连续.*位置'):
        configure_project(store, graph, {'project_name': 'scene', 'target_seconds': 30, 'scene_contract': scene},
                          asset_freezer=expensive, encoder_checker=expensive, export_asset_freezer=expensive)
    assert store.read()['windows'] == []


def test_append_with_explicit_advanced_selection_is_rejected_during_pure_validation():
    from nodes.long_video_plan import validate_contract
    scene = contract()
    scene.update(parent_reference={'mode': 'append', 'frame_policy': 'delivery_last'},
                 reference_selection={'pictures': [1, 2], 'audios': []})
    with pytest.raises(ValueError, match='append.*reference_selection|reference_selection.*replace'):
        validate_contract(scene, 720)


def test_frozen_image_tampering_blocks_create_take_without_readable_receipt_fallback(tmp_path):
    store, parent, window = accepted_parent(tmp_path)
    request = child(store, parent, window)
    request.update(fingerprint='child', parent_reference_image=freeze(store, request))
    store.path(request['parent_reference_image']['png']['name']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash|Artifact'):
        store.create_take(request)


def test_private_image_signature_always_invalidates_actual_comfy_cache_contract():
    """Use the installed cache's real signature builder without importing Comfy."""
    import ast
    import asyncio
    from collections.abc import Mapping, Sequence
    import itertools
    from abc import ABC, abstractmethod
    cache_path = Path('D:/ComfyUI_windows_portable-G313/ComfyUI/comfy_execution/caching.py')
    if not cache_path.is_file():
        pytest.skip('Installed Comfy cache source is unavailable')
    names = {'CacheKeySet', 'Unhashable', 'to_hashable', 'CacheKeySetInputSignature'}
    parsed = ast.parse(cache_path.read_text(encoding='utf-8'))
    definitions = [node for node in parsed.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    node_type = parent_module().MiniMaxH3LongVideoParentImage
    namespace = {'ABC': ABC, 'abstractmethod': abstractmethod, 'Mapping': Mapping, 'Sequence': Sequence,
                 'itertools': itertools, 'nodes': types.SimpleNamespace(NODE_CLASS_MAPPINGS={'Parent': node_type}),
                 'include_unique_id_in_input': lambda _: False, 'is_link': lambda _: False}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(cache_path), 'exec'), namespace)
    dynamic = types.SimpleNamespace(has_node=lambda _: True, get_node=lambda _: {'class_type': 'Parent', 'inputs': {'request_json': '{}'}})
    class Changes:
        async def get(self, node_id):
            return [node_type.IS_CHANGED(request_json='{}')]
    async def signature():
        cache = namespace['CacheKeySetInputSignature'](dynamic, ['image'], Changes())
        return await cache.get_node_signature(dynamic, 'image')
    first, second = asyncio.run(signature()), asyncio.run(signature())
    assert first != second
    assert {first: 'cached PNG'}.get(second) is None


class Backend:
    def __init__(self):
        self.jobs, self.submitted = {}, []

    def submit(self, prompt_id, graph):
        self.jobs[prompt_id] = 'queued'
        self.submitted.append(graph)

    def status(self, prompt_id):
        return self.jobs.get(prompt_id, 'missing')


def mark_native(store, backend, *, accepted=True):
    state = store.read()
    job = state['active_job']
    record = store.active(state, job['take_id'])
    base = 'takes/' + record['take_id']
    movie(store.path(base + '/native.mp4'), record['window']['sample_frames'])
    clean = store.path(base + '/clean_video.safetensors')
    clean.write_bytes(b'clean AV lifecycle fixture; never passed to a model')
    artifacts = {'native': store.artifact(store.path(base + '/native.mp4')), 'clean_video': store.artifact(clean),
                 'request': store.artifact(store.path(base + '/request.json'))}
    receipt = store.write_once(base + '/native_receipt.json', {'artifacts': deepcopy(artifacts), 'content_review': 'pending'})
    artifacts['native_receipt'] = store.artifact(receipt)
    store.record_native(record['take_id'], {'artifacts': artifacts})
    backend.jobs[job['prompt_id']] = 'completed'
    # Queue reconciliation normally owns this removal; retain its durable job.
    from nodes.long_video_service import Coordinator
    Coordinator(store, backend).reconcile_job()
    store.review(record['take_id'], accepted, 'Actual adult holds one cup.', 'Native delivery/outgoing state.')
    return store.read()['windows'][record['window']['index']]


def continuation_runner(tmp_path, mode='replace', seconds=30):
    from nodes.long_video_service import Coordinator, configure_project
    store = ProjectStore(tmp_path, 'scene')
    scene = contract()
    scene['total_frames'] = seconds * 24
    scene['beats'][0]['end_frame'] = seconds * 24
    graph = base_graph()
    if mode == 'replace':
        scene.update(parent_reference={'mode': 'replace', 'picture_id': 2, 'frame_policy': 'delivery_last'},
                     reference_selection={'pictures': [1, 2, 6, 9], 'audios': []})
    else:
        from nodes.long_video_references import PICTURE_SLOTS
        scene['parent_reference'] = {'mode': 'append', 'frame_policy': 'delivery_last'}
        for field in PICTURE_SLOTS[4:]:
            graph['10']['inputs'].pop(field, None)
    configure_project(store, graph, {'project_name': 'scene', 'target_seconds': seconds, 'scene_contract': scene},
        asset_freezer=lambda _: {}, export_asset_freezer=lambda _: {}, encoder_checker=lambda: {})
    backend = Backend()
    runner = Coordinator(store, backend, history_provider=lambda *_: 'actual-current-clean-history')
    assert runner.step()
    parent = mark_native(store, backend)
    export = store.path('takes/' + parent['take_id'] + '/approved-export.mp4')
    audio = store.path('takes/' + parent['take_id'] + '/delivery.wav')
    export.write_bytes(b'export fixture')
    audio.write_bytes(b'audio fixture')
    with store.transaction() as state:
        state['windows'][0]['exports'][state['config']['export_fingerprint']] = {
            'video': store.artifact(export), 'audio': store.artifact(audio)}
    return store, backend, runner


def cpu_export(store, index):
    """Publish a tiny real CPU delivery fixture from the accepted native pixels."""
    import wave
    record = store.read()['windows'][index]
    window = record['window']
    output = store.path('takes/' + record['take_id'] + '/cpu-delivery.mp4')
    audio = store.path('takes/' + record['take_id'] + '/cpu-delivery.wav')
    with av.open(str(store.path(record['artifacts']['native']['name']))) as source, av.open(str(output), 'w') as dest:
        original = source.streams.video[0]
        stream = dest.add_stream('libx264rgb', rate=24)
        stream.width, stream.height, stream.pix_fmt = original.width, original.height, 'rgb24'
        stream.options = {'crf': '0', 'preset': 'ultrafast'}
        for number, frame in enumerate(source.decode(video=0)):
            if number >= window['local_new_end']:
                break
            if number >= window['local_new_start']:
                image = av.VideoFrame.from_ndarray(frame.to_ndarray(format='rgb24'), format='rgb24')
                for packet in stream.encode(image):
                    dest.mux(packet)
        for packet in stream.encode():
            dest.mux(packet)
    with wave.open(str(audio), 'wb') as sound:
        sound.setnchannels(2)
        sound.setsampwidth(2)
        sound.setframerate(32000)
        sound.writeframes(b'\0' * round((window['new_end'] - window['new_start']) * 32000 / 24) * 4)
    with store.transaction() as state:
        state['windows'][index]['exports'][state['config']['export_fingerprint']] = {
            'video': store.artifact(output), 'audio': store.artifact(audio),
            'frames': window['new_end'] - window['new_start']}


def configure_duration(store, seconds):
    from nodes.long_video_service import configure_project
    scene = contract()
    scene['total_frames'] = seconds * 24
    scene['beats'][0]['end_frame'] = seconds * 24
    scene.update(parent_reference={'mode': 'replace', 'picture_id': 2, 'frame_policy': 'delivery_last'},
                 reference_selection={'pictures': [1, 2, 6, 9], 'audios': []})
    configure_project(store, base_graph(), {'project_name': store.name, 'target_seconds': seconds, 'scene_contract': scene},
        asset_freezer=lambda _: {}, export_asset_freezer=lambda _: {}, encoder_checker=lambda: {})


def test_extending_accepted_twenty_second_tail_freezes_for_the_new_exact_child_window(tmp_path, monkeypatch):
    store, backend, runner = continuation_runner(tmp_path, seconds=20)
    cpu_export(store, 0)
    assert runner.step()
    old_tail = mark_native(store, backend)
    cpu_export(store, 1)
    before = deepcopy(store.read())
    old_tail = before['windows'][1]
    old_image = deepcopy(old_tail['parent_reference_image'])
    assert old_tail['window']['new_end'] == 480 and old_tail['window']['sample_frames'] == 277
    assert all(item['status'] == 'accepted' and item['exports'] for item in before['windows'])
    old_png = store.path(old_image['png']['name']).read_bytes()
    for item in before['windows']:
        exported = item['exports'][before['config']['export_fingerprint']]
        with av.open(str(store.path(exported['video']['name']))) as source:
            assert sum(1 for _ in source.decode(video=0)) == exported['frames']
    decode, calls = parent_module().decode_native_frame, []
    def counted(*args):
        calls.append(args[1])
        return decode(*args)
    monkeypatch.setattr(parent_module(), 'decode_native_frame', counted)
    configure_duration(store, 30)
    queued = len(backend.submitted)
    assert runner.step()
    current = store.read()
    assert len(backend.submitted) == queued + 1 and current['active_job']['stage'] == 'native'
    assert current['windows'][0] == before['windows'][0], 'Complete accepted/exported W00 was replaced'
    tail = current['windows'][1]
    assert tail['take_id'] != old_tail['take_id']
    assert tail['window']['new_end'] == 549 and tail['window']['sample_frames'] == 328
    assert tail['parent'] == old_tail['parent'] and calls == [293]
    assert tail['parent_reference_image']['window'] == tail['window']
    assert tail['parent_reference_image']['source_native'] == old_image['source_native']
    assert tail['parent_reference_image']['png']['name'] != old_image['png']['name']
    assert store.path(old_image['png']['name']).read_bytes() == old_png
    assert any(item['take_id'] == old_tail['take_id'] for item in current['superseded'])


@pytest.mark.parametrize('damage', ['missing', 'corrupt'])
def test_unchanged_twenty_second_child_window_cannot_reextract_damaged_frozen_png(tmp_path, monkeypatch, damage):
    store, backend, runner = continuation_runner(tmp_path, seconds=20)
    assert runner.step()
    tail = mark_native(store, backend)
    cpu_export(store, 1)
    image = store.path(tail['parent_reference_image']['png']['name'])
    if damage == 'missing':
        image.unlink()
    else:
        image.write_bytes(b'corrupt frozen PNG')
    configure_duration(store, 20)
    queued = len(backend.submitted)
    monkeypatch.setattr(parent_module(), 'resolve_parent_reference', lambda *_args, **_kwargs: pytest.fail('unchanged frozen PNG re-extracted'))
    with pytest.raises(ValueError, match='Artifact|missing|hash'):
        runner.step()
    assert len(backend.submitted) == queued and store.read()['windows'][1]['take_id'] == tail['take_id']
    assert not image.exists() if damage == 'missing' else image.read_bytes() == b'corrupt frozen PNG'


def test_repair_for_twenty_second_tail_rejects_changed_thirty_second_window_before_extraction(tmp_path, monkeypatch):
    store, backend, runner = continuation_runner(tmp_path, seconds=20)
    assert runner.step()
    tail = mark_native(store, backend, accepted=False)
    old_image = deepcopy(tail['parent_reference_image'])
    store.request_repair(1, [[20, 24]], take_id=tail['take_id'])
    configure_duration(store, 30)
    queued = len(backend.submitted)
    monkeypatch.setattr(parent_module(), 'resolve_parent_reference', lambda *_args, **_kwargs: pytest.fail('changed-window repair extracted a new parent'))
    with pytest.raises(ValueError, match='Repair.*window.*changed.*(select|base generation)'):
        runner.step()
    state = store.read()
    assert len(backend.submitted) == queued and state['repair']['source_take'] == tail['take_id']
    assert state['windows'][1]['status'] == 'invalidated'
    store.verify({'old_parent_png': old_image['png']})


def test_coordinator_freezes_actual_binding_before_fingerprint_and_reuses_it_without_extraction(tmp_path, monkeypatch):
    store, backend, runner = continuation_runner(tmp_path)
    before = deepcopy(store.read()['config'])
    assert runner.step()
    record = store.read()['windows'][1]
    assert record['parent_reference_image']['frame_index'] == 293
    assert record['reference_bindings']['original']['pictures'][1]['binding'] == {'link': ['h3long_parent_reference', 0]}
    assert record['parent_reference_image']['source_history_sha256'] == record['parent']['history_sha256']
    assert backend.submitted[-1]['10']['inputs']['last_image'] == ['h3long_parent_reference', 0]
    assert store.read()['config'] == before, 'Global graph/catalog/asset identity was expanded'
    monkeypatch.setattr(parent_module(), 'decode_native_frame', lambda *_: pytest.fail('resumed take extracted another frame'))
    # A restart removed the queued job; retry the same immutable take and PNG.
    state = store.read()
    backend.jobs.pop(state['active_job']['prompt_id'])
    with store.transaction() as current:
        current.pop('active_job')
    assert runner.step()
    assert store.read()['windows'][1]['take_id'] == record['take_id']
    assert store.read()['windows'][1]['parent_reference_image'] == record['parent_reference_image']


@pytest.mark.parametrize('stage', ['native', 'preview', 'finish'])
def test_corrupt_existing_frozen_input_blocks_all_stage_graphs_without_reextracting(tmp_path, stage, monkeypatch):
    from nodes.long_video_graph import build_window_graph
    store, backend, runner = continuation_runner(tmp_path)
    runner.step()
    record = store.read()['windows'][1]
    request = json.loads(store.path('takes/' + record['take_id'] + '/request.json').read_text(encoding='utf-8'))
    store.path(record['parent_reference_image']['png']['name']).unlink()
    monkeypatch.setattr(parent_module(), 'decode_native_frame', lambda *_: pytest.fail('existing frozen PNG silently regenerated'))
    with pytest.raises(ValueError, match='Artifact|hash|missing'):
        build_window_graph(store.read()['config']['base_graph'], request, stage, parent_store=store)


def test_parent_snapshot_race_before_create_take_never_queues_stale_frozen_image(tmp_path, monkeypatch):
    store, backend, runner = continuation_runner(tmp_path)
    original = store.create_take
    calls = len(backend.submitted)
    def racing(spec, **kwargs):
        assert spec.get('parent_reference_image'), 'Coordinator did not resolve the parent before take creation'
        parent = store.read()['windows'][0]
        store.request_retry(0, replace_accepted=True, take_id=parent['take_id'])
        return original(spec, **kwargs)
    monkeypatch.setattr(store, 'create_take', racing)
    assert runner.step() is True
    assert len(backend.submitted) == calls
    assert len(store.read()['windows']) == 1


def test_four_original_static_images_append_one_real_frozen_local_picture5(tmp_path):
    from nodes.long_video_references import snapshot_bindings
    store, backend, runner = continuation_runner(tmp_path, mode='append')
    original = deepcopy(store.read()['config']['base_graph'])
    first = store.read()['windows'][0]
    assert len(first['reference_bindings']['original']['pictures']) == 4
    assert runner.step()
    child_record = store.read()['windows'][1]
    trace = child_record['reference_bindings']
    assert trace['maps']['pictures'] == {str(index): index for index in range(1, 6)}
    assert trace['parent_reference']['local_picture_id'] == 5
    assert trace['parent_reference']['destination'] == 'reference_image_3'
    settings = backend.submitted[-1]['10']['inputs']
    for field in ('first_image', 'last_image', 'reference_image_1', 'reference_image_2'):
        assert settings[field] == original['10']['inputs'][field]
    assert settings['reference_image_3'] == ['h3long_parent_reference', 0]
    prompt_link = settings['prompt']
    assert '<Picture 5> is the observed accepted native parent' in backend.submitted[-1][prompt_link[0]]['inputs']['value']
    assert store.read()['config']['base_graph'] == original


def test_repair_retains_same_frozen_reference_and_verifies_it_before_accepting_repair(tmp_path, monkeypatch):
    store, backend, runner = continuation_runner(tmp_path)
    runner.step()
    failed = mark_native(store, backend, accepted=False)
    frozen = deepcopy(failed['parent_reference_image'])
    monkeypatch.setattr(parent_module(), 'decode_native_frame', lambda *_: pytest.fail('repair chose another parent image'))
    store.request_repair(1, [[20, 24]], take_id=failed['take_id'])
    assert runner.step()
    repaired = store.read()['windows'][1]
    assert repaired['take_id'] != failed['take_id']
    assert repaired['parent_reference_image'] == frozen
    assert repaired['parent'] == failed['parent'] and repaired['repair']['source_take'] == failed['take_id']
    assert store.path(frozen['png']['name']).is_file()
    # A subsequent repair requires the same uncorrupted frozen reference.
    mark_native(store, backend, accepted=False)
    store.path(frozen['png']['name']).unlink()
    with pytest.raises(ValueError, match='Artifact|hash|missing'):
        store.request_repair(1, [[20, 24]], take_id=repaired['take_id'])


def test_explicit_new_base_retry_can_resolve_current_source_and_retains_old_files(tmp_path, monkeypatch):
    store, backend, runner = continuation_runner(tmp_path)
    runner.step()
    old = mark_native(store, backend, accepted=False)
    png = store.path(old['parent_reference_image']['png']['name'])
    png.write_bytes(b'damaged old frozen evidence; retained for diagnostics')
    store.request_retry(1, take_id=old['take_id'])
    decode, calls = parent_module().decode_native_frame, []
    def counted(*args):
        calls.append(args[1])
        return decode(*args)
    monkeypatch.setattr(parent_module(), 'decode_native_frame', counted)
    assert runner.step()
    current = store.read()['windows'][1]
    assert calls == [293]
    assert current['parent_reference_image']['png']['name'] != old['parent_reference_image']['png']['name']
    assert current['fingerprint'] != old['fingerprint'] and current['seed'] != old['seed']
    assert png.read_bytes().startswith(b'damaged old frozen evidence')


def test_frozen_image_history_cannot_be_detached_from_clean_av_parent_binding(tmp_path):
    store, parent, window = accepted_parent(tmp_path)
    request = child(store, parent, window)
    request['parent_reference_image'] = freeze(store, request)
    request['parent']['history_sha256'] = 'some-other-clean-history'
    with pytest.raises(ValueError, match='current adjacent accepted source'):
        parent_module().verify_parent_reference(store, request)


def test_ordinary_replacement_does_not_restore_old_preprocessing_filename_after_overlay(tmp_path):
    from nodes.long_video_graph import build_window_graph
    from nodes.long_video_plan import compile_window, validate_contract
    from nodes.long_video_references import PICTURE_SLOTS
    store, parent, window = accepted_parent(tmp_path)
    scene = contract()
    scene['parent_reference'] = {'mode': 'replace', 'picture_id': 2, 'frame_policy': 'delivery_last'}
    graph = base_graph()
    for field in PICTURE_SLOTS[4:]:
        graph['10']['inputs'].pop(field, None)
    graph['10'].setdefault('_meta', {})['h3_long_video_entry'] = 'director_duration'
    graph['10']['inputs']['last_image_file'] = 'old-P01.png'
    request = {**child(store, parent, window), **compile_window(validate_contract(scene, 720), window, 'Actual state.'),
               'seed': 4, 'fingerprint': 'child'}
    request['parent_reference_image'] = freeze(store, request)
    take = store.create_take(request)
    physical = build_window_graph(graph, take, parent_store=store)
    assert physical['10']['inputs']['last_image'] == ['h3long_parent_reference', 0]
    assert not physical['10']['inputs'].get('last_image_file')
    assert physical['10']['_meta']['h3_long_video_reference_source_files']['last_image'] == ''


def test_runtime_reuse_checks_same_name_source_and_png_even_after_previous_load(tmp_path, monkeypatch):
    store, parent, window = accepted_parent(tmp_path)
    request = child(store, parent, window)
    request.update(stage='native', fingerprint='child', parent_reference_image=freeze(store, request))
    take = store.create_take(request)
    monkeypatch.setitem(sys.modules, 'folder_paths', types.SimpleNamespace(get_output_directory=lambda: str(tmp_path)))
    node = parent_module().MiniMaxH3LongVideoParentImage()
    node.load(json.dumps(take))
    with store.path(parent['artifacts']['native']['name']).open('ab') as source:
        source.write(b'same-name source replacement after IMAGE execution')
    with pytest.raises(ValueError, match='Artifact|hash'):
        node.load(json.dumps(take))


def test_soft_parent_instruction_never_edits_or_counts_a_literal_spoken_copy():
    module = parent_module()
    original = ('subject_definitions:\n<Subject 1> (S1).\n\nsummary:\nWalk.\n\n'
        'retention_analysis:\nKeep the cup.\n\ndetailed_description:\n'
        'Say <d>retention_analysis:\nKeep my literal words.\n\ndetailed_description:\nExactly.</d>\n\n'
        'overall_soundscape:\nRoom tone.\n\nnon_diegetic_music:\nNone.')
    once = module.retain_parent_instruction(original, 2)
    instruction = once[once.index('<Picture 2> is the observed'):once.index('\n\ndetailed_description:')]
    spoken = original.replace('Keep the cup.', 'Keep the cup.\n<d>' + instruction + '</d>')
    actual = module.retain_parent_instruction(spoken, 2)
    assert actual.count(instruction) == 2
    literal = spoken[spoken.index('<d>'):spoken.index('</d>') + 4]
    assert literal in actual


def test_real_forward_review_and_native_cpu_frame327_preserve_existing_exception(environment):
    from test_long_video_store_protocol import forward_parent
    from nodes.long_video_av import slice_fingerprint
    store = environment
    record, clean = forward_parent(store, forward=34)
    window = store.read()['plan'][1]
    request = child(store, record, window, 'forward_last', forward_frames=34)
    request['parent']['history_sha256'] = slice_fingerprint(clean, 294, 39, 34)
    with pytest.raises(ValueError, match='forward.*review|forward.*proof'):
        freeze(store, request)
    store.review_forward(record['take_id'], 'Actual native frames294–327 with sound checked.', 'Actual forward endpoint state at frame327.')
    request['parent']['forward_review'] = store.read()['windows'][0]['forward_review']
    frozen = freeze(store, request)
    assert frozen['frame_index'] == 327
    assert frozen['source_forward_review'] == request['parent']['forward_review']
    with av.open(str(store.path(record['artifacts']['native']['name']))) as source:
        for index, frame in enumerate(source.decode(video=0)):
            if index == 327:
                expected = np.asarray(frame.to_image())
                break
    assert np.array_equal(np.asarray(Image.open(store.path(frozen['png']['name']))), expected)
    parent_module().verify_parent_reference(store, {**request, 'parent_reference_image': frozen})
    request['parent']['forward_review'] = record['review']
    with pytest.raises(ValueError, match='forward.*review|forward.*proof'):
        parent_module().verify_parent_reference(store, {**request, 'parent_reference_image': frozen})


def test_inert_first_window_still_enforces_actual_nine_picture_conditioning_limit():
    from nodes.long_video_preflight import preflight
    scene = contract()
    scene.update(total_frames=120, dialogues=[],
        beats=[{'id': 'carry', 'start_frame': 0, 'end_frame': 120, 'action': 'Carry the cup.'}],
        parent_reference={'mode': 'replace', 'picture_id': 2, 'frame_policy': 'delivery_last'})
    with pytest.raises(ValueError, match='9.*pictures|pictures.*9'):
        preflight(base_graph(), {'project_name': 'scene', 'target_seconds': 5, 'scene_contract': scene}, check_encoder=False)


# Exercise the actual CPU clean-AV/native-review machinery for the scoped tail.
from test_long_video_runtime import environment
