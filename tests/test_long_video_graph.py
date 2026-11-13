from copy import deepcopy
import json
from pathlib import Path

import pytest

from nodes.long_video_graph import build_window_graph, generation_settings, freeze_assets


def base_graph():
    graph = json.loads((Path(__file__).parents[1] / 'templates/u11_api.json').read_text(encoding='utf-8'))
    graph['10']['inputs'].update(mode='REF2VA', sampling_method='selflift',
        performance_preset='质量优先二采样', postprocess_mode='vosr2', vosr2_profile='manual_baseline')
    return graph


def request():
    return {'project_name': 'test', 'take_id': 'a' * 32,
            'window': {'index': 1, 'new_start': 294, 'new_end': 549, 'context_frames': 39,
                       'local_new_start': 39, 'local_new_end': 294, 'sample_frames': 328,
                       'source_endpoint': 294, 'endpoint': 294},
            'prompt': 'subject_definitions:\nS\n\nsummary:\nwindow', 'seed': 42,
            'parent': {'take_id': 'b' * 32}, 'fingerprint': 'native-key'}


@pytest.mark.parametrize('stage',['native','preview','finish'])
def test_automatic_original_fields_keep_linked_image_crop_hints_in_the_physical_graph(stage,monkeypatch):
    from nodes.long_video_entry import _automatic_recipe
    from nodes.long_video_graph import _picture_preprocessing,roles
    from nodes.long_video_plan import plan_windows,compile_window
    graph=base_graph();settings=graph['10']['inputs']
    settings.update(mode='I2VA',duration=30,prompt='The woman keeps walking.',voice_mode='none',
                    first_image=['11',0],first_image_file='five-view-identity.png')
    monkeypatch.setattr('nodes.long_video_entry.preflight',lambda *_:None)
    prepared,config=_automatic_recipe((graph,'10',30))
    # The linked image is the asset. The shadow filename is only a crop hint,
    # so hashing must not open it as a second (possibly stale) asset.
    hash_graph=roles(prepared)[0]
    assert 'first_image_file' not in hash_graph['10']['inputs']
    window=plan_windows(720)[0];compiled=compile_window(config['scene_contract'],window)
    item={**request(),**compiled,'window':window,'parent':None}
    physical=build_window_graph(prepared,item,stage)
    actual=physical['10']['inputs']
    assert actual['first_image_file']=='five-view-identity.png'
    assert actual['first_image']==['11',0]
    assert _picture_preprocessing(actual['prompt'],'first_image',actual['first_image_file'],1)=={'cropped':True,'dropped':False}


def test_native_graph_retains_original_sampler_and_both_stream_context():
    original = base_graph()
    graph = build_window_graph(original, request(), 'native')
    assert original == base_graph()
    assert graph['21']['class_type'] == 'MiniMaxH3TwoStageSampler'
    assert graph['21']['inputs']['latent_image'] == ['h3long_context', 0]
    assert graph['h3long_context']['inputs']['latent'] == ['16', 1]
    assert graph['h3long_native']['inputs']['prepared'] == ['h3long_context', 0]
    assert graph['h3long_native']['inputs']['latent'] == ['21', 0]
    assert not any(n['class_type'] in ('MiniMaxH3StreamingVideoCombine', 'MiniMaxH3U11Output') for n in graph.values())
    assert graph['15']['inputs']['community_loras'] == original['15']['inputs']['community_loras']
    assert graph['10']['inputs']['voice_mode'] == original['10']['inputs']['voice_mode']


def test_finish_loads_accepted_clean_av_and_never_runs_h3():
    graph = build_window_graph(base_graph(), request(), 'finish')
    assert graph['21']['class_type'] == 'MiniMaxH3LongVideoLoadAV'
    assert graph['24']['class_type'] == 'MiniMaxH3LongVideoExport'
    assert not any('Sampler' in n['class_type'] for n in graph.values())
    assert graph['29']['class_type'] == 'MiniMaxH3ColorGuard'
    assert graph['10']['inputs']['vosr2_profile'] == 'manual_baseline'


def test_export_settings_do_not_change_native_generation_fingerprint():
    base = base_graph()
    base['10']['inputs']['resolution_preset'] = '2K QHD'
    changed = deepcopy(base)
    changed['24']['inputs']['quality'] = 16
    changed['10']['inputs'].update(width=3840, height=2160, resolution_preset='4K UHD')
    assert generation_settings(base) == generation_settings(changed)
    changed['15']['inputs']['community_loras'] = '仅 U22 细节链'
    assert generation_settings(base) != generation_settings(changed)


def test_long_video_does_not_silently_switch_unsupported_quality_recipe():
    graph = base_graph()
    graph['10']['inputs']['sampling_method'] = 'standard'
    with pytest.raises(ValueError, match='SelfLift'):
        build_window_graph(graph, request(), 'native')


def test_long_video_normalizes_original_standard_author_selection_without_changing_source():
    graph = base_graph()
    graph['10']['inputs'].update(sampling_method='standard', community_lora_profile='combat')
    saved = deepcopy(graph)
    actual = build_window_graph(graph, request(), 'native')
    assert actual['10']['inputs']['sampling_method'] == 'selflift'
    assert actual['10']['inputs']['community_lora_profile'] == 'combat'
    assert graph == saved


def test_actual_u11_switch_prunes_inactive_sampler_and_guide():
    graph = base_graph()
    for node_id in ('16', '17', '21', '22', '23'):
        duplicate = deepcopy(graph[node_id])
        duplicate['inputs'] = {k: [v[0] + '_fl', v[1]] if isinstance(v, list) and v[0] in {'16','17','21'} else v
                               for k,v in duplicate['inputs'].items()}
        graph[node_id + '_fl'] = duplicate
    graph['switch'] = {'class_type': 'ComfySwitchNode', 'inputs': {
        'on_false': ['22',0], 'on_true': ['22_fl',0], 'switch': ['10',7]}}
    graph['29']['inputs']['images'] = ['switch',0]
    result = build_window_graph(graph, request(), 'native')
    assert '21' in result and '21_fl' not in result
    assert '16_fl' not in result


def test_export_aspect_is_part_of_the_actual_native_canvas():
    graph = base_graph()
    changed = deepcopy(graph)
    changed['10']['inputs']['aspect_ratio'] = '9:16'
    assert generation_settings(graph) != generation_settings(changed)


def test_payload_records_stage_and_reserved_names_cannot_collide():
    graph = build_window_graph(base_graph(), request(), 'preview')
    assert json.loads(graph['h3long_native']['inputs']['request_json'])['stage'] == 'preview'
    base = base_graph()
    base['h3long_native'] = {'class_type': 'Anything', 'inputs': {}}
    with pytest.raises(ValueError, match='reserved'):
        build_window_graph(base, request())


def voice_graph():
    base = base_graph()
    base['10']['inputs'].update(voice_mode='h3_reference',
        voice_reference_audio=['13', 0], voice_reference_audio_2_file='second.wav',
        voice_reference_audio_3_file='third.wav', voice_reference_name_1='First speaker',
        voice_reference_name_2='Second speaker', voice_reference_name_3='Third speaker',
        reference_transcript='Historical sample words.')
    return base


@pytest.mark.parametrize('stage', ['native', 'preview', 'finish'])
def test_noncontiguous_selection_compacts_actual_files_links_and_records_binding_maps(stage):
    base = voice_graph()
    base['10']['inputs']['reference_image_7_file'] = 'stale-shadowed.png'
    value = request()
    value.update(reference_selection={'pictures': [9, 2], 'audios': [3, 1]},
        prompt='<Subject 8> (S2) from <Picture 9>, scene <Picture 2>, timbre <Audio 3>. '
               '<d><Picture 9> <Audio 3></d>')
    before = deepcopy(base)
    graph = build_window_graph(base, value, stage)
    settings = graph['10']['inputs']
    assert base == before
    assert settings['first_image'] == ['37', 0] and settings['last_image'] == ['12', 0]
    assert not any(key.startswith('reference_image_') for key in settings)
    assert settings['voice_reference_audio_file'] == 'third.wav'
    assert settings['voice_reference_audio_2'] == ['13', 0]
    assert settings['voice_reference_name_1'] == 'Third speaker'
    assert settings['voice_reference_name_2'] == 'First speaker'
    assert not any('audio_3' in key or key == 'voice_reference_name_3' for key in settings)
    assert settings['reference_transcript'] == ''
    assert '<Subject 8> (S2) from <Picture 1>, scene <Picture 2>, timbre <Audio 1>.' in settings['prompt']
    assert '<d><Picture 9> <Audio 3></d>' in settings['prompt']
    trace = graph['10']['_meta']['h3_long_video_reference_bindings']
    assert trace['maps'] == {'pictures': {'9': 1, '2': 2}, 'audios': {'3': 1, '1': 2}}
    assert trace['original']['pictures'][0]['slot'] == 'first_image'
    assert trace['original']['pictures'][8]['slot'] == 'reference_image_7'
    assert trace['selected']['pictures'][0]['original_id'] == 9
    node = '24' if stage == 'finish' else 'h3long_native'
    payload = json.loads(graph[node]['inputs']['request_json'])
    assert {key: val for key, val in payload.items() if key != 'stage'} == value
    assert '31' not in graph and '37' in graph


def test_compiled_prompt_reference_maps_are_not_applied_twice():
    value = request()
    value.update(reference_selection={'pictures': [9, 2], 'audios': [3]},
                 prompt='<Subject 8> (S2) uses <Picture 1> and <Audio 1>.',
                 audit={'prompt_reference_space': 'window',
                        'reference_maps': {'pictures': {'9': 1, '2': 2}, 'audios': {'3': 1}}})
    graph = build_window_graph(voice_graph(), value)
    assert graph['10']['inputs']['prompt'] == value['prompt']


@pytest.mark.parametrize('selection', [
    {'pictures': [12], 'audios': []}, {'pictures': [1, 1], 'audios': []},
    {'pictures': [True], 'audios': []}, {'pictures': [1], 'audios': [4]},
    {'pictures': [1], 'audios': [], 'guess': True},
])
def test_invalid_or_missing_selected_bindings_fail(selection):
    value = {**request(), 'reference_selection': selection}
    with pytest.raises(ValueError):
        build_window_graph(voice_graph(), value)


def test_original_reference_gaps_and_dangling_directing_markers_fail():
    base = voice_graph()
    base['10']['inputs'].pop('reference_image_2')
    with pytest.raises(ValueError, match='gap|contiguous'):
        build_window_graph(base, {**request(), 'reference_selection': {'pictures': [1], 'audios': []}})
    with pytest.raises(ValueError, match='Picture 11|dangling'):
        build_window_graph(voice_graph(), {**request(), 'prompt': 'Use <Picture 11>.',
            'reference_selection': {'pictures': [1], 'audios': []}})


@pytest.mark.parametrize('stage', ['native', 'preview', 'finish'])
def test_quiet_owned_dialogue_contract_omits_all_local_voice_fields(stage):
    value = request()
    value.update(voice_policy='dialogue_only', dialogue_ids=[],
        prompt='<Subject 1> (S1): same adult; <Audio 1> is voice timbre reference only.')
    graph = build_window_graph(voice_graph(), value, stage)
    settings = graph['10']['inputs']
    assert settings['voice_mode'] == 'none'
    assert not any(key.startswith('voice_reference_') for key in settings)
    assert settings['reference_transcript'] == ''
    assert '<Subject 1> (S1): same adult' in settings['prompt']
    assert '<Audio' not in settings['prompt'] and 'automatically generated' not in settings['prompt']
    assert '13' not in graph


def test_legacy_and_owned_speech_inherit_original_reference_bindings():
    base = voice_graph()
    legacy = build_window_graph(base, request())
    spoken = build_window_graph(base, {**request(), 'voice_policy': 'dialogue_only', 'dialogue_ids': ['line']})
    assert legacy['10']['inputs'] == spoken['10']['inputs']
    assert legacy['10']['inputs']['voice_mode'] == 'h3_reference'


def test_quiet_policy_requires_an_explicit_owned_dialogue_list():
    with pytest.raises(ValueError, match='dialogue_ids'):
        build_window_graph(voice_graph(), {**request(), 'voice_policy': 'dialogue_only'})


@pytest.mark.parametrize('endpoint,source', [('first_image', '11'), ('last_image', '12')])
def test_explicit_selection_flattens_independently_optional_endpoint_slots(endpoint, source):
    base = voice_graph()
    for field in ('first_image', 'last_image'):
        if field != endpoint:
            base['10']['inputs'].pop(field)
    value = {**request(), 'reference_selection': {'pictures': [2, 1], 'audios': [1]},
             'prompt': '<Picture 2> is the person; <Picture 1> is the scene.'}
    graph = build_window_graph(base, value)
    settings = graph['10']['inputs']
    assert settings['first_image'] == ['31', 0] and settings['last_image'] == [source, 0]
    trace = graph['10']['_meta']['h3_long_video_reference_bindings']
    assert [r['slot'] for r in trace['original']['pictures'][:2]] == [endpoint, 'reference_image_1']


def test_selection_preserves_transcript_only_if_original_first_audio_stays_first():
    value = {**request(), 'reference_selection': {'pictures': [], 'audios': [1, 3]}}
    graph = build_window_graph(voice_graph(), value)
    assert graph['10']['inputs']['reference_transcript'] == 'Historical sample words.'
    trace = graph['10']['_meta']['h3_long_video_reference_bindings']
    assert trace['original']['audios'][0]['transcript'] == 'Historical sample words.'


@pytest.mark.parametrize('marker', ['<Picture 0>', '<Audio -1>', '<Picture 1.5>', '<Audio x>',
                                  'Use <Picture 1', 'Use <Audio 1', 'Use <Picture1>'])
def test_malformed_directing_reference_markers_fail_without_touching_literal_speech(marker):
    with pytest.raises(ValueError, match='marker|reference'):
        build_window_graph(voice_graph(), {**request(), 'prompt': marker})
    graph = build_window_graph(voice_graph(), {**request(), 'prompt': '<d>' + marker + '</d>'})
    assert graph['10']['inputs']['prompt'] == '<d>' + marker + '</d>'


def test_compiled_quiet_selection_cannot_claim_a_missing_original_audio_binding():
    value = {**request(), 'reference_selection': {'pictures': [1], 'audios': []},
        'voice_policy': 'dialogue_only', 'dialogue_ids': [],
        'audit': {'prompt_reference_space': 'window',
                  'reference_maps': {'pictures': {'1': 1}, 'audios': {}},
                  'original_reference_selection': {'pictures': [1], 'audios': [3]}}}
    with pytest.raises(ValueError, match='missing|binding'):
        build_window_graph(base_graph(), value)


def test_missing_graph_links_and_original_audio_gaps_are_rejected():
    base = voice_graph()
    base['10']['inputs'].pop('voice_reference_audio_2_file')
    with pytest.raises(ValueError, match='gap|contiguous'):
        build_window_graph(base, request())
    base = voice_graph()
    base['10']['inputs']['reference_image_1'] = ['missing-node', 0]
    with pytest.raises(ValueError, match='missing|binding'):
        build_window_graph(base, request())


def test_supplied_immutable_binding_snapshot_must_match_actual_base_bindings():
    from nodes.long_video_graph import roles, apply_window_reference_policy
    base, value = voice_graph(), request()
    value.update(reference_selection={'pictures': [9], 'audios': [3]},
                 prompt='<Picture 9> and <Audio 3> define this window.')
    active, director, *_ = roles(base)
    settings = active[director]['inputs']
    settings['prompt'] = value['prompt']
    trace = apply_window_reference_policy(settings, value, active)
    immutable = {**value, 'reference_bindings': json.loads(json.dumps(trace))}
    build_window_graph(base, immutable)
    changed = deepcopy(base)
    changed['10']['inputs']['voice_reference_audio_3_file'] = 'replacement.wav'
    with pytest.raises(ValueError, match='binding|snapshot'):
        build_window_graph(changed, immutable)


def test_direct_empty_audio_selection_safely_removes_reference_only_prose():
    value = {**request(), 'reference_selection': {'pictures': [1], 'audios': []},
        'prompt': '<Subject 1> (S1): same adult; <Audio 1> is voice timbre reference only.\n'
                  'Keep the same person. <d>Say <Audio 1> literally.</d>'}
    graph = build_window_graph(voice_graph(), value)
    settings = graph['10']['inputs']
    assert settings['voice_mode'] == 'none'
    assert 'reference only' not in settings['prompt']
    assert '<Subject 1> (S1): same adult' in settings['prompt']
    assert '<d>Say <Audio 1> literally.</d>' in settings['prompt']
    assert '\nKeep the same person.' in settings['prompt']


@pytest.mark.parametrize('marker', ['<Audio 0>', '<Audio -1>', '<Audio 1.5>', '<Audio x>',
                                  '<Audio1>', '<Audio 1'])
def test_direct_empty_audio_selection_rejects_malformed_marker_before_voice_omission(marker):
    value = {**request(), 'reference_selection': {'pictures': [], 'audios': []},
             'prompt': '<Subject 1> (S1): same adult; ' + marker + ' is voice timbre reference only.'}
    with pytest.raises(ValueError, match='reference|marker'):
        build_window_graph(voice_graph(), value)
    value['prompt'] = '<d>' + marker + ' is voice timbre reference only.</d>'
    assert build_window_graph(voice_graph(), value)['10']['inputs']['prompt'] == value['prompt']


@pytest.mark.parametrize('linked', [False, True])
def test_selection_rejects_storyboard_that_actual_director_would_filter(linked):
    import torch
    from nodes.director import _sanitize_h3_composite_refs
    base = base_graph()
    settings = base['10']['inputs']
    if not linked:
        settings.pop('reference_image_1')
    settings['reference_image_1_file'] = 'six_phase_storyboard.png'
    image = torch.zeros((1, 100, 100, 3), device='cpu')
    _, ordinary, _ = _sanitize_h3_composite_refs('Current scene.', None, [image], '',
                                               [settings['reference_image_1_file']])
    assert ordinary == []
    value = {**request(), 'prompt': 'Current scene.',
             'reference_selection': {'pictures': [3], 'audios': []}}
    with pytest.raises(ValueError, match='storyboard|preprocess|individual'):
        build_window_graph(base, value)


@pytest.mark.parametrize('prompt,source', [('Use the five-view identity convention.', 'scene.png'),
                                        ('Use the current scene.', 'five-view-identity.png')])
def test_selection_rejects_new_first_slot_crop_using_actual_compiled_prompt(prompt, source):
    import torch
    from nodes.director import _sanitize_h3_composite_refs
    base = base_graph()
    base['10']['inputs'].update(prompt='Old story without the current convention.', reference_image_1_file=source)
    image = torch.zeros((1, 100, 100, 3), device='cpu')
    _, ordinary, _ = _sanitize_h3_composite_refs(prompt, None, [image], '', [source])
    relocated, _, _ = _sanitize_h3_composite_refs(prompt, image, [], source, [])
    assert ordinary[0] is image and relocated.shape != image.shape
    value = {**request(), 'prompt': prompt, 'reference_selection': {'pictures': [3], 'audios': []}}
    with pytest.raises(ValueError, match='crop|preprocess|compatible|individual'):
        build_window_graph(base, value)


def test_compatible_first_slot_keeps_filename_crop_hint_even_when_link_wins():
    import torch
    from nodes.director import _sanitize_h3_composite_refs
    base = base_graph()
    base['10']['inputs']['first_image_file'] = 'five-view-identity.png'
    value = {**request(), 'prompt': 'Current identity and scene.',
             'reference_selection': {'pictures': [1, 3], 'audios': []}}
    graph = build_window_graph(base, value)
    settings = graph['10']['inputs']
    assert settings['first_image'] == ['11', 0]
    assert settings['first_image_file'] == 'five-view-identity.png'
    trace = graph['10']['_meta']['h3_long_video_reference_bindings']
    assert trace['original']['pictures'][0]['preprocessing_source'] == 'five-view-identity.png'
    image = torch.zeros((1, 100, 100, 3), device='cpu')
    cropped, _, _ = _sanitize_h3_composite_refs(settings['prompt'], image, [], settings['first_image_file'])
    assert tuple(cropped.shape) == (1, 88, 32, 3)
    # No selection retains the existing graph input cleanup behavior.
    assert 'first_image_file' not in build_window_graph(base, request())['10']['inputs']


def test_uncropped_ordinary_images_still_reorder_under_current_compiled_prompt():
    import torch
    from nodes.director import _sanitize_h3_composite_refs
    base = base_graph()
    base['10']['inputs'].update(prompt='The old base story mentions five-view.', reference_image_1_file='scene.png')
    value = {**request(), 'prompt': 'Current plain scene.',
             'reference_selection': {'pictures': [3, 4], 'audios': []}}
    graph = build_window_graph(base, value)
    settings = graph['10']['inputs']
    assert settings['first_image'] == ['31', 0] and settings['last_image'] == ['32', 0]
    assert settings['first_image_file'] == 'scene.png'
    image = torch.zeros((1, 100, 100, 3), device='cpu')
    first, _, notes = _sanitize_h3_composite_refs(settings['prompt'], image, [], settings['first_image_file'])
    assert first is image and notes == []


def test_static_reference_trace_cannot_substitute_for_production_sanitizer_validation():
    from nodes.long_video_graph import roles, apply_window_reference_policy
    from nodes.long_video_references import apply_reference_policy
    base = base_graph()
    value = {**request(), 'prompt': 'Current scene.', 'reference_selection': {'pictures': [3], 'audios': []}}
    active, director, *_ = roles(base)
    settings = active[director]['inputs']
    settings['prompt'] = value['prompt']
    static_trace = apply_reference_policy(settings, value, active)
    assert static_trace['preprocessing_validation'] == 'not_performed_static'
    with pytest.raises(ValueError, match='immutable|snapshot|binding'):
        build_window_graph(base, {**value, 'reference_bindings': static_trace})
    active, director, *_ = roles(base)
    settings = active[director]['inputs']
    settings['prompt'] = value['prompt']
    production_trace = apply_window_reference_policy(settings, value, active)
    assert production_trace['preprocessing_validation'] == 'director_policy_checked'
    graph = build_window_graph(base, {**value, 'reference_bindings': production_trace})
    assert graph['10']['_meta']['h3_long_video_reference_bindings'] == production_trace


@pytest.mark.parametrize('serialized', [False, True])
def test_asset_fingerprint_tracks_active_lora_stack_weights(tmp_path, serialized):
    from types import SimpleNamespace
    weight = tmp_path / 'detail.safetensors'
    weight.write_bytes(b'original weights')
    rows = [{'lora': weight.name, 'on': True}, {'lora': 'disabled.safetensors', 'on': False}]
    graph = {'lora': {'class_type': 'DaSiWa_LTX2LoraLoader', 'inputs': {
        'stack_data': json.dumps(rows) if serialized else rows}}}
    paths = SimpleNamespace(get_full_path=lambda category, name: str(tmp_path / name))
    original = freeze_assets(graph, paths)
    assert list(original) == ['lora:stack_data:0']
    weight.write_bytes(b'replaced weights under same filename')
    changed = freeze_assets(graph, paths)
    assert original['lora:stack_data:0']['sha256'] != changed['lora:stack_data:0']['sha256']


def test_replaced_weights_change_actual_comfy_loader_and_director_cache_signatures(monkeypatch):
    import asyncio
    import nodes
    comfy_root = Path('D:/ComfyUI_windows_portable-G313/ComfyUI')
    if not comfy_root.is_dir():
        pytest.skip('Installed ComfyUI cache implementation is not available')
    monkeypatch.syspath_prepend(str(comfy_root))
    from comfy_execution.caching import CacheKeySetInputSignature, CacheKeySetID
    import comfy_execution.caching as caching
    from comfy_execution.graph import DynamicPrompt
    monkeypatch.setattr(caching, 'NODE_CLASS_CONTAINS_UNIQUE_ID', {})

    first = request()
    first['asset_revision'] = 'a' * 64
    second = {**first, 'asset_revision': 'b' * 64}
    base = base_graph()
    base['detail'] = {'class_type': 'LoraLoaderModelOnly', 'inputs': {
        'model': base['15']['inputs']['model'],
        'lora_name': 'MysticXXX_MMH3-V5-ref2va.safetensors', 'strength_model': 0.8}}
    base['15']['inputs']['model'] = ['detail', 0]
    first_graph = build_window_graph(base, first)
    second_graph = build_window_graph(base, second)
    monkeypatch.setattr(nodes, 'NODE_CLASS_MAPPINGS', {
        node['class_type']: type('CacheFixture', (), {'INPUT_TYPES': classmethod(lambda cls: {})})
        for node in first_graph.values()}, raising=False)

    class Changed:
        async def get(self, node_id):
            return False

    async def signatures(graph):
        cache = CacheKeySetInputSignature(DynamicPrompt(graph), [], Changed())
        return {node.get('_meta', {}).get('h3_long_video_source_node_id', node_id):
                    await cache.get_node_signature(DynamicPrompt(graph), node_id)
                for node_id, node in graph.items() if node['class_type'] in {
                    'UNETLoader', 'CLIPLoader', 'VAELoader', 'LoraLoaderModelOnly', 'MiniMaxH3DirectorPlus'}}

    async def instances(graph):
        cache = CacheKeySetID(DynamicPrompt(graph), [], Changed())
        await cache.add_keys(list(graph))
        return {node['_meta']['h3_long_video_source_node_id']: cache.keys[node_id]
                for node_id, node in graph.items() if 'h3_long_video_source_node_id' in node.get('_meta', {})}

    before, after = asyncio.run(signatures(first_graph)), asyncio.run(signatures(second_graph))
    assert len(before) >= 4
    assert all(before[node_id] != after[node_id] for node_id in before)
    next_window = {**first, 'prompt': first['prompt'] + '\nContinue the next action.', 'seed': 43}
    next_graph = build_window_graph(base, next_window)
    next_signatures = asyncio.run(signatures(next_graph))
    assert before['10'] != next_signatures['10']
    # The optional LoRA follows the guide-dependent model router; raw file
    # loaders remain shared even when a later window updates that guide.
    assert all(before[node_id] == next_signatures[node_id] for node_id in before if node_id not in {'10', 'detail'})
    first_instances = asyncio.run(instances(first_graph))
    assert all(key != asyncio.run(instances(second_graph))[source_id] for source_id, key in first_instances.items())
    assert first_instances == asyncio.run(instances(next_graph))


def test_revision_keeps_plain_mystic_loader_selection_visible_to_actual_router(monkeypatch):
    from nodes.additional_loras import upstream_loras
    graph = base_graph()
    graph['detail'] = {'class_type': 'LoraLoaderModelOnly', 'inputs': {
        'model': graph['15']['inputs']['model'],
        'lora_name': 'MysticXXX_MMH3-V5-ref2va.safetensors', 'strength_model': 0.8}}
    graph['15']['inputs']['model'] = ['detail', 0]
    original = upstream_loras(graph, '15')
    assert len(original) == 1
    compiled = build_window_graph(graph, {**request(), 'asset_revision': 'a' * 64})
    assert upstream_loras(compiled, '15') == original
    from nodes import performance
    from types import SimpleNamespace
    base = object()
    loaded = SimpleNamespace(parent=SimpleNamespace(clone=lambda: base))
    monkeypatch.setattr(performance, 'resolve_registered_model_name', lambda category, name: name)
    monkeypatch.setattr(performance, '_apply_acceleration', lambda model, guide: (model, 'ok', True, model))
    guide = {'sampling_method': 'selflift', 'resolved_backend': 'ref2va_model'}
    result = performance.MiniMaxH3AccelerationRouter().apply(loaded, guide, prompt=compiled, unique_id='15')
    assert result[0] is base and result[3] is base
    assert guide['natural_texture_policy'] == 'selflift_u22_baseline'
