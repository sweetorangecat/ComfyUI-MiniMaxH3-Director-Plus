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
