from unittest.mock import patch
import pytest

from nodes.long_video_entry import MiniMaxH3U11Output, MiniMaxH3LongVideoConfig
from nodes.stream_output import MiniMaxH3StreamingVideoCombine
from copy import deepcopy
from nodes.director import MiniMaxH3DirectorPlus


def test_short_entry_keeps_all_old_inputs_and_waits_for_connected_lazy_audio():
    original = MiniMaxH3StreamingVideoCombine.INPUT_TYPES()
    updated = MiniMaxH3U11Output.INPUT_TYPES()
    assert set(updated['required']) == set(original['required'])
    assert updated['required']['images'][1]['lazy'] is True
    node = MiniMaxH3U11Output()
    assert set(node.check_lazy_status(images=None, guide=None, audio=None)) == {'images','guide','audio'}
    assert node.check_lazy_status(images=object(), guide={}, audio={}) == []
    assert node.check_lazy_status(long_video={'enabled': True}) == []


def test_disabled_long_config_delegates_short_output_without_modifying_it():
    node = MiniMaxH3U11Output()
    guide, images = {}, object()
    with patch.object(MiniMaxH3StreamingVideoCombine, 'combine', return_value='short result') as combine:
        assert node.combine(images=images, guide=guide, long_video={'enabled': False}, frame_rate=24) == 'short result'
        assert combine.call_args.kwargs == {'images': images, 'guide': guide, 'frame_rate': 24}


def test_default_config_keeps_long_video_off():
    fields = MiniMaxH3LongVideoConfig.INPUT_TYPES()['required']
    assert fields['enabled'][1]['default'] is False
    assert fields['target_seconds'][1]['default'] == 30


def test_long_launch_returns_without_evaluating_short_sampler_inputs():
    node = MiniMaxH3U11Output()
    graph = {'output': {'class_type': 'MiniMaxH3U11Output','inputs': {}}}
    with patch('nodes.long_video_entry.launch', return_value={'project_name':'test','status':'new','output':''}) as launch:
        result = node.combine(long_video={'enabled': True,'project_name':'test'},
                              prompt=graph, unique_id='output', frame_rate=24)
    assert launch.call_args.args == (graph, {'enabled': True,'project_name':'test'})
    assert result['ui']['h3_long_video'][0]['project_name'] == 'test'


def test_enabled_empty_contract_fails_before_launch_with_import_help():
    with pytest.raises(ValueError, match='U11'):
        MiniMaxH3LongVideoConfig().configure(True,90,'test','{}')


def test_disabled_config_does_not_validate_or_reset_saved_short_draft():
    config = MiniMaxH3LongVideoConfig().configure(False,float('nan'),'saved','unfinished JSON')[0]
    assert config['scene_contract'] == 'unfinished JSON'
    assert config['project_name'] == 'saved'


def director_graph(seconds=30, **settings):
    return {
        'director': {'class_type': 'MiniMaxH3DirectorPlus', 'inputs': {
            'duration': seconds, 'prompt': '人物自然走动。', 'mode': 'REF2VA',
            'voice_mode': 'h3_reference', 'seed': 314159,
            'reference_image_1_file': 'identity.png',
            'voice_reference_audio_file': 'voice.wav',
            'resolution_preset': '2K QHD', 'performance_preset': '稳定质量',
            'sampling_method': 'standard', 'postprocess_mode': 'ai_upscale',
            **settings}},
        'output': {'class_type': 'MiniMaxH3U11Output', 'inputs': {
            'guide': ['director', 0], 'frame_rate': 24,
            'codec': 'H.264', 'container': 'MP4', 'save_output': True}},
    }


@pytest.mark.parametrize('changes', [dict(mode='I2VA'), dict(performance_preset='稳定质量'),
    dict(motion_smoothing='rife_x2')])
def test_automatic_long_author_choice_rejects_incompatible_original_fields(changes):
    from nodes.long_video_entry import _automatic_recipe
    graph = director_graph(30, voice_mode='none', performance_preset='质量优先二采样',
        community_lora_profile='combat', **{key: value for key, value in changes.items() if key != 'performance_preset'})
    if 'performance_preset' in changes:
        graph['director']['inputs']['performance_preset'] = changes['performance_preset']
    with patch('nodes.long_video_entry.preflight', return_value={}), \
         patch('nodes.long_video_entry.contract_from_prompt', return_value={}):
        with pytest.raises(ValueError, match='REF2VA|质量优先|RIFE|24FPS'):
            _automatic_recipe((graph, 'director', 30))


def test_original_director_duration_accepts_long_targets_without_changing_physical_requests():
    spec = MiniMaxH3DirectorPlus.INPUT_TYPES()['required']['duration'][1]
    assert spec.get('max', float('inf')) > 120
    from nodes.schema import normalize_request, RequestError
    with pytest.raises(RequestError):
        normalize_request({'duration': 30})


@pytest.mark.parametrize('seconds', [16, 30, 90, 127])
def test_long_duration_on_original_director_skips_short_model_and_audio_dependencies(seconds):
    assert MiniMaxH3U11Output().check_lazy_status(
        prompt=director_graph(seconds), unique_id='output',
        images=None, guide=None, audio=None) == []


def test_fifteen_seconds_keeps_short_route_with_the_original_prompt_graph():
    node = MiniMaxH3U11Output()
    graph = director_graph(15)
    assert set(node.check_lazy_status(prompt=graph, unique_id='output',
                                     images=None, guide=None, audio=None)) == {'images','guide','audio'}
    with patch.object(MiniMaxH3StreamingVideoCombine, 'combine', return_value='short') as old:
        assert node.combine(images='frames', guide={}, unique_id='output', prompt=graph) == 'short'
    assert old.call_args.kwargs['prompt'] == graph


def test_automatic_long_route_uses_original_recipe_and_preflights_without_a_config_node():
    graph = director_graph(90)
    before = deepcopy(graph)
    contract = {'auto': 'compiled'}
    with patch('nodes.long_video_entry.contract_from_prompt', create=True, return_value=contract) as compile_prompt, \
         patch('nodes.long_video_entry.preflight', create=True, return_value={}) as check, \
         patch('nodes.long_video_entry.launch', return_value={'project_name':'auto','status':'new'}) as start:
        result = MiniMaxH3U11Output().combine(unique_id='output', prompt=graph, frame_rate=24)
    assert graph == before
    compile_prompt.assert_called_once()
    active, config = start.call_args.args
    assert config['target_seconds'] == 90 and config['scene_contract'] == contract
    assert config['entry_mode'] == 'director_duration'
    assert config['project_name'].startswith('u11-')
    check.assert_called_once_with(active, config)
    inputs = active['director']['inputs']
    assert inputs['seed'] == 314159
    assert inputs['reference_image_1_file'] == 'identity.png'
    assert inputs['voice_reference_audio_file'] == 'voice.wav'
    assert inputs['voice_mode'] == 'h3_reference' and inputs['resolution_preset'] == '2K QHD'
    assert inputs['sampling_method'] == 'selflift'
    assert inputs['vosr2_profile'] == 'manual_baseline'
    assert result['ui']['h3_long_video'][0]['project_name'] == 'auto'


def test_hidden_disabled_legacy_config_cannot_prevent_automatic_duration_routing():
    graph = director_graph(30)
    assert MiniMaxH3U11Output().check_lazy_status(
        prompt=graph, unique_id='output', long_video={'enabled':False}, images=None, guide=None) == []


def test_long_fish_voice_errors_before_models_instead_of_silently_switching_voice():
    graph = director_graph(30, voice_mode='fish_lock')
    with patch('nodes.long_video_entry.launch') as start, pytest.raises(ValueError, match='音色'):
        MiniMaxH3U11Output().combine(unique_id='output', prompt=graph)
    start.assert_not_called()


def test_long_duration_resolves_only_connected_director_and_internal_short_windows_stay_short():
    graph = director_graph(13)
    graph['unused'] = {'class_type':'MiniMaxH3DirectorPlus','inputs':{'duration':90}}
    assert set(MiniMaxH3U11Output().check_lazy_status(
        prompt=graph, unique_id='output', images=None, guide=None)) == {'images','guide'}


def test_old_output_gets_an_upgrade_hint_before_any_resolution_or_media_work():
    from nodes.schema import RequestError
    with patch('nodes.director.calculate_resolution', side_effect=AssertionError('expensive path')), \
         pytest.raises(RequestError, match='U11.*输出'):
        MiniMaxH3DirectorPlus().build(mode='REF2VA', prompt='Walk.', duration=30,
            width=1344, height=768, voice_mode='none', ref_image_size='match',
            performance_preset='稳定质量', timeline_data='{}',
            target_dialogue='', reference_transcript='')


def test_legacy_linked_short_duration_still_evaluates_the_original_dependencies():
    graph=director_graph(5)
    graph['number']={'class_type':'ExistingCustomInteger','inputs':{}}
    graph['director']['inputs']['duration']=['number',0]
    assert set(MiniMaxH3U11Output().check_lazy_status(prompt=graph,unique_id='output',images=None,guide=None)) == {'images','guide'}


@pytest.mark.parametrize('mode',['L2VA','FL2VA'])
def test_last_only_long_reference_is_compacted_without_changing_its_picture_identity(mode):
    graph=director_graph(30,mode=mode,last_image_file='ending.png')
    original=deepcopy(graph)
    with patch('nodes.long_video_entry.contract_from_prompt',return_value={'compiled':True}), \
         patch('nodes.long_video_entry.preflight'), \
         patch('nodes.long_video_entry.launch',return_value={'project_name':'test'}) as start:
        MiniMaxH3U11Output().combine(unique_id='output',prompt=graph)
    active=start.call_args.args[0]['director']['inputs']
    assert graph==original
    assert active['first_image_file']=='ending.png'
    assert 'last_image_file' not in active
    assert 'reference_image_1_file' not in active


@pytest.mark.parametrize('explicit',[False,True])
def test_comfy_runtime_cache_nan_is_removed_from_recipe_without_mutating_original_prompt(explicit):
    import json,math
    graph=director_graph(30)
    graph['output']['is_changed']=[MiniMaxH3U11Output.IS_CHANGED()]
    graph['director']['is_changed']=[float('nan')]
    graph['director']['_meta']={'h3_preserved_role':'actual media'}
    legacy={'enabled':True,'project_name':'legacy','target_seconds':30,'scene_contract':{'compiled':True}} if explicit else None
    with patch('nodes.long_video_entry.contract_from_prompt',return_value={'compiled':True}), \
         patch('nodes.long_video_entry.preflight'), \
         patch('nodes.long_video_entry.launch',return_value={'project_name':'test'}) as start:
        MiniMaxH3U11Output().combine(unique_id='output',prompt=graph,long_video=legacy)
    active=start.call_args.args[0]
    assert all('is_changed' not in node for node in active.values())
    assert active['director']['_meta']['h3_preserved_role']=='actual media'
    json.dumps(active,allow_nan=False)
    assert math.isnan(graph['output']['is_changed'][0]) and math.isnan(graph['director']['is_changed'][0])
