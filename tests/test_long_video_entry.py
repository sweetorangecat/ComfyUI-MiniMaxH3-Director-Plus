from unittest.mock import patch
import pytest

from nodes.long_video_entry import MiniMaxH3U11Output, MiniMaxH3LongVideoConfig
from nodes.stream_output import MiniMaxH3StreamingVideoCombine


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
