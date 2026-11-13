from copy import deepcopy
import json
from pathlib import Path

from tools.build_u11_workflow import upgrade_long_video_entry, build_api_template
from tools.validate_workflow import validate_workflow


def test_original_u11_upgrade_keeps_widgets_models_assets_and_links():
    path = next((Path(__file__).parents[1] / 'examples').glob('U11-*.json'))
    workflow = json.loads(path.read_text(encoding='utf-8'))
    original = deepcopy(workflow)
    upgrade_long_video_entry(workflow)
    output = next(n for n in workflow['nodes'] if n['type'] == 'MiniMaxH3U11Output')
    old = next(n for n in original['nodes'] if n['id'] == output['id'])
    assert output['widgets_values'] == old['widgets_values']
    untouched = {n['id']: n for n in original['nodes'] if n['id'] != output['id']}
    assert all(n == untouched[n['id']] for n in workflow['nodes'] if n['id'] in untouched)
    assert not any(n['type'] == 'MiniMaxH3LongVideoConfig' for n in workflow['nodes'])
    assert not any(g.get('id') == 'u11-long-video' for g in workflow.get('groups', []))
    validate_workflow(workflow, require_sections=True)
    snapshot = deepcopy(workflow)
    upgrade_long_video_entry(workflow)
    assert workflow == snapshot


def test_api_template_uses_the_original_director_duration_without_an_extra_config():
    template = build_api_template()
    assert template['24']['class_type'] == 'MiniMaxH3U11Output'
    assert 'long_video' not in template['24']['inputs']
    assert not any(node['class_type'] == 'MiniMaxH3LongVideoConfig' for node in template.values())


def test_original_upgrade_publishes_user_entry_contract_without_resetting_saved_settings():
    path = next((Path(__file__).parents[1] / 'examples').glob('U11-*.json'))
    workflow = json.loads(path.read_text(encoding='utf-8'))
    before = deepcopy(workflow['nodes'])
    upgrade_long_video_entry(workflow)
    assert workflow['nodes'] == before
    info = workflow['extra']['u11_long_video_user_entry']
    assert info['duration_source'] == 'MiniMaxH3DirectorPlus.duration'
    assert info['automatic_above_seconds'] == 15
    assert info['preflight'] == 'automatic_before_generation'
    template = build_api_template()
    assert not any(node['class_type'] == 'MiniMaxH3LongVideoConfig' for node in template.values())


def test_old_saved_default_config_is_retired_without_touching_original_director_choices():
    path = next((Path(__file__).parents[1] / 'examples').glob('U11-*.json'))
    workflow = json.loads(path.read_text(encoding='utf-8'))
    # The pre-simplification shipped canvas has a marked, default-off config.
    output = next(n for n in workflow['nodes'] if n['type'] == 'MiniMaxH3U11Output')
    config = next((n for n in workflow['nodes'] if n['type'] == 'MiniMaxH3LongVideoConfig'), None)
    if config is None:
        config = {'id':999999,'type':'MiniMaxH3LongVideoConfig','widgets_values':[False,30,'h3-native-av','{}','target'],
                  'properties':{'cnr_id':'ComfyUI-MiniMaxH3-Director-Plus'}}
        workflow['nodes'].append(config)
    original = deepcopy(next(n for n in workflow['nodes'] if n['type'] == 'MiniMaxH3DirectorPlus'))
    upgrade_long_video_entry(workflow)
    assert not any(n['type']=='MiniMaxH3LongVideoConfig' for n in workflow['nodes'])
    assert next(n for n in workflow['nodes'] if n['id']==original['id']) == original


def test_explicit_saved_expert_contract_is_preserved_as_legacy_compatibility():
    path = next((Path(__file__).parents[1] / 'examples').glob('U11-*.json'))
    workflow = json.loads(path.read_text(encoding='utf-8'))
    config = {'id':999999,'type':'MiniMaxH3LongVideoConfig','widgets_values':[True,90,'my-film','{"custom":true}','target'],
              'properties':{'cnr_id':'ComfyUI-MiniMaxH3-Director-Plus'}}
    workflow['nodes'].append(config)
    original = deepcopy(config)
    upgrade_long_video_entry(workflow)
    assert next(n for n in workflow['nodes'] if n['id']==config['id']) == original


def test_default_valued_legacy_config_with_connected_enable_is_not_deleted():
    workflow = {'nodes':[
        {'id':1,'type':'PrimitiveBoolean','inputs':[], 'outputs':[{'links':[1]}]},
        {'id':2,'type':'MiniMaxH3LongVideoConfig','widgets_values':[False,30,'h3-native-av','{}','target'],
         'properties':{'cnr_id':'ComfyUI-MiniMaxH3-Director-Plus'},
         'inputs':[{'name':'enabled','link':1}], 'outputs':[{'links':[2]}]},
        {'id':3,'type':'MiniMaxH3U11Output','properties':{'Node name for S&R':'MiniMaxH3U11Output'},
         'inputs':[{'name':'long_video','link':2}], 'outputs':[]}],
        'links':[[1,1,0,2,0,'BOOLEAN'],[2,2,0,3,0,'MINIMAX_H3_LONG_VIDEO']]}
    original = deepcopy(workflow)
    upgrade_long_video_entry(workflow)
    assert workflow['nodes'] == original['nodes']
    assert workflow['links'] == original['links']
