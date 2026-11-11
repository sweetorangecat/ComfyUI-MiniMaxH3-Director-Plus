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
    config = next(n for n in workflow['nodes'] if n['type'] == 'MiniMaxH3LongVideoConfig')
    assert config['widgets_values'][:2] == [False, 30.0]
    validate_workflow(workflow, require_sections=True)
    snapshot = deepcopy(workflow)
    upgrade_long_video_entry(workflow)
    assert workflow == snapshot


def test_api_template_long_entry_is_disabled_and_short_compatible():
    template = build_api_template()
    assert template['24']['class_type'] == 'MiniMaxH3U11Output'
    config_id = template['24']['inputs']['long_video'][0]
    assert template[config_id]['class_type'] == 'MiniMaxH3LongVideoConfig'
    assert template[config_id]['inputs']['enabled'] is False
