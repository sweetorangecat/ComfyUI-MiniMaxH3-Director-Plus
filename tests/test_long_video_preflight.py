"""Read-only, actual-graph checks, independent of project journals/queues."""
from copy import deepcopy
import importlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


def fixture(seconds=30):
    graph = json.loads((ROOT / 'templates/u11_api.json').read_text(encoding='utf-8'))
    graph['10']['inputs'].update(mode='REF2VA', sampling_method='selflift',
        performance_preset='质量优先二采样', postprocess_mode='vosr2', vosr2_profile='manual_baseline')
    # Use real graph bindings, with optional endpoints and unused original slots unbound.
    for name in ('first_image', 'last_image', *[f'reference_image_{i}' for i in range(3, 10)]):
        graph['10']['inputs'].pop(name, None)
    total = round(seconds * 24)
    contract = {'schema_version': 1, 'fps': 24, 'total_frames': total,
        'not_ready_for_video': True, 'reference_assets': [
            {'slot': 'Picture 1', 'responsibility': 'Same face', 'approved': False, 'mounted': False}],
        'subject_definitions': '<Subject 1> adult using <Picture 1> face and <Picture 2> scale.',
        'summary': 'Walk.', 'retention_analysis': 'Same adult and cup.',
        'scale_contract': 'Adult 170 cm; cup 10 cm.', 'overall_soundscape': 'Room tone.',
        'non_diegetic_music': 'None.', 'beats': [
            {'id': 'walk', 'start_frame': 0, 'end_frame': total, 'action': 'Walk carrying the same cup.'}],
        'dialogues': []}
    return graph, {'enabled': True, 'target_seconds': seconds, 'scene_contract': contract,
                   'project_name': 'test', 'stop_condition': 'target'}


def preflight(*args, **kwargs):
    assert importlib.util.find_spec('nodes.long_video_preflight'), 'read-only preflight is missing'
    return importlib.import_module('nodes.long_video_preflight').preflight(*args, check_encoder=False, **kwargs)


def test_preflight_uses_every_actual_window_without_project_or_queue_side_effects(tmp_path, monkeypatch):
    graph, config = fixture(97.125)
    saved_graph, saved_config = deepcopy(graph), deepcopy(config)
    from nodes.long_video_store import ProjectStore
    monkeypatch.setattr(ProjectStore, '__init__', lambda *_: pytest.fail('preflight created a project'))
    result = preflight(graph, config)
    assert result['total_frames'] == 2331
    assert result['total_seconds'] == 97.125
    assert sum(w['net_new_frames'] for w in result['windows']) == 2331
    assert len(result['windows']) > 8
    assert result['incoming_state'] == 'runtime_observation_required'
    assert result['static_future_checks'] == 'synthetic_state_only_never_persisted_or_approved'
    assert all(w['reference_bindings']['preprocessing_validation'] == 'not_applicable_no_selection'
               for w in result['windows'])
    assert len(result['binding_inventory']['pictures']) == 2
    assert result['source_facts']['not_ready_for_video'] is True
    assert result['source_facts']['reference_assets'][0]['approved'] is False
    assert result['source_facts']['reference_assets'][0]['mounted'] is False
    assert result['warnings']
    assert graph == saved_graph and config == saved_config


@pytest.mark.parametrize('raw,match', [('{}', 'U11'), ({'source_beats': []}, '导出'),
                                     ([], 'object'), (None, 'object')])
def test_preflight_empty_or_rich_planning_contract_has_actionable_error(raw, match):
    graph, config = fixture()
    config['scene_contract'] = raw
    with pytest.raises(ValueError, match=match):
        preflight(graph, config)


def test_preflight_duration_mismatch_does_not_rescale_story():
    graph, config = fixture()
    config['target_seconds'] = 60
    with pytest.raises(ValueError, match='30.*60|60.*30'):
        preflight(graph, config)
    assert config['scene_contract']['total_frames'] == 720


def test_preflight_checks_future_overrides_and_real_reference_handshake():
    graph, config = fixture()
    from nodes.long_video_plan import plan_windows
    windows = plan_windows(720)
    w = windows[1]
    config['scene_contract']['window_overrides'] = [{
        'new_start': w['new_start'], 'new_end': w['new_end'],
        'sections': {'subject_definitions': '<Subject 1> adult using <Picture 3>.'},
        'reference_selection': {'pictures': [3], 'audios': []}}]
    with pytest.raises(ValueError, match='missing|缺少'):
        preflight(graph, config)


def test_preflight_forward_geometry_checked_without_claiming_actual_state():
    graph, config = fixture()
    from nodes.long_video_plan import plan_windows
    windows = plan_windows(720)
    w = windows[1]
    config['scene_contract']['window_overrides'] = [{
        'new_start': w['new_start'], 'new_end': w['new_end'], 'forward_frames': 17}]
    result = preflight(graph, config)
    assert result['windows'][1]['forward_review_required'] is True
    assert result['windows'][1]['forward_geometry']['source_local_range'] == [294, 311]
    config['scene_contract']['window_overrides'][0]['forward_frames'] = 102
    with pytest.raises(ValueError, match='tail|余量'):
        preflight(graph, config)


def test_contract_schema_and_real_export_keep_pending_metadata():
    path = ROOT / 'templates/u11_long_video_contract.schema.json'
    assert path.is_file(), 'canonical machine-readable schema is missing'
    schema = json.loads(path.read_text(encoding='utf-8'))
    assert schema['properties']['fps']['const'] == 24
    assert schema['properties']['window_overrides']['items']['properties']['forward_frames']['multipleOf'] == 17
    example = ROOT / 'examples/u11_long_video_scene_contract.en.json'
    assert example.is_file(), 'actual canonical H3 U11 export example is missing'
    contract = json.loads(example.read_text(encoding='utf-8'))
    from nodes.long_video_plan import validate_contract, validate_window_overrides, plan_windows, protected_intervals
    validated = validate_contract(contract, 2160)
    windows = plan_windows(2160, protected_intervals=protected_intervals(validated))
    validate_window_overrides(validated, windows)
    assert len(windows) == 9
    assert validated['not_ready_for_video'] is True
    assert all(a['approved'] is False and a['mounted'] is False for a in validated['reference_assets'])


@pytest.mark.parametrize('language',['en','zh'])
def test_real_skill_export_checks_nine_windows_with_actual_eight_plus_voice_bank(language):
    graph, config = fixture(90)
    settings = graph['10']['inputs']
    for i in range(1,9):
        settings[f'reference_image_{i}'] = [str(30+i),0]
    settings.update(voice_mode='h3_reference',voice_reference_audio=['13',0])
    contract = json.loads((ROOT / f'examples/u11_long_video_scene_contract.{language}.json').read_text(encoding='utf-8'))
    config['scene_contract'] = contract
    result = preflight(graph,config)
    assert len(result['windows']) == 9
    assert len(result['binding_inventory']['pictures']) == 8
    assert len(result['binding_inventory']['audios']) == 1
    assert all(w['reference_bindings']['preprocessing_validation'] == 'director_policy_checked' for w in result['windows'])
    assert [(w['start_seconds']*24,w['end_seconds']*24,w['forward_geometry']['forward_frames'])
            for w in result['windows'] if w['forward_review_required']] == [(549,753,17),(1518,1773,34),(1977,2160,34)]
    assert result['source_facts']['not_ready_for_video'] is True
    assert all(not a['approved'] and not a['mounted'] for a in result['source_facts']['reference_assets'])
    assert settings.get('first_image') is None and settings.get('last_image') is None


def test_schema_accepts_real_export_and_rejects_audio_slot_four_without_capping_speakers():
    jsonschema = pytest.importorskip('jsonschema')
    schema = json.loads((ROOT / 'templates/u11_long_video_contract.schema.json').read_text(encoding='utf-8'))
    jsonschema.Draft202012Validator.check_schema(schema)
    for language in ('en','zh'):
        exported = json.loads((ROOT / f'examples/u11_long_video_scene_contract.{language}.json').read_text(encoding='utf-8'))
        jsonschema.validate(exported,schema)
    _, config = fixture()
    value = config['scene_contract']
    value['dialogues'] = [{'id':'many','start_frame':0,'end_frame':24,
        'speaker':'<Subject 7> (S8)','text':'Hello.','audio_slot':3}]
    jsonschema.validate(value,schema)
    value['dialogues'][0]['audio_slot'] = 4
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(value,schema)


@pytest.mark.parametrize('meta',[None,[],{'h3_long_video_reference_source_files':None},
    {'h3_long_video_reference_source_files':[]},{'h3_long_video_reference_source_files':{'first_image':False}}])
def test_malformed_graph_metadata_has_clear_value_error_not_attribute_error(meta):
    graph,config = fixture()
    graph['10']['_meta'] = meta
    with pytest.raises(ValueError,match='meta|metadata|source'):
        preflight(graph,config)


@pytest.mark.parametrize('name,value',[('enabled','yes'),('project_name',[]),('project_name','CON'),
    ('project_name','../outside'),('project_name','trailing.'),('stop_condition',[])])
def test_preflight_rejects_wrong_config_types_without_creating_project(name,value,monkeypatch):
    graph,config = fixture()
    config[name] = value
    from nodes.long_video_store import ProjectStore
    monkeypatch.setattr(ProjectStore,'__init__',lambda *_: pytest.fail('preflight created a project'))
    with pytest.raises(ValueError,match=name):
        preflight(graph,config)
