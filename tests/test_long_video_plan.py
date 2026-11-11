"""Frame-domain regressions for native H3 continuation, without ComfyUI."""
import copy
import importlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def module():
    assert (ROOT / 'nodes/long_video_plan.py').is_file(), 'Native long-video planner is missing'
    return importlib.import_module('nodes.long_video_plan')


def contract(total=720):
    return {
        'schema_version': 1, 'fps': 24, 'total_frames': total,
        'subject_definitions': '<Subject 1> is the adult from the mounted identity reference.',
        'summary': 'An uninterrupted conversation while walking through a courtyard.',
        'retention_analysis': 'Preserve the same identities, costume, courtyard and perspective.',
        'scale_contract': 'Adult height 1.70 m; cup height 0.12 m; one cup, held in the right hand.',
        'overall_soundscape': 'Continuous courtyard ambience. Only scheduled dialogue.',
        'non_diegetic_music': 'None.',
        'beats': [{'id': 'walk', 'start_frame': 0, 'end_frame': total,
                   'action': 'Walk forward at normal speed while keeping the cup upright.',
                   'camera': 'Maintain the existing eye-level medium tracking composition.',
                   'sound': 'Footsteps stay in time with the steps.', 'indivisible': False}],
        'dialogues': [],
    }


def test_native_30_second_geometry_and_handoffs():
    p = module()
    windows = p.plan_windows(720)
    assert [w['new_end'] - w['new_start'] for w in windows] == [294, 255, 171]
    assert [w['sample_frames'] for w in windows] == [328, 328, 277]
    assert [w['context_frames'] for w in windows] == [0, 39, 39]
    assert [w['source_endpoint'] for w in windows] == [0, 294, 294]
    assert [w['endpoint'] for w in windows] == [294, 294, 243]
    assert p.video_steps(39) == 12
    assert p.audio_ticks(39) == 65
    assert p.audio_ticks(328) == 547
    assert p.audio_ticks(277) == 462


@pytest.mark.parametrize('total', [1, 39, 90, 294, 295, 720, 1440, 2880, 100001])
def test_arbitrary_finite_lengths_cover_exactly_with_positive_progress(total):
    p = module()
    windows = p.plan_windows(total)
    cursor = 0
    for i, w in enumerate(windows):
        assert w['index'] == i
        assert w['new_start'] == cursor < w['new_end']
        assert w['local_new_start'] == w['context_frames']
        assert w['local_new_end'] == w['context_frames'] + w['new_end'] - cursor
        assert 4 * p.FPS <= w['sample_frames'] <= 15 * p.FPS
        assert (w['sample_frames'] - 5) % 17 == 0
        assert w['local_new_end'] <= w['endpoint'] <= w['sample_frames'] - 34
        if i < len(windows) - 1:
            assert w['local_new_end'] == w['endpoint']
            assert (w['endpoint'] - 39) % 51 == 0
        if i:
            assert w['source_endpoint'] == windows[i - 1]['local_new_end']
        cursor = w['new_end']
    assert cursor == total


@pytest.mark.parametrize('seconds, expected', [(30, 720), (60, 1440), (120, 2880), (1 / 48, 1), (30.03, 721)])
def test_target_duration_rounds_to_nearest_frame(seconds, expected):
    assert module().target_frame_count(seconds) == expected


@pytest.mark.parametrize('seconds', [True, 0, -1, float('nan'), float('inf'), '30', 0.001])
def test_invalid_duration_rejected(seconds):
    with pytest.raises(ValueError):
        module().target_frame_count(seconds)


def test_protection_moves_boundary_only_to_shared_av_endpoint():
    p = module()
    windows = p.plan_windows(720, protected_intervals=[(280, 310)])
    assert windows[0]['new_end'] == 243
    assert all(not 280 < w['new_end'] < 310 for w in windows)
    assert windows[1]['source_endpoint'] == 243
    assert p.plan_windows(720, protected_intervals=[(294, 310)])[0]['new_end'] == 294
    with pytest.raises(ValueError, match='protected|indivisible|dialogue'):
        p.plan_windows(720, protected_intervals=[(1, 400)])


@pytest.mark.parametrize('kwargs', [
    {'total_frames': 0}, {'total_frames': True}, {'total_frames': 720.0},
    {'total_frames': 720, 'context_frames': 40},
    {'total_frames': 720, 'max_sample_frames': 95},
    {'total_frames': 720, 'max_sample_frames': 362},
    {'total_frames': 720, 'safety_tail_frames': -1},
    {'total_frames': 720, 'protected_intervals': [(0, 721)]},
])
def test_invalid_geometry_is_explicit(kwargs):
    with pytest.raises(ValueError):
        module().plan_windows(**kwargs)


def test_contract_validation_is_detached_and_collects_complete_protection():
    p = module()
    s = contract()
    s['beats'][0]['indivisible'] = True
    s['dialogues'] = [{'id': 'line', 'start_frame': 280, 'end_frame': 310,
                       'speaker': '<Subject 1> (S1)', 'text': '我们继续走吧。', 'language': 'Chinese'}]
    canonical = p.validate_contract(json.dumps(s), 720)
    assert set(p.protected_intervals(canonical)) == {(0, 720), (280, 310)}
    canonical['beats'][0]['action'] = 'Changed'
    assert s['beats'][0]['action'] != 'Changed'


@pytest.mark.parametrize('mutation', [
    lambda s: s.update(fps=25),
    lambda s: s.update(schema_version=True),
    lambda s: s.update(total_frames=721),
    lambda s: s.update(scale_contract='Same scale.'),
    lambda s: s.update(summary=''),
    lambda s: s['beats'][0].update(end_frame=719),
    lambda s: s['beats'][0].update(indivisible=1),
    lambda s: s['beats'].append(copy.deepcopy(s['beats'][0])),
    lambda s: s.update(dialogues=[{'id': 'line', 'start_frame': 0, 'end_frame': 20,
                                  'speaker': 'Someone', 'text': 'Hello'}]),
])
def test_invalid_contract_rejected_before_compilation(mutation):
    s = contract()
    mutation(s)
    with pytest.raises(ValueError):
        module().validate_contract(s, 720)


def test_whole_dialogue_has_one_owner_and_no_safety_tail_repetition():
    p = module()
    s = contract()
    s['dialogues'] = [
        {'id': 'first', 'start_frame': 240, 'end_frame': 280, 'speaker': '<Subject 1> (S1)',
         'text': 'Only the first line.', 'language': 'English', 'audio_slot': 2},
        {'id': 'second', 'start_frame': 300, 'end_frame': 336, 'speaker': '<Subject 1> (S1)',
         'text': 'Only the second line.', 'language': 'English'},
    ]
    windows = p.plan_windows(720, protected_intervals=p.protected_intervals(s))
    outputs = [p.compile_window(s, w, incoming_state='Observed: right hand holds the cup; tracking camera continues.')
               for w in windows]
    assert [x for o in outputs for x in o['dialogue_ids']] == ['first', 'second']
    assert 'Only the second line.' not in outputs[0]['prompt']
    assert 'Only the first line.' not in outputs[1]['prompt']
    assert '<Audio 2>' in outputs[0]['prompt']
    assert '<d>[English] Only the second line.</d>' in outputs[1]['prompt']
    for o in outputs:
        for section in ('subject_definitions', 'summary', 'retention_analysis', 'detailed_description',
                        'overall_soundscape', 'non_diegetic_music'):
            assert o['prompt'].count(section + ':') == 1
        assert s['scale_contract'] in o['prompt']
        assert '[Shot' not in o['prompt']
        assert '<Video ' not in o['prompt']


def test_local_beat_mapping_and_observed_incoming_state():
    p = module()
    s = contract()
    s['beats'].insert(1, {'id': 'gesture', 'start_frame': 312, 'end_frame': 336,
                          'action': 'Raise the empty left hand.', 'indivisible': True})
    w = p.plan_windows(720)[1]
    with pytest.raises(ValueError, match='observed|incoming'):
        p.compile_window(s, w)
    out = p.compile_window(s, w, incoming_state='Observed cup contact, gait and camera motion.')
    beat = next(b for b in out['mapped_beats'] if b['id'] == 'gesture')
    assert (beat['local_start'], beat['local_end']) == (57, 81)
    assert '2.375' in out['prompt'] and '3.375' in out['prompt']
    assert 'Observed cup contact' in out['prompt']
    assert 'ongoing' in out['prompt'].lower()


def test_unsafe_dialogue_window_and_plan_state_fallback_are_rejected():
    p = module()
    s = contract()
    s['dialogues'] = [{'id': 'line', 'start_frame': 280, 'end_frame': 310,
                       'speaker': '<Subject 1> (S1)', 'text': 'An unbroken sentence.'}]
    with pytest.raises(ValueError, match='dialogue'):
        p.compile_window(s, p.plan_windows(720)[0])
    s['planned_outgoing_state'] = 'A plausible but unobserved tail.'
    with pytest.raises(ValueError, match='observed|incoming'):
        p.compile_window(s, p.plan_windows(720)[1])


def test_appending_story_preserves_existing_prefix_prompt():
    p = module()
    short = contract()
    short['beats'][0]['end_frame'] = 720
    long = copy.deepcopy(short)
    long['total_frames'] = 1440
    long['beats'].append({'id': 'later', 'start_frame': 720, 'end_frame': 1440,
                          'action': 'Set the cup down at the courtyard table.', 'indivisible': False})
    a = p.compile_window(short, p.plan_windows(720)[0])
    b = p.compile_window(long, p.plan_windows(1440)[0])
    assert a == b


def test_example_contract_is_valid_but_does_not_claim_mounted_assets():
    p = module()
    path = ROOT / 'templates/u11_long_video_contract.json'
    assert path.is_file(), 'Long-video contract template is missing'
    s = p.validate_contract(path.read_text(encoding='utf-8'), 720)
    assert s['reference_assets']
    assert s['scale_reference']['approved'] is False
    assert s['scale_reference']['mounted'] is False
    windows = p.plan_windows(720, protected_intervals=p.protected_intervals(s))
    assert windows[-1]['new_end'] == 720
