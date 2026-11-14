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


def test_audio_reference_slot_three_valid_four_rejected_without_limiting_speakers():
    value = contract()
    value['dialogues'] = [{'id':'many-speakers','start_frame':0,'end_frame':24,
        'speaker':'<Subject 7> (S8)','text':'Hello.','audio_slot':3}]
    assert module().validate_contract(value,720)['dialogues'][0]['speaker'] == '<Subject 7> (S8)'
    value['dialogues'][0]['audio_slot'] = 4
    with pytest.raises(ValueError,match='audio_slot'):
        module().validate_contract(value,720)


def test_exact_window_target_recipe_is_carried_separately_from_audio_reference_slots():
    p = module()
    value = contract()
    first, second, *_ = p.plan_windows(720)
    recipe = {'kind': 'complete_wav', 'sha256': 'a' * 64}
    value['window_overrides'] = [{'new_start': first['new_start'], 'new_end': first['new_end'], 'target_audio': recipe}]
    compiled = p.compile_window(value, first)
    assert compiled['target_audio_recipe'] == recipe
    assert 'target_audio_recipe' not in p.compile_window(value, second, 'Actual state')
    assert '<Audio' not in compiled['prompt']


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


def test_speech_is_interleaved_with_overlapping_timed_actions_in_local_time_order():
    p = module()
    s = contract()
    s['beats'][0].update(action='Look, speak, then close the lips.', timed_actions=[
        {'start_frame': 0, 'end_frame': 36, 'action': 'Look toward the camera.'},
        {'start_frame': 36, 'end_frame': 187, 'action': 'Keep the empty palm visible.'},
        {'start_frame': 187, 'end_frame': 720, 'action': 'Close the lips and rest the hand.'},
    ])
    s['dialogues'] = [{'id': 'opening', 'start_frame': 36, 'end_frame': 187,
                       'speaker': '<Subject 1> (S1)', 'text': 'The whole opening sentence.'}]
    result = p.compile_window(s, p.plan_windows(720, protected_intervals=p.protected_intervals(s))[0])
    text = result['prompt']
    ordered = ['Look toward the camera.', 'Keep the empty palm visible.',
               '<d>The whole opening sentence.</d>', 'Close the lips and rest the hand.']
    assert [text.index(part) for part in ordered] == sorted(text.index(part) for part in ordered)
    assert 'local frames' not in text, 'Frame arithmetic belongs in the audit, not H3 scene prose'
    assert result['mapped_beats'][0]['timed_actions'][1]['local_start'] == 36


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


def phase_contract(total=2160):
    s = contract(total)
    s['beats'][0].update(id='drink', action='Pick up the cup, sip, put it down, and speak.',
                         camera='Follow the cup rising to the lips.', sound='A sip and cup contact.')
    s['summary'] = 'Pick up the cup and sip before speaking.'
    return s


@pytest.mark.parametrize('index', [0, 1])
def test_exact_remaining_phase_replaces_entire_original_beat_prose(index):
    p = module()
    total = 294 if index == 0 else 2160
    s = phase_contract(total)
    w = p.plan_windows(total)[index]
    override = {'new_start': w['new_start'], 'new_end': w['new_end'],
                'remaining_beats': [{'id': 'drink', 'action': 'Speak with small empty-hand gestures.'}],
                'sections': {'summary': 'The cup is already on the table; continue the conversation.'}}
    s['window_overrides'] = [override]
    before = copy.deepcopy(s)
    result = p.compile_window(s, w, 'Observed empty hands; cup stable on table.')
    assert s == before
    assert 'Speak with small empty-hand gestures.' in result['prompt']
    for completed in ('Pick up', 'sip', 'cup rising', 'cup contact'):
        assert completed not in result['prompt']
    assert result['mapped_beats'][0]['action'] == override['remaining_beats'][0]['action']
    assert 'camera' not in result['mapped_beats'][0]
    assert 'sound' not in result['mapped_beats'][0]
    assert result['audit']['original_beats'][0]['action'] == before['beats'][0]['action']
    assert result['audit']['original_sections']['summary'] == before['summary']
    assert result['audit']['window_override'] == override
    assert result['reference_selection'] is None and result['voice_policy'] == 'inherit'
    assert result['forward_frames'] == 0


@pytest.mark.parametrize('mutation', [
    lambda x: x.update(mystery='guess'),
    lambda x: x.update(new_end=5000),
    lambda x: x.update(forward_frames=1),
    lambda x: x.update(forward_frames=True),
    lambda x: x.update(sections={'summery': 'typo'}),
    lambda x: x.update(remaining_beats=[{'id': 'missing', 'action': 'Nothing'}]),
    lambda x: x.update(remaining_beats=[{'id': 'walk', 'action': 'Move', 'unknown': True}]),
    lambda x: x.update(remaining_beats=[{'id': 'walk', 'action': 'Move'}, {'id': 'walk', 'action': 'Again'}]),
])
def test_malformed_or_unknown_phase_overrides_fail_validation(mutation):
    s = contract()
    item = {'new_start': 294, 'new_end': 549, 'remaining_beats': [{'id': 'walk', 'action': 'Move'}]}
    mutation(item)
    s['window_overrides'] = [item]
    with pytest.raises(ValueError):
        module().validate_contract(s, 720)


def test_ambiguous_and_nonexact_overrides_fail_before_model_prompt():
    p = module()
    s = contract()
    item = {'new_start': 294, 'new_end': 549, 'sections': {'summary': 'Continue.'}}
    s['window_overrides'] = [item, copy.deepcopy(item)]
    with pytest.raises(ValueError, match='duplicate|overlap|ambiguous'):
        p.validate_contract(s, 720)
    s['window_overrides'] = [{**item, 'new_end': 548}]
    with pytest.raises(ValueError, match='exact'):
        p.compile_window(s, p.plan_windows(720)[1], 'Observed state.')


def test_explicit_event_clocks_convert_with_sentence_punctuation_and_keep_dimensions():
    p = module()
    s = contract(2160)
    w = p.plan_windows(2160)[6]  # global 65.375--76.000, local 1.625--12.250
    s['beats'][0]['action'] = ('Gesture at 65.400. Settle by 73.400. 在65.400秒抬手，到73.400时放下。 '
                               '在65.400。于73.400。Adult 165cm, cup 8.5cm, stand at 12.345m.')
    s['dialogues'] = [{'id': 'literal', 'start_frame': 1600, 'end_frame': 1650,
                       'speaker': '<Subject 1> (S1)',
                       'text': 'At 65.400. <Picture 9> and <Audio 3> cost 12.345m.'}]
    result = p.compile_window(s, w, 'Observed empty hands.')
    prompt = result['prompt']
    assert 'at 1.650.' in prompt and 'by 9.650.' in prompt
    assert '在1.650秒' in prompt and '到9.650时' in prompt
    assert '在1.650。于9.650。' in prompt
    assert '165cm' in prompt and '8.5cm' in prompt and '12.345m' in prompt
    assert '<d>At 65.400. <Picture 9> and <Audio 3> cost 12.345m.</d>' in prompt


def test_forward_phase_and_structured_actions_skip_consumed_events():
    p = module()
    s = phase_contract(2160)
    w = p.plan_windows(2160)[6]
    free = w['new_start'] + 17
    s['window_overrides'] = [{'new_start': w['new_start'], 'new_end': w['new_end'], 'forward_frames': 17,
        'remaining_beats': [{'id': 'drink', 'action': 'Completed drinking is audit only.', 'timed_actions': [
            {'start_frame': w['new_start'], 'end_frame': free, 'action': 'Already consumed gesture.'},
            {'start_frame': free, 'end_frame': w['new_end'],
             'action': 'Continue empty hands at {{frame:' + str(free) + '}}.'}]}],
        'sections': {'summary': 'Continue the conversation with empty hands.'}}]
    result = p.compile_window(s, w, 'Observed tabled cup.')
    assert result['forward_frames'] == 17
    assert 'Already consumed gesture.' not in result['prompt']
    assert 'Completed drinking' not in result['prompt'] and 'Pick up' not in result['prompt']
    assert result['mapped_beats'][0]['local_start'] == 56
    assert 'Continue empty hands at 2.333.' in result['prompt']
    assert 'New visible content occupies local 2.333' in result['prompt']


def test_explicit_past_clock_requires_current_phase_override():
    p = module()
    s = contract()
    s['beats'][0]['action'] = 'Pick up at 1.000 seconds, then walk.'
    with pytest.raises(ValueError, match='remaining_beats|current phase'):
        p.compile_window(s, p.plan_windows(720)[1], 'Observed holding the cup.')


def test_selection_remaps_only_directing_markers_and_preserves_spoken_literals():
    p = module()
    s = contract()
    s['subject_definitions'] = '<Subject 7> (S4): identity from <Picture 9>; timbre from <Audio 3>.'
    s['reference_selection'] = {'pictures': [9, 2], 'audios': [3]}
    s['dialogues'] = [{'id': 'literal', 'start_frame': 50, 'end_frame': 75,
        'speaker': '<Subject 7> (S4)', 'audio_slot': 3,
        'text': '<Picture 9> and <Audio 3> at 65.400.'}]
    result = p.compile_window(s, p.plan_windows(720)[0])
    assert '<Subject 7> (S4): identity from <Picture 1>; timbre from <Audio 1>.' in result['prompt']
    assert '<d><Picture 9> and <Audio 3> at 65.400.</d>' in result['prompt']
    assert result['audit']['reference_maps'] == {'pictures': {'9': 1, '2': 2}, 'audios': {'3': 1}}
    s['reference_selection']['pictures'] = [2]
    with pytest.raises(ValueError, match='Picture 9|dangling'):
        p.compile_window(s, p.plan_windows(720)[0])


def test_dialogue_only_quiet_policy_removes_binding_prose_but_retains_identity():
    p = module()
    s = contract()
    s['voice_policy'] = 'dialogue_only'
    s['subject_definitions'] = '<Subject 1> (S1): adult in a blue coat; <Audio 2> is voice timbre reference only.'
    s['retention_analysis'] += ' Audio references define timbre only.'
    result = p.compile_window(s, p.plan_windows(720)[1], 'Observed same speaker and coat.')
    assert '<Subject 1> (S1): adult in a blue coat' in result['prompt']
    assert '<Audio ' not in result['prompt'] and 'Audio references define timbre only' not in result['prompt']
    assert 'automatically generated' not in result['prompt']
    assert result['voice_policy'] == 'dialogue_only' and result['dialogue_ids'] == []
    assert s['scale_contract'] + '\nAll times below are local' in result['prompt']
    s['subject_definitions'] = '<Subject 1> (S1): blue coat and <Audio 2> identify the person.'
    with pytest.raises(ValueError, match='subject_definitions|local|binding'):
        p.compile_window(s, p.plan_windows(720)[1], 'Observed state.')


@pytest.mark.parametrize('seconds', [30, 60, 90, 97.125])
def test_phase_extensions_preserve_duration_and_legacy_compiler_interface(seconds):
    p = module()
    total = p.target_frame_count(seconds)
    s = contract(total)
    windows = p.plan_windows(total)
    outputs = [p.compile_window(s, w, 'Observed continuous walk.') for w in windows]
    assert windows[-1]['new_end'] == total
    assert all(set(o) == {'prompt', 'dialogue_ids', 'mapped_beats', 'reference_selection',
                          'voice_policy', 'forward_frames', 'audit'} for o in outputs)


@pytest.mark.parametrize('field,value', [('voice_policy', 'quiet'), ('voice_policy', []),
    ('forward_frames', 17), ('remaining_beats', []), ('anchors', [])])
def test_unknown_or_misplaced_advanced_contract_fields_fail(field, value):
    s = contract()
    s[field] = value
    with pytest.raises(ValueError):
        module().validate_contract(s, 720)


@pytest.mark.parametrize('token', ['{{frame:abc}}', '{{frame:-1}}', '{{frame:24.5}}'])
def test_malformed_explicit_frame_tokens_fail(token):
    s = contract()
    s['beats'][0]['action'] = 'Gesture at ' + token + '.'
    with pytest.raises(ValueError, match='frame|clock'):
        module().compile_window(s, module().plan_windows(720)[0])


def test_all_six_current_scene_sections_override_original_prose():
    p = module()
    s = contract()
    w = p.plan_windows(720)[1]
    texts = {key: key + ' current phase.' for key in
             ('subject_definitions', 'summary', 'retention_analysis', 'detailed_description',
              'overall_soundscape', 'non_diegetic_music')}
    s['window_overrides'] = [{'new_start': w['new_start'], 'new_end': w['new_end'], 'sections': texts}]
    result = p.compile_window(s, w, 'Observed walking.')
    assert all(text in result['prompt'] for text in texts.values())
    assert s['summary'] not in result['prompt']
    assert result['audit']['original_sections']['summary'] == s['summary']


def test_structured_source_actions_render_only_current_time_range():
    p = module()
    s = contract()
    s['beats'][0].update(action='Pick up and sip, then gesture.', timed_actions=[
        {'start_frame': 0, 'end_frame': 294, 'action': 'Pick up and sip.'},
        {'start_frame': 294, 'end_frame': 720, 'action': 'Gesture with empty hands.'}])
    result = p.compile_window(s, p.plan_windows(720)[1], 'Observed cup on table.')
    assert 'Pick up' not in result['prompt']
    assert 'Gesture with empty hands.' in result['prompt']


def test_mapped_beat_references_use_the_same_window_numbers_as_prompt():
    p = module()
    s = contract()
    s['reference_selection'] = {'pictures': [9], 'audios': []}
    s['beats'][0]['action'] = 'Gesture like <Picture 9>.'
    result = p.compile_window(s, p.plan_windows(720)[0])
    assert result['mapped_beats'][0]['action'] == 'Gesture like <Picture 1>.'


def test_structured_actions_have_deterministic_clock_order_and_covering_metadata():
    p = module()
    s = contract()
    s['beats'][0]['timed_actions'] = [
        {'start_frame': 340, 'end_frame': 500, 'action': 'Second phase.'},
        {'start_frame': 294, 'end_frame': 340, 'action': 'First phase.'}]
    result = p.compile_window(s, p.plan_windows(720)[1], 'Observed state.')
    beat = result['mapped_beats'][0]
    assert (beat['global_start'], beat['global_end']) == (294, 500)
    assert result['prompt'].index('First phase.') < result['prompt'].index('Second phase.')


def test_preflight_can_reject_a_nonmatching_later_override_before_first_window_queue():
    p = module()
    s = contract()
    windows = p.plan_windows(720)
    s['window_overrides'] = [{'new_start': 549, 'new_end': 719, 'sections': {'summary': 'Local phase.'}}]
    with pytest.raises(ValueError, match='exact'):
        p.validate_window_overrides(s, windows)
    s['window_overrides'][0]['new_end'] = 720
    result = p.validate_window_overrides(s, windows)
    assert result == s['window_overrides']
    result[0]['sections']['summary'] = 'Changed result only.'
    assert s['window_overrides'][0]['sections']['summary'] == 'Local phase.'


def test_explicit_empty_audio_selection_removes_only_binding_prose_and_keeps_owned_speech():
    p = module()
    s = contract()
    s['reference_selection'] = {'pictures': [], 'audios': []}
    s['subject_definitions'] = '<Subject 1> (S1): same adult; <Audio 1> is voice timbre reference only.'
    s['dialogues'] = [{'id': 'line', 'start_frame': 50, 'end_frame': 100,
        'speaker': '<Subject 1> (S1)', 'audio_slot': 1, 'text': 'Say <Audio 1> literally.'}]
    result = p.compile_window(s, p.plan_windows(720)[0])
    assert result['voice_policy'] == 'inherit' and result['dialogue_ids'] == ['line']
    assert '<Subject 1> (S1): same adult' in result['prompt']
    assert 'voice timbre reference only' not in result['prompt']
    assert 'using <Audio' not in result['prompt']
    assert '<d>Say <Audio 1> literally.</d>' in result['prompt']
    assert 'automatically generated' not in result['prompt']


@pytest.mark.parametrize('text', ['One adult and 2 cups.', 'Walk from 1 to 3 meters.',
                                  'Hold for 3 seconds.', '在此保持3秒，然后走2米。'])
def test_counts_measurement_ranges_and_relative_durations_are_not_absolute_event_clocks(text):
    p = module()
    s = contract()
    s['summary'] = text
    s['beats'][0]['action'] = text
    for window in p.plan_windows(720):
        result = p.compile_window(s, window, 'Observed the same scene.')
        assert text in result['prompt']
        assert result['mapped_beats'][0]['action'] == text


@pytest.mark.parametrize('text', ['Use <Picture 1', 'Use <Audio 1', 'Use <Picture1>'])
def test_malformed_directing_reference_syntax_fails_compilation_but_preserves_speech(text):
    p = module()
    s = contract()
    s['subject_definitions'] = text
    with pytest.raises(ValueError, match='reference|marker'):
        p.compile_window(s, p.plan_windows(720)[0])
    s = contract()
    s['dialogues'] = [{'id': 'line', 'start_frame': 50, 'end_frame': 100,
        'speaker': '<Subject 1> (S1)', 'text': text}]
    assert '<d>' + text + '</d>' in p.compile_window(s, p.plan_windows(720)[0])['prompt']


@pytest.mark.parametrize('source,expected', [
    ('Gesture from 65.400 seconds to 73.400 seconds.', 'Gesture from 1.650 seconds to 9.650 seconds.'),
    ('Gesture between 65.400s and 73.400s.', 'Gesture between 1.650s and 9.650s.'),
    ('从65.400秒到73.400秒保持手势。', '从1.650秒到9.650秒保持手势。'),
])
def test_explicit_unit_labelled_event_ranges_convert_both_endpoints(source, expected):
    p = module()
    s = contract(2160)
    s['beats'][0]['action'] = source
    result = p.compile_window(s, p.plan_windows(2160)[6], 'Observed empty hands.')
    assert expected in result['prompt']
    assert result['mapped_beats'][0]['action'] == expected


def test_remaining_phase_makes_original_global_action_summary_audit_only_by_default():
    p = module()
    s = phase_contract()
    w = p.plan_windows(2160)[1]
    s['window_overrides'] = [{'new_start': w['new_start'], 'new_end': w['new_end'],
        'remaining_beats': [{'id': 'drink', 'action': 'Speak with small empty-hand gestures.'}]}]
    result = p.compile_window(s, w, 'Observed empty hands and cup already on table.')
    assert 'Pick up' not in result['prompt'] and 'sip' not in result['prompt']
    assert result['audit']['original_sections']['summary'] == s['summary']
    assert 'Speak with small empty-hand gestures.' in result['prompt']


@pytest.mark.parametrize('marker', ['<Audio 0>', '<Audio -1>', '<Audio 1.5>', '<Audio x>',
                                  '<Audio1>', '<Audio 1'])
def test_empty_audio_selection_validates_marker_syntax_before_removing_binding_clause(marker):
    p = module()
    s = contract()
    s['reference_selection'] = {'pictures': [], 'audios': []}
    s['subject_definitions'] = '<Subject 1> (S1): same adult; ' + marker + ' is voice timbre reference only.'
    with pytest.raises(ValueError, match='reference|marker'):
        p.compile_window(s, p.plan_windows(720)[0])
    s['subject_definitions'] = '<Subject 1> (S1): same adult.'
    s['dialogues'] = [{'id': 'literal', 'start_frame': 50, 'end_frame': 100,
        'speaker': '<Subject 1> (S1)', 'text': marker + ' is voice timbre reference only.'}]
    prompt = p.compile_window(s, p.plan_windows(720)[0])['prompt']
    assert '<d>' + marker + ' is voice timbre reference only.</d>' in prompt


def test_timed_remaining_actions_inherit_effective_defaults_then_override_per_action():
    p = module()
    s = phase_contract(720)
    w = p.plan_windows(720)[1]
    s['window_overrides'] = [{'new_start': w['new_start'], 'new_end': w['new_end'],
        'remaining_beats': [{'id': 'drink', 'action': 'Completed onset stays audit-only.',
            'camera': 'Keep a locked wide shot.', 'sound': 'Keep rain constant.',
            'timed_actions': [
                {'start_frame': 294, 'end_frame': 400, 'action': 'Gesture with empty hands.'},
                {'start_frame': 400, 'end_frame': 549, 'action': 'Meet the other adult\'s gaze.',
                 'camera': 'Ease toward a medium shot.', 'sound': 'Keep rain and one distant bird.'}]}]}]
    result = p.compile_window(s, w, 'Observed empty hands and cup already on table.')
    first, second = result['mapped_beats'][0]['timed_actions']
    assert first['camera'] == 'Keep a locked wide shot.'
    assert first['sound'] == 'Keep rain constant.'
    assert second['camera'] == 'Ease toward a medium shot.'
    assert second['sound'] == 'Keep rain and one distant bird.'
    for part in (first, second):
        assert part['camera'] in result['prompt'] and part['sound'] in result['prompt']
    assert 'Completed onset' not in result['prompt']
    assert 'Pick up' not in result['prompt'] and 'sip' not in result['prompt']
    assert 'cup rising' not in result['prompt'] and 'cup contact' not in result['prompt']


def test_timed_remaining_actions_do_not_inherit_removed_original_camera_and_sound():
    p = module()
    s = phase_contract(720)
    w = p.plan_windows(720)[1]
    s['window_overrides'] = [{'new_start': w['new_start'], 'new_end': w['new_end'],
        'remaining_beats': [{'id': 'drink', 'action': 'Current phase only.', 'timed_actions': [
            {'start_frame': 294, 'end_frame': 549, 'action': 'Gesture with empty hands.'}]}]}]
    result = p.compile_window(s, w, 'Observed empty hands and cup already on table.')
    part = result['mapped_beats'][0]['timed_actions'][0]
    assert 'camera' not in part and 'sound' not in part
    assert s['beats'][0]['camera'] not in result['prompt']
    assert s['beats'][0]['sound'] not in result['prompt']
    assert result['audit']['original_beats'][0]['camera'] == s['beats'][0]['camera']


@pytest.mark.parametrize('unit', ['秒', '时'])
def test_chinese_event_clocks_allow_whitespace_before_time_units(unit):
    p = module()
    s = contract(2160)
    w = p.plan_windows(2160)[6]
    s['beats'][0]['action'] = f'在 65.400 {unit}抬手，到 73.400 {unit}放下。'
    result = p.compile_window(s, w, 'Observed empty hands.')
    assert f'在 1.650 {unit}抬手，到 9.650 {unit}放下。' in result['prompt']
    s['beats'][0]['action'] = f'在 1.000 {unit}拿起杯子。'
    with pytest.raises(ValueError, match='current phase|remaining_beats'):
        p.compile_window(s, w, 'Observed empty hands.')


def test_chinese_paired_event_range_allows_whitespace_before_units():
    p = module()
    s = contract(2160)
    s['beats'][0]['action'] = '从 65.400 秒 到 73.400 秒保持手势。'
    result = p.compile_window(s, p.plan_windows(2160)[6], 'Observed empty hands.')
    assert '从 1.650 秒 到 9.650 秒保持手势。' in result['prompt']


def test_timed_action_onset_and_continuation_use_each_action_interval():
    p = module()
    s = contract()
    s['beats'][0].update(action='Generic backbone only.', timed_actions=[
        {'start_frame': 294, 'end_frame': 340, 'action': 'Pick up for the first time.'},
        {'start_frame': 340, 'end_frame': 500, 'action': 'Start drinking.'},
        {'start_frame': 100, 'end_frame': 315, 'action': 'Keep the established gait.'},
        {'start_frame': 500, 'end_frame': 600, 'action': 'Continue talking into the next window.'}])
    result = p.compile_window(s, p.plan_windows(720)[1], 'Observed walking toward the cup.')
    parts = {part['action']: part for part in result['mapped_beats'][0]['timed_actions']}
    for action in ('Pick up for the first time.', 'Start drinking.'):
        assert parts[action]['partial_start'] is False
        assert parts[action]['partial_end'] is False
        line = next(line for line in result['prompt'].splitlines() if action in line)
        assert 'without replaying its onset' not in line
    ongoing = parts['Keep the established gait.']
    assert ongoing['partial_start'] is True and ongoing['partial_end'] is False
    line = next(line for line in result['prompt'].splitlines() if 'Keep the established gait.' in line)
    assert 'without replaying its onset' in line
    assert parts['Continue talking into the next window.']['partial_end'] is True


def test_pure_contract_modules_compile_and_bind_without_director_comfy_or_torch():
    import subprocess
    import sys
    script = r'''
import ast
import json
from pathlib import Path
import sys
import types

root = Path(sys.argv[1])
package = types.ModuleType('_isolated_u11_cpu')
package.__path__ = []
sys.modules[package.__name__] = package
modules = {}
allowed = {'__future__', 'copy', 'json', 'math', 're', 'decimal', 'fractions'}
for name in ('long_video_references', 'long_video_plan'):
    path = root / 'nodes' / (name + '.py')
    raw = path.read_bytes()
    for node in ast.walk(ast.parse(raw)):
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module == 'long_video_references':
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [item.name.split('.')[0] for item in node.names] if isinstance(node, ast.Import) else [node.module]
            assert all(item in allowed for item in names), 'Pure U11 module imports an application dependency'
    module = types.ModuleType(package.__name__ + '.' + name)
    module.__package__ = package.__name__
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    modules[name] = module
api, refs = modules['long_video_plan'], modules['long_video_references']
scene = json.loads(sys.argv[2])
scene['reference_selection'] = {'pictures': [1], 'audios': []}
scene = api.validate_contract(scene, 720)
compiled = api.compile_window(scene, api.plan_windows(720)[0])
settings = {'prompt': compiled['prompt'], 'voice_mode': 'none', 'reference_image_1_file': 'CPU_STATIC_FIXTURE_ONLY.png'}
trace = refs.apply_reference_policy(settings, compiled, {})
assert trace['selection'] == compiled['reference_selection']
assert trace['preprocessing_validation'] == 'not_performed_static'
assert settings['first_image_file'] == 'CPU_STATIC_FIXTURE_ONLY.png'
assert not any(name == 'torch' or name.startswith(('torch.', 'comfy', '_isolated_u11_cpu.director')) for name in sys.modules)
print('pure_compile_and_static_bindings_passed')
'''
    result = subprocess.run([sys.executable, '-I', '-S', '-c', script, str(ROOT), json.dumps(contract())],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == 'pure_compile_and_static_bindings_passed'
