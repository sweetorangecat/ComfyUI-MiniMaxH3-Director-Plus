"""The original Director prompt is the only ordinary long-video authoring input."""
import copy
import importlib
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def parser():
    assert (ROOT / 'nodes/long_video_prompt.py').is_file(), 'Original-prompt compiler is missing'
    return importlib.import_module('nodes.long_video_prompt')


def planner():
    return importlib.import_module('nodes.long_video_plan')


def prompt(detail, *, subjects='<Subject 1> (S1) is the same adult from <Picture 1>.',
           summary='A quiet courtyard walk.', sound='Footsteps and quiet courtyard ambience.'):
    return ('subject_definitions:\n' + subjects + '\n\nsummary:\n' + summary
            + '\n\nretention_analysis:\nPreserve identity, clothes and reference-relative scale 1:1.'
            + '\n\ndetailed_description:\n' + detail
            + '\n\noverall_soundscape:\n' + sound + '\n\nnon_diegetic_music:\nN/A')


def compile_all(contract):
    p = planner()
    windows = p.plan_windows(contract['total_frames'], protected_intervals=p.protected_intervals(contract))
    return [p.compile_window(contract, window,
                            'CPU TEST ONLY: no actual AV observation or approval.' if i else '')
            for i, window in enumerate(windows)]


@pytest.mark.parametrize('seconds', [30, 90, 120])
@pytest.mark.parametrize('language', ['en', 'zh'])
def test_single_full_length_six_section_prompt_compiles_arbitrary_duration(seconds, language):
    actions = ['Walk toward the courtyard arch.', 'Keep walking along the path.', 'Walk past the trees.']
    if language == 'zh':
        actions = ['朝院门自然行走。', '沿小路继续自然行走。', '从树旁自然走过。']
    bounds = [0, 8, 16, seconds]
    detail = '\n'.join(f'[{a}-{b}s] {action}' for a, b, action in zip(bounds, bounds[1:], actions))
    contract = parser().contract_from_prompt(prompt(detail), seconds)
    assert contract['schema_version'] == 1
    assert contract['total_frames'] == seconds * 24
    assert planner().validate_contract(contract, seconds * 24) == contract
    assert [(beat['start_frame'], beat['end_frame']) for beat in contract['beats']] == [
        (0, 192), (192, 384), (384, seconds * 24)]
    assert [beat['action'] for beat in contract['beats']] == actions
    assert compile_all(contract)[-1]['mapped_beats'][-1]['global_end'] == seconds * 24


@pytest.mark.parametrize('detail', [
    '[0-8s] Walk forward.\n[00:08-00:16] Keep walking.\n[16-30秒] Walk past the tree.',
    '[0–8s] Walk forward.\n[00:08–00:16] Keep walking.\n[16–30秒] Walk past the tree.',
    '00:00.000-00:08.000 Walk forward.\n00:08.000-00:16.000 Keep walking.\n00:16.000-00:30.000 Walk past the tree.',
])
def test_global_clock_spellings_and_en_dash_have_the_same_intervals(detail):
    value = parser().contract_from_prompt(prompt(detail), 30)
    assert [(b['start_frame'], b['end_frame']) for b in value['beats']] == [(0, 192), (192, 384), (384, 720)]


def test_complete_dialogue_words_and_original_subject_speaker_audio_ids_appear_once():
    words = 'Wait—keep “<Picture 99>” in the sentence; do not shorten this complete line.'
    subjects = ('Mara <Subject 7> (S8) is the adult from <Picture 1>.\n'
                '<Audio 2> is her voice reference only.')
    detail = ('[0-8s] Walk forward.\n'
              f'[8-11s] Mara <Subject 7> (S8) uses <Audio 2>, looks right and says: <d>[English]{words}</d>\n'
              '[11-30s] Keep walking quietly.')
    value = parser().contract_from_prompt(prompt(detail, subjects=subjects), 30)
    assert len(value['dialogues']) == 1
    line = value['dialogues'][0]
    assert line['speaker'] == '<Subject 7> (S8)'
    assert line['text'] == words
    assert line['language'] == 'English'
    assert line['audio_slot'] == 2
    assert (line['start_frame'], line['end_frame']) == (192, 264)
    compiled = compile_all(value)
    assert sum(item['prompt'].count(words) for item in compiled) == 1
    assert sum(item['dialogue_ids'].count(line['id']) for item in compiled) == 1
    assert '<Subject 7> (S8)' in next(item['prompt'] for item in compiled if item['dialogue_ids'])
    for beat in value['beats']:
        assert words not in beat['action'] and '<d>' not in beat['action']
    for section in ('subject_definitions', 'summary', 'retention_analysis', 'overall_soundscape', 'non_diegetic_music'):
        assert words not in value[section]


def test_same_words_at_two_explicit_times_are_two_deliberate_utterances():
    detail = ('[0-2s] <Subject 1> (S1) says: <d>Stay.</d>\n[2-16s] Walk quietly.\n'
              '[16-18s] <Subject 1> (S1) says: <d>Stay.</d>\n[18-30s] Walk quietly.')
    value = parser().contract_from_prompt(prompt(detail), 30)
    assert len({line['id'] for line in value['dialogues']}) == 2
    assert sum(item['prompt'].count('<d>Stay.</d>') for item in compile_all(value)) == 2


def test_speaker_binding_can_be_inherited_from_subject_definition_without_renumbering():
    detail = '[0-3s] (S8) says: <d>Keep going.</d>\n[3-30s] Walk quietly.'
    value = parser().contract_from_prompt(prompt(detail, subjects='<Subject 7> (S8) is Mara.'), 30)
    assert value['dialogues'][0]['speaker'] == '<Subject 7> (S8)'


def test_no_supplied_voice_does_not_invent_audio_or_asset_approvals():
    detail = '[0-3s] <Subject 1> (S1) says: <d>Keep going.</d>\n[3-30s] Walk quietly.'
    value = parser().contract_from_prompt(prompt(detail), 30,
                                        {'voice_mode': 'none', 'reference_image_1_file': 'identity.png'})
    assert 'audio_slot' not in value['dialogues'][0]
    assert '<Audio' not in '\n'.join(item['prompt'] for item in compile_all(value))
    assert not {'approved', 'mounted', 'reference_assets', 'scale_reference', 'incoming_state'} & value.keys()
    assert '1:1' in value['scale_contract']
    assert 'instruction' in value['scale_contract'].lower()
    assert not any(unit in value['scale_contract'] for unit in ('cm', 'centimet', 'height', 'measured'))


def test_original_file_and_link_counts_are_inherited_without_new_selection_or_mutation():
    settings = {'reference_image_1_file': 'identity.png', 'reference_image_2': ['source-images', 0],
                'voice_mode': 'h3_reference', 'voice_reference_audio_file': 'mara.wav',
                'voice_reference_audio_2': ['source-audio', 1]}
    original = copy.deepcopy(settings)
    subjects = '<Subject 7> (S8) is Mara from <Picture 2>. <Audio 2> is her voice reference only.'
    detail = '[0-3s] <Subject 7> (S8) uses <Audio 2> and says: <d>Keep going.</d>\n[3-30s] Walk quietly.'
    value = parser().contract_from_prompt(prompt(detail, subjects=subjects), 30, settings)
    assert settings == original
    assert 'reference_selection' not in value
    assert 'window_overrides' not in value
    assert value['dialogues'][0]['audio_slot'] == 2
    assert '<Picture 2>' in value['subject_definitions']
    assert '<Audio 2>' in value['subject_definitions']


@pytest.mark.parametrize('kind,subjects,settings', [
    ('Picture', '<Subject 1> (S1) is the adult from <Picture 2>.', {'reference_image_1_file': 'one.png'}),
    ('Audio', '<Subject 1> (S1) is an adult. <Audio 1> is her voice reference only.', {'voice_mode': 'none'}),
])
def test_missing_actual_original_reference_is_a_prompt_error(kind, subjects, settings):
    with pytest.raises(ValueError, match=kind) as error:
        parser().contract_from_prompt(prompt('[0-30s] Walk quietly.', subjects=subjects), 30, settings)
    assert 'JSON' not in str(error.value)


def test_timed_events_do_not_leak_to_unrelated_windows_or_static_summary():
    source = prompt('[0-8s] Touch the brass latch once.\n[8-16s] Turn away.\n[16-30s] Walk through the courtyard.',
                    summary='A quiet courtyard scene in one continuous shot.')
    value = parser().contract_from_prompt(source, 30)
    windows = compile_all(value)
    assert 'Touch the brass latch once' in windows[0]['prompt']
    assert all('Touch the brass latch once' not in item['prompt'] for item in windows[1:])
    assert value['source_prompt_sections']['summary'] == 'A quiet courtyard scene in one continuous shot.'


@pytest.mark.parametrize('seconds', [30, 90])
@pytest.mark.parametrize('shot_label', ['', '[Shot 1]\n'])
def test_stable_summary_visual_and_camera_constraints_survive_every_compiled_window(seconds, shot_label):
    summary = 'Black-and-white infrared security footage in a fixed overhead view.'
    value = parser().contract_from_prompt(prompt(shot_label + f'[0-{seconds}s] Walk at a steady natural pace.',
                                                 summary=summary), seconds)
    assert value['summary'] == summary
    compiled = compile_all(value)
    assert len(compiled) >= 3
    assert all(summary in item['prompt'] for item in compiled)


@pytest.mark.parametrize('summary', [
    'Touch the brass latch once, then turn away and walk through the courtyard.',
    'A woman picks up one cup and drinks its contents. She puts the empty cup on the table.',
    '女子拿起杯子喝完水，把空杯放回桌上。',
    'A woman assembles a sandwich, washes the knife, and serves lunch in one continuous shot.',
])
def test_narrative_summary_requires_relocating_finite_events_without_silently_discarding_them(summary):
    with pytest.raises(ValueError, match='summary.*detailed_description') as error:
        value = parser().contract_from_prompt(prompt('[0-30s] Walk naturally.', summary=summary), 30)
        compile_all(value)
    assert '时间' in str(error.value)
    assert 'JSON' not in str(error.value)


@pytest.mark.parametrize('action', [
    'A woman picks up one cup and drinks its contents. She puts the empty cup on the table.',
    'A woman picks up one cup.',
    'She opens the door and steps outside.',
    '女子拿起一只杯子喝水。她把空杯放到桌上。',
    '她打开门走出房间。',
    'A pilot executes an unfamiliar maneuver.',
    'A woman assembles a sandwich, washes the knife, and serves lunch in one continuous shot.',
    'A woman walks through the courtyard, assembles a sandwich, and serves lunch.',
    'A woman walks through the courtyard and assembles a sandwich.',
    'A woman assembles a sandwich in a fixed overhead shot.',
    'She washes a knife in a static view.',
])
@pytest.mark.parametrize('six_sections', [False, True])
def test_untimed_finite_or_ambiguous_action_cannot_replay_from_completed_incoming_state(action, six_sections):
    with pytest.raises(ValueError, match='时间|持续') as error:
        value = parser().contract_from_prompt(prompt(action) if six_sections else action, 30)
        p = planner()
        windows = p.plan_windows(value['total_frames'], protected_intervals=p.protected_intervals(value))
        for index, window in enumerate(windows):
            p.compile_window(value, window, 'CPU REPRO: cup is empty and already on the table.' if index else '')
    assert 'JSON' not in str(error.value)


@pytest.mark.parametrize('action', [
    'A fixed overhead view of a quiet courtyard with soft rain.',
    'Black-and-white infrared security footage in a fixed overhead view.',
    'A woman is motionless in a fixed overhead view.',
    'A woman stands quietly, holding one cup while the camera tracks alongside.',
    'A woman walks steadily through the courtyard. The camera tracks alongside her.',
    '女子持续自然行走，镜头平稳跟拍。',
])
def test_simple_sustained_or_static_plain_scene_still_compiles_without_headings(action):
    value = parser().contract_from_prompt(action, 30)
    assert value['beats'][0]['action'] == action
    assert compile_all(value)[-1]['mapped_beats'][-1]['global_end'] == 720


def test_plain_untimed_continuous_scene_is_one_continuous_action_without_fictional_timing():
    action = 'Walk at a natural steady pace through the courtyard while the camera tracks alongside.'
    value = parser().contract_from_prompt(prompt(action), 90)
    assert len(value['beats']) == 1
    assert value['beats'][0]['action'] == action
    assert (value['beats'][0]['start_frame'], value['beats'][0]['end_frame']) == (0, 2160)
    assert 'without replaying its onset' in compile_all(value)[1]['prompt']


def test_safe_plain_text_does_not_require_headings_or_invent_subjects_or_voices():
    action = 'A woman walks steadily through a courtyard while the camera tracks alongside.'
    value = parser().contract_from_prompt(action, 90)
    assert value['beats'][0]['action'] == action
    assert not value['dialogues']
    assert not re_ids(value['subject_definitions'])
    assert compile_all(value)[-1]['mapped_beats'][-1]['global_end'] == 2160


def re_ids(text):
    import re
    return re.findall(r'<(?:Subject|Audio)\s+\d+>|\bS\d+\b', text)


def test_prompt_format_errors_explain_the_fix_in_chinese():
    with pytest.raises(ValueError) as error:
        parser().contract_from_prompt(prompt('[0-9s] Walk.\n[8-30s] Continue.'), 30)
    assert '提示词' in str(error.value)
    assert '重叠' in str(error.value)


def test_shot_headers_stay_with_their_global_cut_and_do_not_restart_on_continuation():
    detail = ('[Shot 1]\n[0-8s] Walk forward.\n[8-16s] Keep walking.\n'
              '[Shot 2]\n[16-30s] Track the adult walking past the tree.')
    value = parser().contract_from_prompt(prompt(detail), 30)
    assert '[Shot 2]' not in value['beats'][1]['action']
    assert value['beats'][2]['action'].startswith('[Shot 2]')
    compiled = compile_all(value)
    assert sum(item['prompt'].count('[Shot 2]') for item in compiled) == 1


@pytest.mark.parametrize('detail', ['00:00-00:bad Walk.', '[-1-30s] Walk.', '[0-8s Walk.'])
def test_malformed_timestamp_cannot_silently_become_a_continuous_scene(detail):
    with pytest.raises(ValueError):
        parser().contract_from_prompt(prompt(detail), 30)


@pytest.mark.parametrize('detail', ['<Subject 1> (S1) says: Keep going.', '<Subject 1> (S1) 说：继续走。',
                                  '(S1): Keep going.', '旁白：他仍然向前走。'])
def test_explicit_untagged_untimed_speech_is_not_repeated_as_continuous_prose(detail):
    with pytest.raises(ValueError, match='台词|发言|发声'):
        parser().contract_from_prompt(prompt(detail), 90)


@pytest.mark.parametrize('seconds', [90, 120])
def test_long_full_timeline_keeps_last_complete_utterance_owned_once(seconds):
    detail = (f'[0-{seconds - 5}s] Walk naturally through the courtyard.\n'
              f'[{seconds - 5}-{seconds - 2}s] <Subject 1> (S1) says: <d>我们继续往前走。</d>\n'
              f'[{seconds - 2}-{seconds}s] Keep walking naturally.')
    value = parser().contract_from_prompt(prompt(detail), seconds)
    assert sum(item['prompt'].count('我们继续往前走。') for item in compile_all(value)) == 1


@pytest.mark.parametrize('detail,reason', [
    ('[0-8s] Walk.\n[9-30s] Continue.', '空缺|覆盖'),
    ('[0-9s] Walk.\n[8-30s] Continue.', '重叠'),
    ('[1-30s] Walk.', '0 秒|覆盖'),
    ('[0-16s] Walk.', '终点|覆盖'),
    ('[0-31s] Walk.', '目标|时长'),
    ('[8-0s] Walk.\n[0-30s] Continue.', '时间段|晚于'),
    ('[0-eight seconds] Walk.', '时间段|时间戳'),
    ('[00:70-01:30] Walk.', '时间戳'),
    ('[0-0.001s] Walk.\n[0.001-30s] Continue.', '帧|时间段'),
    ('[0-30s]', '动作|空'),
    ('Walk, then pick up the cup and finally put it down.', '时间段|阶段'),
    ('<Subject 1> (S1) says: <d>Keep going.</d>', '时间'),
    ('<Subject 1> (S1) says: “Keep going.”', '时间|<d>'),
    ('[0-30s] <Subject 1> (S1) says: <d>Keep going.</d>', '台词|发声'),
    ('[0-3s] <Subject 1> (S1) says: <d>One.</d> <d>Two.</d>\n[3-30s] Walk.', '台词|时间段'),
    ('[0-3s] An adult says: <d>Keep going.</d>\n[3-30s] Walk.', 'Subject|Sx|speaker'),
    ('[0-3s] <Subject 1> (S1) says: <d>Keep going.\n[3-30s] Walk.', '<d>|台词'),
])
def test_ambiguous_or_impossible_timing_has_concise_prompt_format_error(detail, reason):
    with pytest.raises(ValueError, match=reason) as error:
        parser().contract_from_prompt(prompt(detail), 30)
    assert '提示词' in str(error.value)
    assert 'JSON' not in str(error.value)
    assert len(str(error.value)) < 450


@pytest.mark.parametrize('section', ['summary', 'retention_analysis', 'overall_soundscape'])
def test_dialogue_in_reusable_sections_is_not_repeated_or_silently_discarded(section):
    source = prompt('[0-30s] Walk quietly.')
    source = source.replace(section + ':\n', section + ':\n<Subject 1> (S1): <d>Keep going.</d>\n')
    with pytest.raises(ValueError, match='detailed_description|timed'):
        parser().contract_from_prompt(source, 30)


def test_missing_and_duplicate_sections_explain_normal_prompt_format():
    source = prompt('[0-30s] Walk quietly.')
    for bad in (source.replace('retention_analysis:', 'other:'), source + '\nsummary:\nAgain.'):
        with pytest.raises(ValueError, match='字段|retention_analysis') as error:
            parser().contract_from_prompt(bad, 30)
        assert 'JSON' not in str(error.value)


def test_timestamp_like_spoken_text_is_literal_and_not_an_action_clock():
    detail = ('[0-3s] <Subject 1> (S1) says: <d>Read [16-30秒] and <Audio 99> exactly.</d>\n'
              '[3-30s] Walk quietly.')
    value = parser().contract_from_prompt(prompt(detail), 30, {'reference_image_1_file': 'one.png'})
    assert value['dialogues'][0]['text'] == 'Read [16-30秒] and <Audio 99> exactly.'
    assert sum(item['prompt'].count('Read [16-30秒] and <Audio 99> exactly.') for item in compile_all(value)) == 1
