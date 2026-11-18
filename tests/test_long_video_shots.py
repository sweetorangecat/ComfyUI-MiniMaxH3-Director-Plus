from copy import deepcopy
import json

import pytest

from nodes.long_video_shots import automatic_shots, build_shot_graph, plan_shots


def prompt(seconds=45, voice=True):
    voice_definition = '<Audio 1> supplies only her voice.' if voice else ''
    sections = {
        'subject_definitions': f'Mara <Subject 1> (S1) uses <Picture 1> for identity. {voice_definition}',
        'summary': 'A natural live-action office conversation under constant daylight.',
        'retention_analysis': 'Preserve original identity, clothes, daylight and relative scale 1:1.',
        'detailed_description': '\n'.join(
            f'[Shot {i + 1}]\n[{a}-{a+2}s] Eye-level medium view; hands rest on the desk.\n'
            f'[{a+2}-{a+10}s] <Subject 1> (S1) says: <d>第{i+1}句，先把事情做好。</d>\n'
            f'[{a+10}-{a+15}s] Lips closed, gaze and breathing remain natural.'
            for i, a in enumerate(range(0, seconds, 15))),
        'overall_soundscape': 'Quiet room tone. Only scheduled dialogue.',
        'non_diegetic_music': 'N/A',
    }
    return '\n\n'.join(f'{key}:\n{value}' for key, value in sections.items())


def graph(**changes):
    return {
        'director': {'class_type': 'MiniMaxH3DirectorPlus', 'inputs': {
            'duration': 45, 'prompt': prompt(), 'mode': 'REF2VA', 'voice_mode': 'h3_reference',
            'seed': 123, 'performance_preset': '均衡高质8步', 'sampling_method': 'standard',
            'postprocess_mode': 'vosr2', 'vosr2_profile': 'manual_baseline',
            'community_lora_profile': 'off', 'reference_image_1_file': 'original.png',
            'voice_reference_audio_file': 'original.wav', 'voice_reference_name_1': 'Mara',
            'resolution_preset': '1080p FHD', **changes}},
        'sampler': {'class_type': 'MiniMaxH3TwoStageSampler', 'inputs': {'guide': ['director', 0]}},
        'output': {'class_type': 'MiniMaxH3U11Output', 'inputs': {
            'images': ['sampler', 0], 'guide': ['director', 0], 'frame_rate': 24,
            'codec': 'H.264', 'container': 'MP4', 'bit_depth': '8-bit', 'quality': 18,
            'audio_codec': 'AAC', 'audio_bitrate': '128k', 'save_output': True}},
    }


def test_three_short_clips_use_authored_cuts_and_fresh_original_references():
    base = graph()
    before = deepcopy(base)
    active, config = automatic_shots(base, 'director', 45)
    assert base == before
    assert active['director']['inputs'] == before['director']['inputs']
    assert config['entry_mode'] == 'independent_shots'
    shots = config['shot_plan']
    assert [(s['new_start'], s['new_end']) for s in shots] == [(0, 360), (360, 720), (720, 1080)]
    for i, shot in enumerate(shots):
        assert shot['context_frames'] == shot['source_endpoint'] == shot['local_new_start'] == 0
        assert shot['local_new_end'] == 360
        assert '[2-10s]' in shot['prompt']
        assert f'第{i+1}句' in shot['prompt']
        assert sum(f'第{j}句' in shot['prompt'] for j in (1, 2, 3)) == 1
        assert '<Picture 1>' in shot['prompt'] and '<Audio 1>' in shot['prompt']


@pytest.mark.parametrize('seconds', [4, 15])
def test_short_requests_do_not_enter_independent_route(seconds):
    assert automatic_shots(graph(), 'director', seconds) is None


def test_single_continuous_shot_keeps_native_adapter():
    text = prompt().replace('[Shot 2]', '').replace('[Shot 3]', '')
    assert automatic_shots(graph(prompt=text), 'director', 45) is None


def test_small_authored_shots_pack_without_new_cuts_or_short_final_request():
    text = prompt(30).replace(
        '[Shot 1]\n[0-2s]', '[Shot 1]\n[0-2s]').replace('[2-10s]', '[Shot 2]\n[2-10s]')
    # A two-second first shot must stay inside a valid >=4-second request.
    shots = plan_shots(text, 30)
    assert [(s['new_start'], s['new_end']) for s in shots] == [(0, 360), (360, 720)]


def test_no_cut_is_invented_inside_an_overlong_authored_shot():
    text = prompt(30).replace('[Shot 2]\n[15-17s]', '[15-17s]')
    text = text.replace('[17-25s]', '[Shot 2]\n[17-25s]')
    with pytest.raises(ValueError, match='15|切镜'):
        plan_shots(text, 30)


def test_literal_shot_tags_and_times_inside_dialogue_do_not_change_the_plan():
    text = prompt().replace('第1句，先把事情做好。', '说出[Shot 9]和[70-80s]，不切镜。')
    shots = plan_shots(text, 45)
    assert len(shots) == 3
    assert '<d>说出[Shot 9]和[70-80s]，不切镜。</d>' in shots[0]['prompt']


def test_independent_requests_restart_shot_numbering_without_rewriting_dialogue():
    text = prompt().replace('第3句，先把事情做好。', '台词里保留[Shot 8]，先把事情做好。')
    shots = plan_shots(text, 45)

    for shot in shots:
        assert '[Shot 1]\n[0-2s]' in shot['prompt']
    assert '<d>台词里保留[Shot 8]，先把事情做好。</d>' in shots[2]['prompt']


def test_packed_internal_shots_keep_order_with_local_numbers_in_each_request():
    text = prompt(30).replace('[Shot 1]', '[Shot 10]').replace('[Shot 2]', '[Shot 20]')
    text = text.replace('[2-10s]', '[Shot 11]\n[2-10s] A deliberate cut to her hands. ')
    text = text.replace('[17-25s]', '[Shot 21]\n[17-25s] A deliberate cut to her hands. ')
    shots = plan_shots(text, 30)

    assert [(shot['new_start'], shot['new_end']) for shot in shots] == [(0, 360), (360, 720)]
    for shot in shots:
        assert '[Shot 1]\n[0-2s]' in shot['prompt']
        assert '[Shot 2]\n[2-10s] A deliberate cut to her hands.' in shot['prompt']


def test_implicit_opening_still_counts_as_the_first_local_shot():
    text = prompt(30).replace('[Shot 1]\n', '')
    text = text.replace('[2-10s]', '[Shot 7]\n[2-10s] A deliberate cut to her hands. ')
    shots = plan_shots(text, 30)

    assert '[Shot 2]\n[2-10s] A deliberate cut to her hands.' in shots[0]['prompt']
    assert '[Shot 1]\n[0-2s]' in shots[1]['prompt']


@pytest.mark.parametrize(('opening', 'expected'), [
    ('A deliberate cut back to the original wide-medium composition.',
     'The shot opens directly on the original wide-medium composition.'),
    ('Cut to a slightly tighter eye-level medium view.',
     'The shot opens directly on a slightly tighter eye-level medium view.'),
    ('A hard cut to the same office at eye level.',
     'The shot opens directly on the same office at eye level.'),
    ('Cut to <Subject 1> looking at the camera.',
     'The shot opens directly on <Subject 1> looking at the camera.'),
    ('镜头切回原来的办公室中景。', '开场即为原来的办公室中景。'),
    ('画面直接切到同一办公室的近景。', '开场即为同一办公室的近景。'),
])
def test_compiled_clip_starts_in_the_authored_view_after_an_external_cut(opening, expected):
    text = prompt(30).replace(
        '[15-17s] Eye-level medium view; hands rest on the desk.',
        '[15-17s] ' + opening)
    shots = plan_shots(text, 30)

    assert '[0-2s] ' + expected in shots[1]['prompt']
    assert '[0-2s] Eye-level medium view; hands rest on the desk.' in shots[0]['prompt']
    assert '<d>第2句，先把事情做好。</d>' in shots[1]['prompt']


def test_deliberate_internal_cuts_and_spoken_cut_words_keep_their_meaning():
    text = prompt(30).replace(
        '[15-17s] Eye-level medium view; hands rest on the desk.',
        '[15-17s] Cut to the original office medium view.')
    text = text.replace('[17-25s]', '[Shot 3]\n[17-25s] A deliberate cut to her hands. ')
    text = text.replace('第2句，先把事情做好。', 'Cut to the office，[Shot 9]是台词。')
    shots = plan_shots(text, 30)

    assert '[0-2s] The shot opens directly on the original office medium view.' in shots[1]['prompt']
    assert '[2-10s] A deliberate cut to her hands.' in shots[1]['prompt']
    assert '<d>Cut to the office，[Shot 9]是台词。</d>' in shots[1]['prompt']


@pytest.mark.parametrize('voice_mode', ['none', 'h3_reference'])
@pytest.mark.parametrize(('preset', 'method'), [
    ('质量优先加速', 'standard'),
    ('质量优先二采样', 'selflift'),
    ('低显存', 'standard'),
])
def test_original_graph_and_sampler_modes_are_not_replaced_by_native_selflift(voice_mode, preset, method):
    base = graph(voice_mode=voice_mode, performance_preset=preset, sampling_method=method)
    base.update({
        'model': {'class_type': 'UNETLoader', 'inputs': {'unet_name': 'original-model.safetensors', 'weight_dtype': 'default'}},
        'lora': {'class_type': 'LoraLoaderModelOnly', 'inputs': {
            'model': ['model', 0], 'lora_name': 'original-detail.safetensors', 'strength_model': 0.35}},
        'guider': {'class_type': 'BasicGuider', 'inputs': {'model': ['lora', 0], 'conditioning': ['director', 1]}},
        'noise': {'class_type': 'RandomNoise', 'inputs': {'noise_seed': 123}},
    })
    base['sampler']['inputs'].update(guider=['guider', 0], noise=['noise', 0])
    original = deepcopy(base)
    active, config = automatic_shots(base, 'director', 45)
    spec = {'project_name': config['project_name'], 'take_id': 'testtake',
            'window': config['shot_plan'][1], 'prompt': config['shot_plan'][1]['prompt'], 'seed': 777}
    compiled = build_shot_graph(active, spec)
    assert base == original
    inputs = compiled['director']['inputs']
    for key in ('mode', 'performance_preset', 'sampling_method', 'postprocess_mode', 'vosr2_profile',
                'community_lora_profile', 'reference_image_1_file'):
        assert inputs[key] == original['director']['inputs'][key]
    for key in ('model', 'lora', 'guider', 'sampler'):
        assert compiled[key] == original[key]
    assert inputs.get('voice_reference_audio_file') == active['director']['inputs'].get('voice_reference_audio_file')
    assert compiled['noise']['inputs']['noise_seed'] == 777
    assert set(compiled) == set(original)
    assert inputs['duration'] == 15 and inputs['seed'] == 777
    assert not any(node['class_type'].startswith(('MiniMaxH3LongVideoContext', 'MiniMaxH3LongVideoParent'))
                   for node in compiled.values())
    assert compiled['output']['class_type'] == 'MiniMaxH3LongVideoShotOutput'
    assert json.loads(compiled['output']['inputs']['request_json']) == spec


def test_nine_original_pictures_need_no_reserved_parent_slot():
    base = graph(**{f'reference_image_{i}_file': f'original{i}.png' for i in range(1, 8)},
                 first_image_file='first.png', last_image_file='last.png')
    _, config = automatic_shots(base, 'director', 45)
    assert len(config['shot_plan']) == 3
    assert 'parent_reference' not in config


def test_no_clone_drops_stale_audio_without_changing_generated_dialogue():
    base = graph(voice_mode='none')
    active, config = automatic_shots(base, 'director', 45)
    assert 'voice_reference_audio_file' not in active['director']['inputs']
    assert all('<Audio ' not in s['prompt'] and '<d>' in s['prompt'] for s in config['shot_plan'])


def test_global_hard_endpoints_are_not_silently_reinterpreted():
    with pytest.raises(ValueError, match='REF2VA'):
        automatic_shots(graph(mode='FL2VA'), 'director', 45)


@pytest.mark.parametrize('setting', [{'frame_rate': 30}, {'pingpong': True}, {'crop_to_audio': True},
                                    {'save_output': False}])
def test_invalid_short_clip_timing_fails_before_queue_work(setting):
    base = graph()
    base['output']['inputs'].update(setting)
    with pytest.raises(ValueError):
        automatic_shots(base, 'director', 45)
