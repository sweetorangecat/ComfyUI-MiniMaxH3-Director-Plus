"""CPU-only complete-audio contracts; generated speech is never media approval."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
import wave

import pytest
import torch

from nodes.long_video_plan import audio_ticks, plan_windows


def audio_module():
    try:
        return importlib.import_module('nodes.long_video_audio')
    except ModuleNotFoundError:
        pytest.fail('Native complete target-audio contract is missing')


def wav(path, samples, value=100, rate=32000, channels=2):
    with wave.open(str(path), 'wb') as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(int(value).to_bytes(2, 'little', signed=True) * samples * channels)
    return path


def line(**changes):
    result = {'id': 'whole-line', 'start_frame': 48, 'end_frame': 96,
              'speaker': '<Subject 7> (S2)', 'audio_slot': 1, 'text': 'The whole line.', 'language': 'en'}
    result.update(changes)
    return result


def backend():
    return {'backend': 'FishS2VoiceCloneTTS', 'requested_model': 's2-pro (auto download)',
            'resolved_model': {'name': 's2-pro', 'weights': [{'sha256': 'a' * 64, 'bytes': 42}]},
            'code': {'voice_clone_node.py': 'b' * 64},
            'parameters': {'temperature': .8, 'top_p': .8, 'language': 'en'}}


def sentence(tmp_path):
    audio = audio_module()
    source = audio.file_identity(wav(tmp_path / 'original.wav', 8000))
    return audio.sentence_spec(line(), source, 'Actual reference words.', backend(), 71)


def target_recipe(tmp_path, window=None):
    audio = audio_module()
    window = window or plan_windows(720)[0]
    source = audio.file_identity(wav(tmp_path / 'original.wav', 8000))
    target = audio.file_identity(wav(tmp_path / 'full.wav', audio_ticks(window['sample_frames']) * 800))
    recipe = {'kind': 'complete_wav', **target, 'window': window,
              'sample_rate': 32000, 'channels': 2, 'samples': audio_ticks(window['sample_frames']) * 800,
              'sources': [source], 'provenance': {'method': 'supplied complete native soundscape', 'seed': 71},
              'utterances': [{**line(), 'start_sample': 64000, 'end_sample': 96000,
                              'source_sha256': source['sha256'], 'original_reference_sha256': source['sha256'],
                              'generation': backend()}]}
    return recipe, {1: source}


def test_frame_clock_is_rational_and_native_suffix_is_explicit():
    audio = audio_module()
    assert audio.frame_sample(1) == 1333
    assert audio.frame_sample(2) == 2667
    assert audio.frame_sample(294) == 392000
    assert audio.physical_samples(plan_windows(720)[0]) == 547 * 800
    assert audio.frame_sample(328) == 437333  # Known 267-sample encoder suffix.


def test_wave_geometry_checks_payload_in_bounded_chunks(tmp_path, monkeypatch):
    audio = audio_module()
    path = wav(tmp_path / 'chunked.wav', 100000)
    original = wave.open
    reads = []
    def opened(*args, **kwargs):
        stream = original(*args, **kwargs)
        read = stream.readframes
        def bounded(count):
            reads.append(count)
            return read(count)
        stream.readframes = bounded
        return stream
    monkeypatch.setattr(audio.wave, 'open', opened)
    assert audio.wave_geometry(path)['samples'] == 100000
    assert max(reads) <= 8192


def test_complete_target_validates_actual_waveform_and_original_identity(tmp_path):
    audio = audio_module()
    recipe, sources = target_recipe(tmp_path)
    result = audio.validate_target_audio(recipe, recipe['window'], [line()], sources)
    assert result['sha256'] == recipe['sha256']
    assert result['samples'] == 437600 and result['channels'] == 2
    assert result['method'] == 'target_audio_conditioned_native_video'
    assert result['content_review'] == 'pending'


@pytest.mark.parametrize('mutation,match', [
    (lambda recipe: recipe.update(samples=1), 'samples|geometry'),
    (lambda recipe: recipe.update(sha256='0' * 64), 'hash|SHA'),
    (lambda recipe: recipe.update(provenance={}), 'provenance|method'),
    (lambda recipe: recipe['utterances'][0].update(speaker='<Subject 2> (S7)'), 'binding|speaker'),
    (lambda recipe: recipe['utterances'][0].update(text='Shortened.'), 'words|text|utterance'),
    (lambda recipe: recipe['utterances'][0].update(end_sample=128001), 'deadline|interval'),
    (lambda recipe: recipe['sources'][0].update(sha256='f' * 64), 'hash|SHA'),
    (lambda recipe: recipe['utterances'][0].update(original_reference_sha256='d' * 64), 'binding|reference|SHA'),
])
def test_supplied_target_cannot_override_geometry_bindings_or_actual_sources(tmp_path, mutation, match):
    audio = audio_module()
    recipe, sources = target_recipe(tmp_path)
    sources = deepcopy(sources)
    mutation(recipe)
    with pytest.raises(ValueError, match=match):
        audio.validate_target_audio(recipe, recipe['window'], [line()], sources)


def test_source_replaced_in_place_is_not_the_same_reference(tmp_path):
    audio = audio_module()
    recipe, sources = target_recipe(tmp_path)
    wav(tmp_path / 'original.wav', 8000, value=101)
    with pytest.raises(ValueError, match='hash|SHA'):
        audio.validate_target_audio(recipe, recipe['window'], [line()], sources)


def test_whole_utterance_must_fit_deadline_and_delivered_interval():
    audio = audio_module()
    first = plan_windows(720)[0]
    assert audio.place_utterance(first, line(), 64000) == (64000, 128000)
    with pytest.raises(ValueError, match='whole|deadline'):
        audio.place_utterance(first, line(), 64001)
    with pytest.raises(ValueError, match='delivered|interval'):
        audio.place_utterance(first, line(start_frame=292, end_frame=295), 100)
    with pytest.raises(ValueError, match='history|interval'):
        audio.place_utterance(plan_windows(720)[1], line(start_frame=293, end_frame=300), 100)


@pytest.mark.parametrize('sample_rate', [32000, 44100])
def test_forward_utterance_starts_after_actual_protected_audio_ticks_without_shifting(sample_rate):
    from nodes.long_video_av import conditioning_geometry
    audio = audio_module()
    window = plan_windows(720)[1]
    geometry = conditioning_geometry(window, 34)
    assert geometry['audio_ticks'] == 122
    assert geometry['audio_ticks'] * 800 == 97600
    rounded_frame_start = line(start_frame=window['new_start'] + 34, end_frame=window['new_start'] + 60)
    assert audio.frame_sample(39 + 34) == 97333
    with pytest.raises(ValueError, match='protected.*audio|audio.*protected'):
        audio.place_utterance(window, rounded_frame_start, 800, sample_rate, forward_frames=34)
    safe = line(start_frame=window['new_start'] + 35, end_frame=window['new_start'] + 60)
    start = audio.frame_sample(39 + 35, sample_rate)
    assert audio.place_utterance(window, safe, 800, sample_rate, forward_frames=34) == (start, start + 800)
    zero_forward = line(start_frame=window['new_start'], end_frame=window['new_start'] + 24)
    start = audio.frame_sample(39, sample_rate)
    assert audio.place_utterance(window, zero_forward, 800, sample_rate) == (start, start + 800)


def test_sentence_identity_ignores_exports_retries_and_placement_but_binds_exact_inputs(tmp_path):
    audio = audio_module()
    spec = sentence(tmp_path)
    key = audio.sentence_identity(spec)
    unrelated = dict(spec, project='elsewhere', width=3840, codec='h265', video_retry=2, window_index=9)
    assert audio.sentence_identity(unrelated) == key
    for field, value in [('text', 'Changed.'), ('language', 'zh'), ('reference_transcript', 'Different.'),
                         ('speaker', '<Subject 7> (S3)'), ('seed', 72), ('source', dict(spec['source'], sha256='c' * 64)),
                         ('backend', dict(spec['backend'], requested_model='other'))]:
        assert audio.sentence_identity(dict(spec, **{field: value})) != key


def test_unverified_fish_model_or_code_identity_cannot_be_called_verified(tmp_path):
    audio = audio_module()
    spec = sentence(tmp_path)
    for field in ['resolved_model', 'code']:
        changed = deepcopy(spec['backend'])
        changed[field] = {}
        with pytest.raises(ValueError, match='model|code|identit'):
            audio.sentence_spec(line(), spec['source'], '', changed, 71)


def test_cache_publishes_immutable_actual_whole_waveform_and_detects_replacement(tmp_path):
    audio = audio_module()
    spec = sentence(tmp_path)
    cache = audio.AudioCache(tmp_path)
    waveform = {'waveform': torch.full((1, 2, 32000), .1), 'sample_rate': 32000}
    first = cache.publish_sentence(spec, waveform)
    second = cache.publish_sentence(spec, {'waveform': torch.full((1, 2, 32000), .2), 'sample_rate': 32000})
    assert first == second and first['samples'] == 32000
    assert first['seed'] == 71 and first['content_review'] == 'pending'
    wav(first['path'], 32000, value=44)
    with pytest.raises(ValueError, match='hash|SHA'):
        cache.sentence(spec)


def test_cache_rejects_nonfinite_or_nonwhole_waveforms(tmp_path):
    audio = audio_module()
    spec = sentence(tmp_path)
    cache = audio.AudioCache(tmp_path)
    for shape in [(2, 2, 5), (1, 3, 5), (1, 2, 0)]:
        with pytest.raises(ValueError, match='waveform|batch|channel'):
            cache.publish_sentence(spec, {'waveform': torch.zeros(shape), 'sample_rate': 32000})
    with pytest.raises(ValueError, match='finite'):
        cache.publish_sentence(spec, {'waveform': torch.full((1, 2, 5), float('nan')), 'sample_rate': 32000})


def contract(**changes):
    result = {'dialogues': [line()], 'overall_soundscape': 'Intentional silence except scheduled dialogue.',
              'non_diegetic_music': 'N/A', 'beats': []}
    result.update(changes)
    return result


def phased_contract(**changes):
    return contract(schema_version=1, fps=24, total_frames=720,
        subject_definitions='<Subject 7> (S2) uses <Audio 1> for timbre.', summary='One continuous scene.',
        retention_analysis='Keep the same identity.', scale_contract='Adult 170 cm; cup 10 cm.',
        beats=[{'id': 'hold', 'start_frame': 0, 'end_frame': 720, 'action': 'Hold the cup still.'}], **changes)


@pytest.mark.parametrize('override,requested_sound', [
    ({'sections': {'overall_soundscape': 'Room tone and audible footsteps.'}}, 'Room tone and audible footsteps.'),
    ({'sections': {'non_diegetic_music': 'Soft piano.'}}, 'Soft piano.'),
    ({'remaining_beats': [{'id': 'hold', 'action': 'Hold the cup still.', 'sound': 'Cup taps the table.'}]}, 'Cup taps the table.'),
    ({'remaining_beats': [{'id': 'hold', 'action': 'Hold the cup still.', 'timed_actions': [
        {'start_frame': 0, 'end_frame': 294, 'action': 'Hold the cup still.', 'sound': 'Faint fabric rustling.'}]}]}, 'Faint fabric rustling.'),
])
def test_silent_fish_rejects_effective_sound_requirements_present_in_compiled_window(override, requested_sound):
    from nodes.long_video_plan import compile_window
    window = plan_windows(720)[0]
    story = phased_contract(window_overrides=[{'new_start': 0, 'new_end': window['new_end'], **override}])
    assert requested_sound in compile_window(story, window)['prompt']
    with pytest.raises(ValueError, match='soundscape|background|music'):
        audio_module().recipe_for_window(story, window, {'voice_mode': 'fish_lock'})


def test_audio_policy_honors_replaced_sections_and_remaining_beat_precedence_per_window():
    from nodes.long_video_plan import compile_window
    first, second, *_ = plan_windows(720)
    story = phased_contract()
    story.update(overall_soundscape='Room tone.', non_diegetic_music='Piano.')
    story['beats'][0]['sound'] = 'Footsteps.'
    story['window_overrides'] = [{'new_start': 0, 'new_end': first['new_end'],
        'sections': {'overall_soundscape': 'Intentional silence except scheduled dialogue.', 'non_diegetic_music': 'N/A'},
        'remaining_beats': [{'id': 'hold', 'action': 'Hold the cup still.'}]}]
    prompt = compile_window(story, first)['prompt']
    assert all(sound not in prompt for sound in ('Room tone.', 'Piano.', 'Footsteps.'))
    assert audio_module().recipe_for_window(story, first, {'voice_mode': 'fish_lock'})['background']['kind'] == 'silence'
    assert 'Room tone.' in compile_window(story, second, 'Actual current pose.')['prompt']
    with pytest.raises(ValueError, match='soundscape|background|music'):
        audio_module().recipe_for_window(story, second, {'voice_mode': 'fish_lock'})


def test_audio_policy_uses_only_current_timed_sounds_and_explicit_timed_overrides():
    from nodes.long_video_plan import compile_window
    first, second, *_ = plan_windows(720)
    story = phased_contract()
    story['beats'][0].update(sound='Cup tap.', timed_actions=[
        {'start_frame': 0, 'end_frame': 294, 'action': 'Hold the cup still.', 'sound': 'Silence'},
        {'start_frame': 294, 'end_frame': 720, 'action': 'Hold the cup still.'}])
    assert 'Cup tap.' not in compile_window(story, first)['prompt']
    assert audio_module().recipe_for_window(story, first, {'voice_mode': 'fish_lock'})['background']['kind'] == 'silence'
    assert 'Cup tap.' in compile_window(story, second, 'Actual current pose.')['prompt']
    with pytest.raises(ValueError, match='soundscape|background'):
        audio_module().recipe_for_window(story, second, {'voice_mode': 'fish_lock'})


def test_fish_auto_derives_whole_words_only_for_explicit_original_audio1_binding():
    audio = audio_module()
    settings = {'voice_mode': 'fish_lock', 'target_dialogue': '', 'reference_transcript': 'Source words.'}
    result = audio.recipe_for_window(contract(), plan_windows(720)[0], settings)
    assert result['kind'] == 'fish' and result['background'] == {'kind': 'silence', 'intentional': True}
    assert result['target_dialogue'] == line()['text']
    assert result['reference_transcript'] == 'Source words.'


@pytest.mark.parametrize('story,settings,match', [
    (contract(overall_soundscape='Room tone and footsteps.'), {'voice_mode': 'fish_lock'}, 'soundscape|background'),
    (contract(overall_soundscape='Intentional silence except scheduled dialogue. Keep quiet room tone.'),
     {'voice_mode': 'fish_lock'}, 'soundscape|background'),
    (contract(non_diegetic_music='Quiet piano.'), {'voice_mode': 'fish_lock'}, 'music|soundscape'),
    (contract(beats=[{'sound': 'Cup taps the table.'}]), {'voice_mode': 'fish_lock'}, 'soundscape|background'),
    (contract(dialogues=[line(audio_slot=None)]), {'voice_mode': 'fish_lock'}, 'Audio1|speaker'),
    (contract(dialogues=[line(), line(id='ordinary', audio_slot=2)]), {'voice_mode': 'fish_lock'}, 'Audio1|speaker'),
    (contract(), {'voice_mode': 'fish_lock', 'target_dialogue': 'A different line.'}, 'target_dialogue|words'),
])
def test_fish_auto_never_silently_drops_sound_or_assigns_audio1_to_ordinary_speakers(story, settings, match):
    with pytest.raises(ValueError, match=match):
        audio_module().recipe_for_window(story, plan_windows(720)[0], settings)


def test_default_joint_av_routes_are_inert_and_explicit_complete_wav_is_authoritative(tmp_path):
    audio = audio_module()
    assert audio.recipe_for_window(contract(), plan_windows(720)[0], {'voice_mode': 'h3_reference'}) is None
    recipe, _ = target_recipe(tmp_path)
    story = contract(overall_soundscape='Room tone and complete supplied dialogue.', target_audio=recipe)
    assert audio.recipe_for_window(story, recipe['window'], {'voice_mode': 'none'}) == recipe


def test_supplied_complete_wav_still_checks_active_original_fish_target_dialogue(tmp_path):
    audio = audio_module()
    recipe, _ = target_recipe(tmp_path)
    story = contract(target_audio=recipe)
    with pytest.raises(ValueError, match='target_dialogue|words'):
        audio.recipe_for_window(story, recipe['window'], {'voice_mode': 'fish_lock', 'target_dialogue': 'Conflicting words.'})
    for target in ['', line()['text']]:
        assert audio.recipe_for_window(story, recipe['window'], {'voice_mode': 'fish_lock', 'target_dialogue': target}) == recipe


@pytest.mark.parametrize('voice_mode', ['none', 'h3_reference'])
@pytest.mark.parametrize('kind', ['complete_wav', 'fish'])
def test_inactive_original_target_dialogue_does_not_constrain_explicit_audio_recipe(tmp_path, voice_mode, kind):
    audio = audio_module()
    complete, _ = target_recipe(tmp_path)
    recipe = complete if kind == 'complete_wav' else {'kind': 'fish', 'background': {'kind': 'silence', 'intentional': True}}
    story = contract(target_audio=recipe)
    assert audio.recipe_for_window(story, complete['window'], {'voice_mode': voice_mode, 'target_dialogue': 'Stale inactive words.'})['kind'] == kind


@pytest.mark.parametrize('scope', ['root', 'window'])
@pytest.mark.parametrize('kind', ['complete_wav', 'fish'])
def test_published_schema_accepts_actual_root_and_window_audio_recipes_without_approval(tmp_path, scope, kind):
    import jsonschema
    from nodes.long_video_plan import validate_contract, compile_window
    audio = audio_module()
    complete, originals = target_recipe(tmp_path)
    recipe = complete if kind == 'complete_wav' else {'kind': 'fish', 'background': {'kind': 'silence', 'intentional': True}}
    story = phased_contract(approved=False, mounted=False)
    if scope == 'root':
        story['target_audio'] = recipe
    else:
        story['window_overrides'] = [{'new_start': 0, 'new_end': complete['window']['new_end'], 'target_audio': recipe}]
    schema = json.loads((Path(__file__).parents[1] / 'templates/u11_long_video_contract.schema.json').read_text(encoding='utf-8'))
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(story, schema)
    validated = validate_contract(story, 720)
    assert validated['approved'] is False and validated['mounted'] is False
    assert compile_window(validated, complete['window'])['target_audio_recipe'] == recipe
    selected = audio.recipe_for_window(validated, complete['window'], {'voice_mode': 'fish_lock'})
    if kind == 'complete_wav':
        assert audio.validate_target_audio(selected, complete['window'], [line()], originals)['content_review'] == 'pending'
    else:
        assert selected['background'] == {'kind': 'silence', 'intentional': True}


@pytest.mark.parametrize('scope', ['root', 'window'])
@pytest.mark.parametrize('invalid', ['unsupported_kind', 'missing_hash'])
def test_published_schema_and_runtime_both_reject_invalid_audio_recipes(tmp_path, scope, invalid):
    import jsonschema
    audio = audio_module()
    recipe, originals = target_recipe(tmp_path)
    if invalid == 'unsupported_kind':
        recipe['kind'] = 'post_render_mux'
    else:
        recipe.pop('sha256')
    story = phased_contract()
    if scope == 'root':
        story['target_audio'] = recipe
    else:
        story['window_overrides'] = [{'new_start': 0, 'new_end': recipe['window']['new_end'], 'target_audio': recipe}]
    schema = json.loads((Path(__file__).parents[1] / 'templates/u11_long_video_contract.schema.json').read_text(encoding='utf-8'))
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(story, schema)
    with pytest.raises(ValueError):
        selected = audio.recipe_for_window(story, recipe['window'], {'voice_mode': 'fish_lock'})
        audio.validate_target_audio(selected, recipe['window'], [line()], originals)


def test_fish_resolver_requires_installed_weights_codec_and_real_backend_code_without_loading(tmp_path):
    from nodes.long_video_assets import freeze_fish_backend
    import types
    model = tmp_path / 'models' / 'fishaudioS2' / 's2-pro'
    code = tmp_path / 'plugin' / 'nodes' / 'voice_clone_node.py'
    code.parent.mkdir(parents=True)
    code.write_text('# backend code', encoding='utf-8')
    speech = tmp_path / 'fish_speech'
    speech.mkdir()
    (speech / '__init__.py').write_text('# inference code', encoding='utf-8')
    registry = types.SimpleNamespace(models_dir=str(tmp_path / 'models'))
    with pytest.raises(ValueError, match='installed|model|weights'):
        freeze_fish_backend('s2-pro (auto download)', registry, node_source=code, speech_source=speech)
    assert not model.exists()  # Resolver did not invoke the backend's auto downloader.
    model.mkdir(parents=True)
    (model / 'model.safetensors').write_bytes(b'weights')
    (model / 'config.json').write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='codec|decoder'):
        freeze_fish_backend('s2-pro', registry, node_source=code, speech_source=speech)
    (model.parent / 'codec.pth').write_bytes(b'shared actual codec weights')
    result = freeze_fish_backend('s2-pro (auto download)', registry, node_source=code, speech_source=speech)
    assert result['requested_model'] == 's2-pro (auto download)'
    assert result['resolved_model']['name'] == 's2-pro'
    assert len(result['resolved_model']['weights']) >= 3
    assert result['code'] and result['code_files']
    assert result['parameters']['device'] == 'auto'
    assert result['parameters']['keep_model_loaded'] is False
