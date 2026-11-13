"""Real coordinator/node/receipt integration with only the Fish inference seam stubbed."""
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import types
import wave

import pytest
import torch

from nodes.long_video_plan import plan_windows
from nodes.long_video_service import Coordinator, configure_project
from nodes.long_video_store import ProjectStore
import nodes.long_video_audio as audio


class Backend:
    def __init__(self):
        self.submitted, self.jobs = [], {}

    def submit(self, prompt_id, graph):
        self.submitted.append((prompt_id, graph))
        self.jobs[prompt_id] = 'queued'

    def status(self, prompt_id):
        return self.jobs.get(prompt_id, 'missing')


def graph():
    base = json.loads((Path(__file__).parents[1] / 'templates/u11_api.json').read_text(encoding='utf-8'))
    base['10']['inputs'].update(mode='REF2VA', sampling_method='selflift', performance_preset='quality_two_stage',
        postprocess_mode='vosr2', vosr2_profile='manual_baseline', seed=42, voice_mode='fish_lock',
        target_dialogue='', reference_transcript='The actual reference words.', fish_model_path='s2-pro',
        voice_reference_audio=['13', 0])
    return base


def config(name='test'):
    return {'enabled': True, 'project_name': name, 'target_seconds': 30, 'scene_contract': {
        'schema_version': 1, 'fps': 24, 'subject_definitions': '<Subject 7> (S2) uses <Audio 1> for timbre.',
        'summary': 'One continuous scene.', 'retention_analysis': 'Stable identities.',
        'overall_soundscape': 'Intentional silence except scheduled dialogue.', 'non_diegetic_music': 'N/A',
        'scale_contract': 'Adult 170 cm; cup 10 cm.',
        'beats': [{'id': 'standing', 'start_frame': 0, 'end_frame': 720, 'action': 'Stand naturally.'}],
        'dialogues': [{'id': 'whole', 'start_frame': 48, 'end_frame': 96,
                       'text': 'A complete sentence.', 'speaker': '<Subject 7> (S2)', 'audio_slot': 1, 'language': 'en'}]}}


def assets(tmp_path):
    source = tmp_path / 'reference.wav'
    audio.write_wave(source, {'waveform': torch.zeros((1, 2, 8000)), 'sample_rate': 32000})
    weights = tmp_path / 'weights.pth'
    weights.write_bytes(b'actual test weights; no model is imported')
    code = tmp_path / 'fish_node.py'
    code.write_text('# actual test backend code identity', encoding='utf-8')
    return {'original_sources': {1: audio.file_identity(source)}, 'fish_backend': {
        'backend': 'FishS2VoiceCloneTTS', 'requested_model': 's2-pro',
        'resolved_model': {'name': 's2-pro', 'path': str(tmp_path), 'weights': [audio.file_identity(weights)]},
        'code': {'fish_node.py': audio.file_identity(code)['sha256']}, 'code_files': [audio.file_identity(code)],
        'parameters': {'language': 'auto', 'device': 'auto', 'precision': 'auto', 'attention': 'auto',
                       'max_new_tokens': 0, 'chunk_length': 200, 'temperature': .8, 'top_p': .8,
                       'repetition_penalty': 1.1, 'keep_model_loaded': True, 'compile_model': False}}}


def configure(store, base, raw, identities):
    return configure_project(store, base, raw, asset_freezer=lambda _: {}, export_asset_freezer=lambda _: {},
        encoder_checker=lambda: {}, audio_asset_resolver=lambda *_: deepcopy(identities))


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'folder_paths', types.SimpleNamespace(get_output_directory=lambda: str(tmp_path)))
    identities = assets(tmp_path)
    store, backend = ProjectStore(tmp_path, 'test'), Backend()
    configure(store, graph(), config(), identities)
    calls = []
    def generate(spec):
        calls.append(deepcopy(spec))
        return {'waveform': torch.full((1, 2, 32000), .1), 'sample_rate': 32000}
    monkeypatch.setattr(audio, '_generate_fish', generate)
    return store, backend, Coordinator(store, backend), identities, calls


def run_audio_job(store, backend):
    state = store.read()
    job = state['active_job']
    assert job['stage'] == 'audio_prepare'
    graph = dict(backend.submitted)[job['prompt_id']]
    assert len(graph) == 1 and not any('Sampler' in node['class_type'] for node in graph.values())
    payload = graph['h3long_audio_prepare']['inputs']['request_json']
    audio.MiniMaxH3LongVideoAudioPrepare().prepare(payload)
    backend.jobs[job['prompt_id']] = 'completed'
    return json.loads(payload)


def prepare_native(pipeline):
    store, backend, runner, _, _ = pipeline
    assert runner.step()
    speech = run_audio_job(store, backend)
    assert speech['operation'] == 'sentence'
    assert runner.step()
    composed = run_audio_job(store, backend)
    assert composed['operation'] == 'compose'
    assert runner.step()
    return store.read()['windows'][0]


def test_preparation_is_queued_and_final_target_bytes_precede_native_request_and_fingerprint(pipeline):
    store, backend, runner, _, calls = pipeline
    assert runner.step()
    assert store.read()['windows'] == []
    first = run_audio_job(store, backend)
    assert len(calls) == 1 and calls[0]['text'] == 'A complete sentence.'
    assert runner.step()
    assert store.read()['windows'] == []
    run_audio_job(store, backend)
    assert runner.step()
    native = store.read()['windows'][0]
    audio.verify_file(native['target_audio'])
    assert native['target_audio']['sha256'] and native['target_audio']['method'] == audio.METHOD
    assert native['target_audio']['utterances'][0]['end_sample'] == 96000
    assert native['requested_voice']['voice_mode'] == 'fish_lock'
    assert store.path(f"audio/{first['preparation_key']}/receipt.json").is_file()
    assert any(node['class_type'] == 'VAEEncodeAudio' for node in backend.submitted[-1][1].values())
    assert len(calls) == 1


def test_stop_finishes_shared_sentence_receipt_then_resumes_without_duplicate_tts(pipeline):
    store, backend, runner, _, calls = pipeline
    runner.step()
    store.stop()
    assert runner.step()  # Existing queue work remains owned.
    run_audio_job(store, backend)
    assert runner.step() is False
    assert store.read()['status'] == 'stopped' and store.read()['windows'] == []
    store.resume()
    assert runner.step()
    assert run_audio_job(store, backend)['operation'] == 'compose'
    assert len(calls) == 1


def test_lost_preparation_jobs_reconcile_receipts_before_retry_and_preserve_immutable_requests(pipeline):
    store, backend, runner, _, calls = pipeline
    runner.step()
    old = deepcopy(store.read()['active_job'])
    run_audio_job(store, backend)
    backend.jobs.clear()  # Receipt durable; the queue history was lost.
    assert runner.step()
    assert run_audio_job(store, backend)['operation'] == 'compose'
    assert len(calls) == 1 and store.path(old['graph']['name']).is_file()


def test_missing_unfinished_sentence_retries_with_same_key_and_seed(pipeline):
    store, backend, runner, _, calls = pipeline
    runner.step()
    old = deepcopy(store.read()['active_job'])
    backend.jobs.clear()
    assert runner.step()
    new = store.read()['active_job']
    assert new['prompt_id'] != old['prompt_id'] and new['preparation_key'] == old['preparation_key']
    run_audio_job(store, backend)
    assert len(calls) == 1


def test_two_coordinators_and_two_projects_share_one_live_sentence_job(pipeline):
    store, backend, runner, identities, calls = pipeline
    assert runner.step()
    assert Coordinator(store, backend).step()
    other = ProjectStore(store.output_root, 'other')
    configure(other, graph(), config('other'), identities)
    second = Coordinator(other, backend)
    assert second.step()
    assert len(backend.submitted) == 1 and other.read()['windows'] == []
    run_audio_job(store, backend)
    assert second.step()
    assert run_audio_job(other, backend)['operation'] == 'compose'
    assert len(calls) == 1


def test_export_only_change_and_video_retry_reuse_complete_audio_and_sentence_seed(pipeline):
    store, backend, runner, identities, calls = pipeline
    first = prepare_native(pipeline)
    backend.jobs.clear()
    runner.reconcile_job()  # Simulate a lost native sampler before clean output.
    changed = graph()
    changed['10']['inputs']['resolution_preset'] = '4K UHD'
    changed['24']['inputs']['quality'] = 12
    configure(store, changed, config(), identities)
    assert runner.step()
    second = store.read()['windows'][0]
    assert second['take_id'] != first['take_id']
    assert second['target_audio'] == first['target_audio'] and len(calls) == 1
    assert all(node['class_type'] != 'MiniMaxH3LongVideoAudioPrepare' for node in backend.submitted[-1][1].values())


def test_actual_reference_replacement_changes_sentence_target_and_native_identity(pipeline):
    store, backend, runner, identities, calls = pipeline
    first = prepare_native(pipeline)
    backend.jobs.clear()
    runner.reconcile_job()
    path = Path(identities['original_sources'][1]['path'])
    with wave.open(str(path), 'wb') as sound:
        sound.setnchannels(2); sound.setsampwidth(2); sound.setframerate(32000)
        sound.writeframes(b'\1\0' * 16000)
    identities['original_sources'][1] = audio.file_identity(path)
    configure(store, graph(), config(), identities)
    assert runner.step()
    assert store.read()['active_job']['stage'] == 'audio_prepare'
    run_audio_job(store, backend)
    assert calls[-1]['source']['sha256'] != calls[0]['source']['sha256']
    assert calls[-1]['seed'] != calls[0]['seed']
    assert first['target_audio']['sources'][0]['sha256'] != calls[-1]['source']['sha256']


def test_sentence_key_uses_reference_bytes_across_project_paths(tmp_path):
    identities = assets(tmp_path)
    dialogue = config()['scene_contract']['dialogues'][0]
    one = audio.sentence_spec(dialogue, identities['original_sources'][1], '', identities['fish_backend'], 7)
    second_path = tmp_path / 'same-bytes.wav'
    second_path.write_bytes(Path(one['source']['path']).read_bytes())
    two = dict(one, source=audio.file_identity(second_path))
    assert audio.sentence_identity(one) == audio.sentence_identity(two)


def test_supplied_complete_window_is_queued_and_uses_its_actual_bytes_without_fish(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'folder_paths', types.SimpleNamespace(get_output_directory=lambda: str(tmp_path)))
    window = plan_windows(720)[0]
    complete = audio.write_wave(tmp_path / 'supplied.wav', {'waveform': torch.full((1, 2, audio.physical_samples(window)), .03), 'sample_rate': 32000})
    recipe = {'kind': 'complete_wav', **complete, 'window': window, 'sources': [], 'utterances': [],
              'provenance': {'method': 'Complete supplied soundscape.', 'seed': None}}
    raw = config()
    raw['scene_contract']['dialogues'] = []
    raw['scene_contract']['subject_definitions'] = '<Subject 7> (S2) adult.'
    raw['scene_contract']['overall_soundscape'] = 'Supplied complete ambience and all sound.'
    raw['scene_contract']['window_overrides'] = [{'new_start': 0, 'new_end': window['new_end'], 'target_audio': recipe}]
    base = graph()
    base['10']['inputs']['voice_mode'] = 'none'
    store, backend = ProjectStore(tmp_path, 'test'), Backend()
    configure(store, base, raw, {'original_sources': {}, 'fish_backend': None})
    monkeypatch.setattr(audio, '_generate_fish', lambda _: pytest.fail('Supplied complete audio invoked Fish'))
    runner = Coordinator(store, backend)
    assert runner.step() and store.read()['windows'] == []
    assert run_audio_job(store, backend)['operation'] == 'compose'
    assert runner.step()
    native = store.read()['windows'][0]
    assert native['target_audio']['sha256'] == complete['sha256']
    assert native['target_audio']['path'] != complete['path']
    assert native['target_audio']['content_review'] == 'pending'
    assert complete['path'] in store.read()['config']['audio_file_stats']


def test_supplied_waveform_mutated_during_real_copy_cannot_publish_target_or_native_request(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'folder_paths', types.SimpleNamespace(get_output_directory=lambda: str(tmp_path)))
    window = plan_windows(720)[0]
    complete = audio.write_wave(tmp_path / 'supplied.wav', {
        'waveform': torch.full((1, 2, audio.physical_samples(window)), .03), 'sample_rate': 32000})
    recipe = {'kind': 'complete_wav', **complete, 'window': window, 'sources': [], 'utterances': [],
              'provenance': {'method': 'Frozen supplied complete soundscape.', 'seed': None}}
    raw, base = config(), graph()
    raw['scene_contract'].update(dialogues=[], subject_definitions='<Subject 7> (S2) adult.',
        window_overrides=[{'new_start': 0, 'new_end': window['new_end'], 'target_audio': recipe}])
    base['10']['inputs']['voice_mode'] = 'none'
    store, backend = ProjectStore(tmp_path, 'test'), Backend()
    configure(store, base, raw, {'original_sources': {}})
    runner = Coordinator(store, backend)
    assert runner.step()
    original_copy = audio.shutil.copyfileobj
    copied = []
    def mutate_then_copy(source, destination, length):
        path = Path(source.name)
        before = path.stat()
        with path.open('r+b') as replacement:
            replacement.seek(-2, 2)
            replacement.write(b'\0\0')
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        assert (path.stat().st_size, path.stat().st_mtime_ns) == (before.st_size, before.st_mtime_ns)
        original_copy(source, destination, length)
        copied.append(True)
    monkeypatch.setattr(audio.shutil, 'copyfileobj', mutate_then_copy)
    with pytest.raises(ValueError, match='hash|SHA|changed'):
        run_audio_job(store, backend)
    assert copied and audio.file_identity(complete['path'])['sha256'] != complete['sha256']
    assert store.read()['windows'] == []
    assert not list(store.root.glob('audio/*/receipt.json'))
    backend.jobs[store.read()['active_job']['prompt_id']] = 'failed'
    assert runner.step() is False
    assert store.read()['windows'] == []


def test_whole_fish_overflow_preserves_unapproved_whole_sentence_without_native_request(pipeline, monkeypatch):
    store, backend, runner, _, _ = pipeline
    monkeypatch.setattr(audio, '_generate_fish', lambda _: {'waveform': torch.ones((1, 2, 64001)) * .1, 'sample_rate': 32000})
    assert runner.step()
    with pytest.raises(ValueError, match='whole|deadline'):
        run_audio_job(store, backend)
    assert store.read()['windows'] == []
    receipts = list(audio.AudioCache(store.output_root).store.root.rglob('receipt.json'))
    assert len(receipts) == 1
    receipt = json.loads(receipts[0].read_text(encoding='utf-8'))
    assert receipt['samples'] == 64001 and receipt['content_review'] == 'pending'


def test_preparation_statistics_cover_sources_from_every_window(tmp_path):
    identities = assets(tmp_path)
    raw = config()
    raw['scene_contract']['dialogues'].append({**raw['scene_contract']['dialogues'][0], 'id': 'later', 'start_frame': 336, 'end_frame': 384})
    ordinary = audio.write_wave(tmp_path / 'ordinary.wav', {'waveform': torch.full((1, 2, 32000), .01), 'sample_rate': 32000})
    raw['scene_contract']['dialogues'][0].pop('audio_slot')
    raw['scene_contract']['target_audio'] = {'kind': 'fish', 'background': {'kind': 'silence', 'intentional': True},
        'speech_sources': {'whole': {**ordinary, 'text': 'A complete sentence.', 'speaker': '<Subject 7> (S2)',
                                     'provenance': {'method': 'Explicit ordinary speaker recording.'}}}}
    store = ProjectStore(tmp_path, 'test')
    configure(store, graph(), raw, identities)
    assert ordinary['path'] in store.read()['config']['audio_file_stats']


def test_window_specific_ordinary_speech_recipe_does_not_require_other_windows_speakers(tmp_path):
    identities = assets(tmp_path)
    raw = config()['scene_contract']
    first, second, *_ = plan_windows(720)
    ordinary = {**raw['dialogues'][0], 'id': 'later', 'start_frame': 336, 'end_frame': 384, 'audio_slot': None}
    raw['dialogues'].append(ordinary)
    raw['window_overrides'] = [{'new_start': second['new_start'], 'new_end': second['new_end'], 'target_audio': {
        'kind': 'fish', 'background': {'kind': 'silence', 'intentional': True}, 'speech_sources': {'later': {'path': 'future'}}}}]
    assert audio.recipe_for_window(raw, first, {'voice_mode': 'fish_lock'})['kind'] == 'fish'


def test_config_guards_distinct_complete_waveforms_from_all_windows(tmp_path):
    raw, base = config(), graph()
    base['10']['inputs']['voice_mode'] = 'none'
    raw['scene_contract']['dialogues'] = []
    raw['scene_contract']['subject_definitions'] = '<Subject 7> (S2) adult.'
    sources = []
    overrides = []
    for window in plan_windows(720)[:2]:
        actual = audio.write_wave(tmp_path / f"window-{window['index']}.wav", {
            'waveform': torch.zeros((1, 2, audio.physical_samples(window))), 'sample_rate': 32000})
        sources.append(actual['path'])
        recipe = {'kind': 'complete_wav', **actual, 'window': window, 'sources': [], 'utterances': [],
                  'provenance': {'method': 'Explicit complete waveform.', 'seed': None}}
        overrides.append({'new_start': window['new_start'], 'new_end': window['new_end'], 'target_audio': recipe})
    raw['scene_contract']['window_overrides'] = overrides
    store = ProjectStore(tmp_path, 'test')
    configure(store, base, raw, {'original_sources': {}})
    assert all(path in store.read()['config']['audio_file_stats'] for path in sources)


def test_original_video_seed_remains_64_bit_while_whole_sentence_seed_fits_fish_input(pipeline):
    store, backend, runner, identities, _ = pipeline
    base = graph()
    original = 2**63 + 7
    base['10']['inputs']['seed'] = original
    configure(store, base, config(), identities)
    assert runner.step()
    payload = run_audio_job(store, backend)
    spec = json.loads(store.path(f"audio/{payload['preparation_key']}/request.json").read_text(encoding='utf-8'))
    sentence = spec['sentences']['whole']
    assert 0 <= sentence['seed'] <= 2**31 - 1
    assert sentence['director_seed'] == original
    assert runner.step()
    run_audio_job(store, backend)
    assert runner.step()
    assert store.read()['windows'][0]['seed'] == original


@pytest.mark.parametrize('requested_device,resolved_device,precision', [('auto', 'cuda', torch.bfloat16), ('cpu', 'cpu', torch.float32)])
def test_queued_fish_records_resolved_device_precision_and_releases_its_cache(tmp_path, monkeypatch, requested_device, resolved_device, precision):
    identities = assets(tmp_path)
    dialogue = config()['scene_contract']['dialogues'][0]
    backend = deepcopy(identities['fish_backend'])
    backend['parameters'].update(device=requested_device, keep_model_loaded=False)
    spec = audio.sentence_spec(dialogue, identities['original_sources'][1], 'The reference words.', backend, 7)
    calls, unloads = [], []
    class Fish:
        __module__ = 'test_installed_fish.nodes.voice_clone_node'
        def generate(self, **kwargs):
            calls.append(kwargs)
            return ({'waveform': torch.zeros((1, 2, 32000)), 'sample_rate': 32000},)
    monkeypatch.setattr('nodes.fish.fish_voice_clone_node', lambda: Fish())
    monkeypatch.setitem(sys.modules, 'comfy_extras.nodes_audio', types.SimpleNamespace(load=lambda _: (torch.zeros((2, 96000)), 32000)))
    monkeypatch.setitem(sys.modules, 'test_installed_fish.nodes.loader', types.SimpleNamespace(
        resolve_device=lambda _: (resolved_device, precision), resolve_precision=lambda *_: precision))
    monkeypatch.setitem(sys.modules, 'test_installed_fish.nodes.model_cache', types.SimpleNamespace(unload_engine=lambda: unloads.append(True)))
    monkeypatch.setitem(sys.modules, 'comfy.model_management', types.SimpleNamespace(unload_all_models=lambda: None, soft_empty_cache=lambda: None))
    result = audio._generate_fish(spec)
    assert calls[0]['device'] == resolved_device and calls[0]['precision'] == str(precision).removeprefix('torch.')
    assert calls[0]['reference_text'] == 'The reference words.'
    assert result['_h3_fish_execution']['device'] == resolved_device
    assert result['_h3_fish_execution']['precision'] == str(precision).removeprefix('torch.')
    assert unloads


@pytest.mark.parametrize('reference_samples,valid', [(32000, False), (64000, True), (480000, True), (640000, False)])
def test_original_audio1_duration_is_checked_before_fish_backend_call(tmp_path, monkeypatch, reference_samples, valid):
    identities = assets(tmp_path)
    original = audio.write_wave(tmp_path / 'duration-reference.wav', {
        'waveform': torch.zeros((1, 2, reference_samples)), 'sample_rate': 32000})
    backend = deepcopy(identities['fish_backend'])
    backend['parameters'].update(device='cpu', keep_model_loaded=False)
    spec = audio.sentence_spec(config()['scene_contract']['dialogues'][0], original, '', backend, 7)
    calls = []
    class Fish:
        __module__ = 'test_duration_fish.nodes.voice_clone_node'
        def generate(self, **kwargs):
            calls.append(kwargs)
            return ({'waveform': torch.zeros((1, 2, 32000)), 'sample_rate': 32000},)
    def load_original(path):
        # The real source is an all-zero PCM WAV. Only the Comfy decoder seam
        # is stubbed; return its actual frame count and sample rate.
        with wave.open(path, 'rb') as stream:
            return torch.zeros((stream.getnchannels(), stream.getnframes())), stream.getframerate()
    monkeypatch.setattr('nodes.fish.fish_voice_clone_node', lambda: Fish())
    monkeypatch.setitem(sys.modules, 'comfy_extras.nodes_audio', types.SimpleNamespace(load=load_original))
    monkeypatch.setitem(sys.modules, 'test_duration_fish.nodes.loader', types.SimpleNamespace(
        resolve_device=lambda _: ('cpu', torch.float32), resolve_precision=lambda *_: torch.float32))
    monkeypatch.setitem(sys.modules, 'test_duration_fish.nodes.model_cache', types.SimpleNamespace(unload_engine=lambda: None))
    if valid:
        audio._generate_fish(spec)
        assert len(calls) == 1 and calls[0]['reference_audio']['waveform'].shape[-1] == reference_samples
    else:
        with pytest.raises(ValueError, match='2.*15|reference.*duration'):
            audio._generate_fish(spec)
        assert calls == []


@pytest.mark.parametrize('keep_loaded', [False, True])
def test_long_fish_evicts_same_name_stale_engine_before_loading_frozen_weights(tmp_path, monkeypatch, keep_loaded):
    identities = assets(tmp_path)
    backend = deepcopy(identities['fish_backend'])
    backend['parameters'].update(device='cpu', keep_model_loaded=keep_loaded)
    spec = audio.sentence_spec(config()['scene_contract']['dialogues'][0], identities['original_sources'][1], '', backend, 7)
    current_sha = backend['resolved_model']['weights'][0]['sha256']
    matching_cache_key = ('s2-pro', 'cpu', 'float32', 'auto')
    cache = {'engine': {'weights_sha256': '0' * 64}, 'key': matching_cache_key}
    events, used = [], []
    def unload():
        events.append('fish_cache_unload')
        cache.update(engine=None, key=None)
    class Fish:
        __module__ = 'test_stale_fish.nodes.voice_clone_node'
        def generate(self, **kwargs):
            key = tuple(kwargs[field] for field in ('model_path', 'device', 'precision', 'attention'))
            assert key == matching_cache_key  # Installed cache keys omit weight bytes.
            if cache['engine'] is None or cache['key'] != key:
                events.append('fish_load_frozen_weights')
                cache.update(engine={'weights_sha256': current_sha}, key=key)
            else:
                events.append('fish_reuse_stale_engine')
            used.append(cache['engine']['weights_sha256'])
            events.append('fish_inference')
            return ({'waveform': torch.zeros((1, 2, 32000)), 'sample_rate': 32000},)
    monkeypatch.setattr('nodes.fish.fish_voice_clone_node', lambda: Fish())
    monkeypatch.setitem(sys.modules, 'comfy_extras.nodes_audio', types.SimpleNamespace(load=lambda _: (torch.zeros((2, 96000)), 32000)))
    monkeypatch.setitem(sys.modules, 'test_stale_fish.nodes.loader', types.SimpleNamespace(
        resolve_device=lambda _: ('cpu', torch.float32), resolve_precision=lambda *_: torch.float32))
    monkeypatch.setitem(sys.modules, 'test_stale_fish.nodes.model_cache', types.SimpleNamespace(unload_engine=unload))
    audio._generate_fish(spec)
    assert used == [current_sha]
    assert events.index('fish_cache_unload') < events.index('fish_load_frozen_weights') < events.index('fish_inference')
    assert cache['engine'] == ({'weights_sha256': current_sha} if keep_loaded else None)


@pytest.mark.parametrize('requested_device,resolved_device', [('auto', 'cuda'), ('cpu', 'cpu')])
@pytest.mark.parametrize('keep_loaded', [False, True])
def test_external_fish_cuda_handoff_releases_comfy_models_before_loading_and_cpu_keeps_them(tmp_path, monkeypatch,
                                                                                         requested_device, resolved_device, keep_loaded):
    identities = assets(tmp_path)
    backend = deepcopy(identities['fish_backend'])
    backend['parameters'].update(device=requested_device, keep_model_loaded=keep_loaded)
    spec = audio.sentence_spec(config()['scene_contract']['dialogues'][0], identities['original_sources'][1], '', backend, 7)
    events, comfy_models = [], ['unrelated_cached_h3_model']
    def unload_comfy():
        events.append('comfy_unload_models')
        comfy_models.clear()
    def empty_comfy_cache():
        assert comfy_models == []
        events.append('comfy_release_cache')
    class Fish:
        __module__ = 'test_handoff_fish.nodes.voice_clone_node'
        def generate(self, **kwargs):
            assert kwargs['device'] == resolved_device
            if resolved_device == 'cuda':
                assert comfy_models == [], 'External Fish loaded while Comfy models still occupy VRAM'
            else:
                assert comfy_models == ['unrelated_cached_h3_model']
            events.append('fish_external_model_load')
            events.append('fish_inference')
            return ({'waveform': torch.zeros((1, 2, 32000)), 'sample_rate': 32000},)
    monkeypatch.setattr('nodes.fish.fish_voice_clone_node', lambda: Fish())
    monkeypatch.setitem(sys.modules, 'comfy_extras.nodes_audio', types.SimpleNamespace(load=lambda _: (torch.zeros((2, 96000)), 32000)))
    monkeypatch.setitem(sys.modules, 'test_handoff_fish.nodes.loader', types.SimpleNamespace(
        resolve_device=lambda _: (resolved_device, torch.float32), resolve_precision=lambda *_: torch.float32))
    monkeypatch.setitem(sys.modules, 'test_handoff_fish.nodes.model_cache', types.SimpleNamespace(
        unload_engine=lambda: events.append('fish_cache_unload')))
    monkeypatch.setitem(sys.modules, 'comfy.model_management', types.SimpleNamespace(
        unload_all_models=unload_comfy, soft_empty_cache=empty_comfy_cache))
    audio._generate_fish(spec)
    assert events[0] == 'fish_cache_unload'
    if resolved_device == 'cuda':
        assert events.index('fish_cache_unload') < events.index('comfy_unload_models')
        assert events.index('comfy_unload_models') < events.index('comfy_release_cache') < events.index('fish_external_model_load')
    else:
        assert not any(event.startswith('comfy_') for event in events)
        assert comfy_models == ['unrelated_cached_h3_model']
    assert events[-1] == ('fish_inference' if keep_loaded else 'fish_cache_unload')


@pytest.mark.parametrize('lose_waveform', [False, True])
def test_clean_av_recovery_never_regenerates_target_preparation(pipeline, lose_waveform):
    store, backend, runner, _, calls = pipeline
    take = prepare_native(pipeline)
    clean = store.path(f"takes/{take['take_id']}/clean_video.safetensors")
    clean.write_bytes(b'clean AV lifecycle fixture')
    store.record_clean(take['take_id'], {'artifacts': {'clean_video': store.artifact(clean)}})
    backend.jobs[store.read()['active_job']['prompt_id']] = 'completed'
    store.path(f"audio/{take['audio_preparation_key']}/receipt.json").unlink()
    if lose_waveform:
        Path(take['target_audio']['path']).unlink()
        with pytest.raises(ValueError, match='missing|Missing|target|Target'):
            runner.step()
    else:
        assert runner.step()
        graph = backend.submitted[-1][1]
        assert any(node['class_type'] == 'MiniMaxH3LongVideoLoadAV' for node in graph.values())
        assert not any(node['class_type'] in {'MiniMaxH3LongVideoAudioPrepare', 'VAEEncodeAudio'} for node in graph.values())
        assert store.path(f"audio/{take['audio_preparation_key']}/receipt.json").is_file()
    assert len(calls) == 1
