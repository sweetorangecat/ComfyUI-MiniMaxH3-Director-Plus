"""Complete native audio targets and content-addressed whole-utterance evidence.

This module loads no models. Waveform preparation runs in ordinary Comfy jobs;
these contracts grant neither audible nor visual acceptance.
"""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import shutil
import uuid
import wave

from .long_video_plan import FPS, audio_ticks, validate_window

SAMPLE_RATE = 32000
SAMPLES_PER_TICK = 800
METHOD = 'target_audio_conditioned_native_video'
CACHE_PROJECT = '_shared_native_target_audio_cache'


def _key(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def frame_sample(frame, sample_rate=SAMPLE_RATE):
    if type(frame) is not int or type(sample_rate) is not int or sample_rate <= 0:
        raise ValueError('Audio clock requires integer frames and positive sample rate')
    return round(Fraction(frame * sample_rate, FPS))


def physical_samples(window):
    return audio_ticks(validate_window(window)['sample_frames']) * SAMPLES_PER_TICK


def file_identity(path):
    from .longfilm_media import sha256_file
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError('Audio source is missing: ' + str(path))
    return {'path': str(path), 'sha256': sha256_file(path), 'bytes': path.stat().st_size}


def verify_file(identity):
    if not isinstance(identity, dict) or not isinstance(identity.get('path'), str):
        raise ValueError('Audio source requires actual path and SHA-256 identity')
    actual = file_identity(identity['path'])
    if any(actual[key] != identity.get(key) for key in ('sha256', 'bytes')):
        raise ValueError('Audio file hash/SHA-256 differs from its immutable identity')
    return actual


def wave_geometry(path):
    try:
        with wave.open(str(path), 'rb') as stream:
            if stream.getcomptype() != 'NONE' or stream.getsampwidth() != 2:
                raise ValueError('Prepared audio requires uncompressed PCM16 WAV')
            result = {'sample_rate': stream.getframerate(), 'channels': stream.getnchannels(),
                      'samples': stream.getnframes()}
            remaining = result['samples']
            while remaining:
                count = min(8192, remaining)
                if len(stream.readframes(count)) != count * result['channels'] * 2:
                    raise ValueError('Audio WAV is truncated')
                remaining -= count
            return result
    except (wave.Error, EOFError) as exc:
        raise ValueError('Prepared audio requires complete PCM16 WAV') from exc


def read_wave(identity):
    import torch
    verify_file(identity)
    geometry = wave_geometry(identity['path'])
    if Fraction(geometry['samples'], geometry['sample_rate']) > 16:
        raise ValueError('Prepared audio loading is bounded to one physical native window or whole scheduled utterance')
    with wave.open(str(identity['path']), 'rb') as stream:
        raw = stream.readframes(geometry['samples'])
    tensor = torch.frombuffer(bytearray(raw), dtype=torch.int16).clone().float() / 32768
    return {'waveform': tensor.reshape(-1, geometry['channels']).T.unsqueeze(0),
            'sample_rate': geometry['sample_rate']}


def write_wave(path, audio):
    import torch
    waveform = audio.get('waveform') if isinstance(audio, dict) else None
    if (not isinstance(waveform, torch.Tensor) or waveform.ndim != 3
            or waveform.shape[0] != 1 or waveform.shape[1] not in (1, 2) or waveform.shape[2] <= 0):
        raise ValueError('Whole audio waveform requires batch one and one or two channels')
    rate = audio.get('sample_rate')
    if type(rate) is not int or rate <= 0:
        raise ValueError('Whole audio waveform requires a positive integer sample rate')
    if not bool(torch.isfinite(waveform).all().item()):
        raise ValueError('Whole audio waveform must be finite')
    if bool((waveform.abs() > 1).any().item()):
        raise ValueError('Whole audio waveform clips; supply an explicit safe mix gain')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (waveform.detach().cpu()[0].T.contiguous() * 32767).round().to(torch.int16)
    # A unique filename is published only by its immutable receipt.
    with path.open('xb') as raw:
        with wave.open(raw, 'wb') as stream:
            stream.setnchannels(int(waveform.shape[1]))
            stream.setsampwidth(2)
            stream.setframerate(rate)
            stream.writeframes(pcm.numpy().tobytes())
    return {**file_identity(path), **wave_geometry(path)}


def place_utterance(window, dialogue, samples, sample_rate=SAMPLE_RATE, forward_frames=0):
    w = validate_window(window)
    if type(samples) is not int or samples <= 0:
        raise ValueError('A whole utterance requires a positive actual sample count')
    start, end = dialogue.get('start_frame'), dialogue.get('end_frame')
    if (type(start) is not int or type(end) is not int
            or not w['new_start'] + forward_frames <= start < end <= w['new_end']):
        raise ValueError('Whole utterance interval must stay inside delivered new content after protected history')
    origin = w['new_start'] - w['context_frames']
    protected_samples = audio_ticks(w['context_frames'] + forward_frames) * SAMPLES_PER_TICK
    if frame_sample(start - origin) < protected_samples:
        raise ValueError('Whole utterance starts inside actual protected audio ticks; choose a later complete interval')
    a = frame_sample(start - origin, sample_rate)
    deadline = frame_sample(end - origin, sample_rate)
    b = a + samples
    if b > deadline:
        raise ValueError('The whole utterance exceeds its explicit deadline; no truncation or retiming is allowed')
    if b > frame_sample(w['local_new_end'], sample_rate):
        raise ValueError('The whole utterance exceeds the delivered interval')
    return a, b


def _backend_identity(backend):
    if not isinstance(backend, dict) or not backend.get('backend') or not backend.get('requested_model'):
        raise ValueError('Speech backend requires requested model provenance')
    model, code = backend.get('resolved_model'), backend.get('code')
    if (not isinstance(model, dict) or not model.get('name') or not model.get('weights')
            or any(not isinstance(item, dict) or not re.fullmatch(r'[0-9a-f]{64}', str(item.get('sha256', '')))
                   for item in model['weights'])):
        raise ValueError('Speech resolved model requires actual weight identities')
    if (not isinstance(code, dict) or not code
            or any(not re.fullmatch(r'[0-9a-f]{64}', str(value)) for value in code.values())):
        raise ValueError('Speech backend requires actual code identities')
    if not isinstance(backend.get('parameters'), dict):
        raise ValueError('Speech backend requires explicit generation parameters')
    return deepcopy(backend)


def sentence_spec(dialogue, source, reference_transcript, backend, seed, director_seed=None):
    if dialogue.get('audio_slot') != 1:
        raise ValueError('Automatic Fish speech requires explicit original Audio1 binding; Subject/Sx numbers are not audio slots')
    if (not isinstance(dialogue.get('text'), str) or not dialogue['text'].strip()
            or not isinstance(dialogue.get('speaker'), str)
            or not re.search(r'<Subject\s+[1-9]\d*>', dialogue['speaker'])
            or not re.search(r'\bS[1-9]\d*\b', dialogue['speaker'])):
        raise ValueError('Whole utterance requires exact words and explicit Subject/Sx speaker binding')
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError('Speech requires a recorded stable generation seed')
    return {'text': dialogue['text'], 'language': dialogue.get('language', 'auto'),
            'speaker': dialogue['speaker'], 'audio_slot': 1, 'source': verify_file(source),
            'reference_transcript': str(reference_transcript or ''),
            'backend': _backend_identity(backend), 'seed': seed, 'director_seed': director_seed}


def sentence_identity(spec):
    identity = {key: deepcopy(spec[key]) for key in ('text', 'language', 'speaker', 'audio_slot', 'source',
                                                    'reference_transcript', 'backend', 'seed')}
    identity['source'] = {key: spec['source'][key] for key in ('sha256', 'bytes')}
    identity['director_seed'] = spec.get('director_seed')
    # Installed paths identify files for verification, not the shared content.
    backend = identity['backend']
    backend.pop('code_files', None)
    backend['resolved_model'].pop('path', None)
    for weight in backend['resolved_model']['weights']:
        weight.pop('path', None)
    return _key(identity)


def _window_sound_requirements(contract, window, override):
    """Mirror the compiler's section, remaining-phase and timed-sound precedence."""
    sections = override.get('sections', {})
    start, end = window['new_start'] + override.get('forward_frames', 0), window['new_end']
    replacements = {beat['id']: beat for beat in override.get('remaining_beats', [])}
    sounds = []
    for original in contract.get('beats', []):
        a, b = max(start, original.get('start_frame', start)), min(end, original.get('end_frame', end))
        if a >= b:
            continue
        # A remaining beat replaces the whole old prose group: omitted old
        # sound is absent, matching compile_window rather than leaking it back.
        beat = replacements.get(original.get('id'), original)
        if 'timed_actions' in beat:
            for action in beat['timed_actions']:
                if max(a, action['start_frame']) < min(b, action['end_frame']):
                    sounds.append(action.get('sound', beat.get('sound', '')))
        else:
            sounds.append(beat.get('sound', ''))
    return (sections.get('overall_soundscape', contract.get('overall_soundscape', '')),
            sections.get('non_diegetic_music', contract.get('non_diegetic_music', '')), sounds)


def recipe_for_window(contract, window, settings):
    """Resolve only explicit soundscape policy; never guess ambience or speakers."""
    recipe = contract.get('target_audio')
    active_override = {}
    for override in contract.get('window_overrides', []):
        if (override['new_start'], override['new_end']) == (window['new_start'], window['new_end']):
            recipe = override.get('target_audio', recipe)
            active_override = override
    if recipe is not None and not isinstance(recipe, dict):
        raise ValueError('target_audio must be an explicit complete audio recipe')
    requested = settings.get('voice_mode', 'none')
    all_lines = contract.get('dialogues', [])
    words = '\n'.join(line['text'] for line in all_lines if line.get('audio_slot') == 1)
    target = str(settings.get('target_dialogue', '') or '')
    if requested == 'fish_lock' and target.strip() and ' '.join(target.split()) != ' '.join(words.split()):
        raise ValueError('Nonempty target_dialogue conflicts with the complete timed cloned words')
    if recipe and recipe.get('kind') == 'complete_wav':
        return deepcopy(recipe)
    if requested != 'fish_lock' and recipe is None:
        return None
    if recipe is not None and recipe.get('kind') != 'fish':
        raise ValueError('target_audio supports complete_wav or explicit Fish composition')
    recipe = deepcopy(recipe or {'kind': 'fish'})
    background = recipe.get('background')
    soundscape, requested_music, sounds = _window_sound_requirements(contract, window, active_override)
    # A phrase inside a larger ambient/Foley request is not a silence recipe.
    # Unrecognised prose needs an explicit complete background rather than
    # semantic guessing that could silently remove requested sound.
    explicit_silence = bool(re.fullmatch(
        r'\s*(?:intentional\s+silence(?:\s+except\s+(?:the\s+)?scheduled\s+(?:dialogue|speech|utterances))?'
        r'|(?:刻意|有意)(?:静音|静默)(?:[，,；;]\s*(?:仅|只)(?:保留|有)?(?:已安排|计划|预定)?的?(?:对白|台词|发言))?)'
        r'[.。]?\s*', soundscape, re.I))
    if background is None and explicit_silence:
        background = {'kind': 'silence', 'intentional': True}
    if not isinstance(background, dict) or background.get('kind') not in {'silence', 'complete_wav'}:
        raise ValueError('Fish target locks the complete soundscape: supply intentional silence or a complete background WAV recipe')
    if background['kind'] == 'silence':
        if background.get('intentional') is not True or not explicit_silence:
            raise ValueError('Silence background must agree with explicit intentional silence in overall_soundscape')
        music = str(requested_music).strip().lower().rstrip('.')
        if music not in {'n/a', 'none', 'no music', 'silence', '无', '无配乐', '无音乐'}:
            raise ValueError('Fish silence cannot omit requested music from the complete soundscape')
        for requested_sound in sounds:
            sound = str(requested_sound).strip().lower().rstrip('.')
            if sound and sound not in {'n/a', 'none', 'silence', 'no sound', '无', '静音', '无声'}:
                raise ValueError('Fish silence cannot omit requested soundscape effects; supply a complete background')
    recipe['background'] = background
    supplied = recipe.get('speech_sources', {})
    if not isinstance(supplied, dict):
        raise ValueError('Speech sources require explicit per-utterance waveform/voice provenance')
    lines = [line for line in all_lines
             if window['new_start'] <= line['start_frame'] < line['end_frame'] <= window['new_end']]
    for line in lines:
        if line.get('audio_slot') != 1 and line['id'] not in supplied:
            raise ValueError('Fish auto requires explicit original Audio1 binding; an ordinary speaker requires a supplied whole waveform')
    recipe['target_dialogue'] = words
    recipe['reference_transcript'] = str(settings.get('reference_transcript', '') or '')
    return recipe


def validate_target_audio(recipe, window, dialogues, original_sources=None, forward_frames=0):
    w = validate_window(window)
    if not isinstance(recipe, dict) or recipe.get('kind') != 'complete_wav':
        raise ValueError('Native target audio requires a complete physical-window WAV recipe')
    identity = verify_file(recipe)
    geometry = wave_geometry(identity['path'])
    expected = {'sample_rate': SAMPLE_RATE, 'channels': 2, 'samples': physical_samples(w)}
    if geometry != expected or any(recipe.get(key) != value for key, value in expected.items()):
        raise ValueError('Target audio sample/channel geometry must cover the complete physical window including its known suffix')
    if recipe.get('window') != w:
        raise ValueError('Target audio window geometry differs from this physical window')
    provenance = recipe.get('provenance')
    if not isinstance(provenance, dict) or not str(provenance.get('method', '')).strip():
        raise ValueError('Complete target audio requires generation method provenance')
    sources = recipe.get('sources')
    if not isinstance(sources, list):
        raise ValueError('Complete target audio requires explicit original/source identities')
    sources = [verify_file(item) for item in sources]
    hashes = {item['sha256'] for item in sources}
    originals = {int(slot): verify_file(item) for slot, item in (original_sources or {}).items()}
    utterances = recipe.get('utterances')
    if not isinstance(utterances, list) or len(utterances) != len(dialogues):
        raise ValueError('Complete target must identify every whole scheduled utterance')
    by_id = {item.get('id'): item for item in utterances if isinstance(item, dict)}
    if len(by_id) != len(utterances):
        raise ValueError('Complete target utterance ids must be unique')
    for line in dialogues:
        item = by_id.get(line['id'], {})
        for field in ('text', 'speaker', 'audio_slot', 'language', 'start_frame', 'end_frame'):
            if item.get(field) != line.get(field):
                raise ValueError('Target utterance words/text or original speaker binding differs from the contract')
        a, b = item.get('start_sample'), item.get('end_sample')
        if type(a) is not int or type(b) is not int or a >= b:
            raise ValueError('Target utterance requires actual complete placement/duration')
        if place_utterance(w, line, b - a, forward_frames=forward_frames) != (a, b):
            raise ValueError('Target utterance interval differs from the original timeline')
        if item.get('source_sha256') not in hashes or not item.get('generation'):
            raise ValueError('Target utterance requires actual source and generation provenance')
        slot = line.get('audio_slot')
        if slot is not None and (slot not in originals or originals[slot]['sha256'] not in hashes
                                 or item.get('original_reference_sha256') != originals[slot]['sha256']):
            raise ValueError('Target original Audio binding SHA-256 differs or is missing')
    return {**deepcopy(recipe), **identity, **geometry, 'sources': sources,
            'method': METHOD, 'content_review': 'pending'}


class AudioCache:
    """One shared namespace; receipt publication is atomic under the store lock."""
    def __init__(self, output_root):
        from .long_video_store import ProjectStore
        self.store = ProjectStore(output_root, CACHE_PROJECT)

    def sentence(self, spec):
        key = sentence_identity(spec)
        path = self.store.path(f'sentences/{key}/receipt.json')
        if not path.is_file():
            return None
        receipt = json.loads(path.read_text(encoding='utf-8'))
        if receipt.get('sentence_key') != key or sentence_identity(receipt.get('spec', {})) != key:
            raise ValueError('Immutable sentence receipt differs from its content identity')
        verify_file(receipt)
        if wave_geometry(receipt['path']) != {key: receipt[key] for key in ('sample_rate', 'channels', 'samples')}:
            raise ValueError('Immutable sentence waveform geometry changed')
        return receipt

    def publish_sentence(self, spec, audio):
        with self.store.transaction():
            existing = self.sentence(spec)
            if existing is not None:
                return existing
            key = sentence_identity(spec)
            path = self.store.path(f'sentences/{key}/{uuid.uuid4().hex}.wav')
            receipt = {**write_wave(path, audio), 'sentence_key': key, 'spec': deepcopy(spec),
                       'seed': spec['seed'], 'content_review': 'pending',
                       'resolved_execution': deepcopy(audio.get('_h3_fish_execution')),
                       'duration': {'samples': int(audio['waveform'].shape[-1]), 'sample_rate': audio['sample_rate']}}
            self.store.write_once(f'sentences/{key}/receipt.json', receipt)
            return receipt


def preparation_spec(contract, window, settings, assets, forward_frames=0):
    recipe = recipe_for_window(contract, window, settings)
    if recipe is None:
        return None
    originals = {str(slot): verify_file(item) for slot, item in assets.get('original_sources', {}).items()}
    lines = [deepcopy(line) for line in contract['dialogues']
             if window['new_start'] + forward_frames <= line['start_frame'] < line['end_frame'] <= window['new_end']]
    result = {'version': 1, 'window': deepcopy(window), 'forward_frames': forward_frames,
              'original_sources': originals, 'dialogues': lines,
              'requested_voice': {key: settings.get(key, '') for key in
                                  ('voice_mode', 'fish_model_path', 'reference_transcript')}, 'sentences': {}}
    if recipe['kind'] == 'complete_wav':
        result['recipe'] = validate_target_audio(recipe, window, lines, originals, forward_frames)
        return result
    recipe.pop('target_dialogue', None)  # Unowned/future words never invalidate an earlier window.
    result['recipe'] = recipe
    background = recipe['background']
    if background['kind'] == 'complete_wav':
        result['recipe']['background'] = validate_target_audio(background, window, [], {}, forward_frames)
    supplied = recipe.get('speech_sources', {})
    result['recipe']['speech_sources'] = {line['id']: deepcopy(supplied[line['id']])
                                         for line in lines if line['id'] in supplied}
    for line in lines:
        if line['id'] in supplied:
            item = supplied[line['id']]
            if (not isinstance(item, dict) or item.get('text') != line['text']
                    or item.get('speaker') != line['speaker'] or not item.get('provenance')):
                raise ValueError('Supplied whole speech requires exact words/speaker and explicit voice provenance')
            actual = verify_file(item)
            geometry = wave_geometry(actual['path'])
            place_utterance(window, line, geometry['samples'], geometry['sample_rate'], forward_frames)
            slot = line.get('audio_slot')
            if slot is not None and (str(slot) not in originals or
                    item['provenance'].get('original_reference_sha256') != originals[str(slot)]['sha256']):
                raise ValueError('Supplied speech original voice binding differs from the actual reference')
            result['recipe']['speech_sources'][line['id']] = {**item, **actual, **geometry}
            continue
        if '1' not in originals:
            raise ValueError('Fish preparation requires the actual original Audio1 source file')
        backend = deepcopy(assets.get('fish_backend'))
        if not backend:
            raise ValueError('Fish requires installed backend/model/code identities before queueing')
        language = line.get('language', 'auto')
        language = {'mandarin': 'zh', 'chinese': 'zh', '中文': 'zh', '普通话': 'zh',
                    'english': 'en', '英语': 'en'}.get(language.lower(), language)
        backend['parameters']['language'] = language
        overrides = recipe.get('parameters', {})
        if not isinstance(overrides, dict) or set(overrides) - set(backend['parameters']):
            raise ValueError('Unsupported Fish generation parameters')
        backend['parameters'].update(overrides)
        director_seed = int(settings.get('seed', 0))
        preliminary = sentence_spec(line, originals['1'], recipe['reference_transcript'], backend, 0, director_seed)
        stable_seed = int(_key({'sentence': sentence_identity(preliminary), 'voice_seed': director_seed})[:8], 16) % 2**31
        result['sentences'][line['id']] = dict(preliminary, seed=stable_seed)
    return result


def preparation_identity(spec):
    return _key(spec)


def preparation_files(spec):
    files = list(spec['original_sources'].values())
    recipe = spec['recipe']
    if recipe['kind'] == 'complete_wav':
        files.extend([recipe, *recipe['sources']])
    else:
        if recipe['background']['kind'] == 'complete_wav':
            files.extend([recipe['background'], *recipe['background']['sources']])
        files.extend(recipe.get('speech_sources', {}).values())
        for sentence in spec['sentences'].values():
            files.extend(sentence['backend']['resolved_model']['weights'])
            files.extend(sentence['backend'].get('code_files', []))
    return files


def load_prepared_target(store, spec, native_request=None):
    key = preparation_identity(spec)
    path = store.path(f'audio/{key}/receipt.json')
    if not path.is_file():
        if native_request is None or native_request.get('audio_preparation_key') != key:
            return None
        # A published native request already binds the final target bytes.
        # Recover only that exact receipt; never run TTS/composition to decode
        # or finish an existing clean AV take after receipt loss.
        original = json.loads(store.path(f'audio/{key}/request.json').read_text(encoding='utf-8'))
        if original != spec:
            raise ValueError('Immutable audio preparation request differs during target recovery')
        target = validate_target_audio(native_request['target_audio'], spec['window'], spec['dialogues'],
                                       spec['original_sources'], spec['forward_frames'])
        with store.transaction():
            if not path.is_file():
                store.write_once(f'audio/{key}/receipt.json', {'preparation_key': key, 'target_audio': target})
    receipt = json.loads(path.read_text(encoding='utf-8'))
    if receipt.get('preparation_key') != key:
        raise ValueError('Immutable prepared target receipt identity differs')
    request = json.loads(store.path(f'audio/{key}/request.json').read_text(encoding='utf-8'))
    if request != spec:
        raise ValueError('Immutable audio preparation request differs')
    return validate_target_audio(receipt['target_audio'], spec['window'], spec['dialogues'],
                                 spec['original_sources'], spec['forward_frames'])


def _verify_backend(spec):
    _backend_identity(spec['backend'])
    weights = spec['backend']['resolved_model']['weights']
    codes = spec['backend'].get('code_files')
    if not codes:
        raise ValueError('Installed Fish code files are missing; provenance is not verified')
    for identity in [spec['source'], *weights, *codes]:
        verify_file(identity)


def _generate_fish(spec):
    """Only called by the queued preparation node, never the coordinator."""
    from .fish import fish_voice_clone_node
    from .voice_guard import H3_REF_AUDIO_MIN_SECONDS, H3_REF_AUDIO_MAX_SECONDS
    from comfy_extras.nodes_audio import load
    import importlib
    waveform, rate = load(spec['source']['path'])
    if type(rate) is not int or rate <= 0:
        raise ValueError('Original Audio1 reference requires a valid sample rate to check duration')
    duration = Fraction(int(waveform.shape[-1]), rate)
    if not H3_REF_AUDIO_MIN_SECONDS <= duration <= H3_REF_AUDIO_MAX_SECONDS:
        raise ValueError('Original Audio1 reference must be 2–15 seconds for the retained H3 reference route')
    reference = {'waveform': waveform.unsqueeze(0), 'sample_rate': rate}
    backend = spec['backend']
    node = fish_voice_clone_node()
    package = node.__class__.__module__.rsplit('.', 1)[0]
    loader = importlib.import_module(package + '.loader')
    parameters = deepcopy(backend['parameters'])
    device, _ = loader.resolve_device(parameters['device'])
    precision = loader.resolve_precision(parameters['precision'], backend['resolved_model']['name'], device)
    parameters.update(device=device, precision=str(precision).removeprefix('torch.'))
    if device != 'cuda' and parameters['attention'] in {'sage_attention', 'flash_attention'}:
        parameters['attention'] = 'sdpa'  # Record the installed loader's actual resolution.
    fish_cache = importlib.import_module(package + '.model_cache')
    # The external cache key omits weight/code bytes. Take fresh ownership so
    # a same-name replacement cannot silently reuse an older engine while the
    # sentence receipt records newly frozen identities.
    fish_cache.unload_engine()
    try:
        if device == 'cuda':
            # Fish loads LLaMA/DAC outside Comfy's managed-model accounting.
            # Hand over VRAM before that external loader runs; explicit CPU
            # preparation leaves unrelated Comfy-managed models resident.
            model_management = importlib.import_module('comfy.model_management')
            model_management.unload_all_models()
            model_management.soft_empty_cache()
        result = node.generate(model_path=backend['resolved_model']['name'], text=spec['text'], reference_audio=reference,
                               reference_text=spec['reference_transcript'], seed=spec['seed'], **parameters)
    finally:
        if not parameters['keep_model_loaded']:
            fish_cache.unload_engine()
    if not isinstance(result, (tuple, list)) or not result:
        raise ValueError('Fish did not return a whole AUDIO waveform')
    return {**result[0], '_h3_fish_execution': {key: parameters[key] for key in
                                             ('device', 'precision', 'attention', 'keep_model_loaded')}}


def _native_waveform(audio):
    import torch
    waveform = audio['waveform']
    if waveform.ndim != 3 or waveform.shape[0] != 1 or waveform.shape[1] not in (1, 2):
        raise ValueError('Whole speech waveform requires batch one, mono or stereo')
    if audio['sample_rate'] != SAMPLE_RATE:
        import comfy.audio
        waveform = comfy.audio.resample(waveform, audio['sample_rate'], SAMPLE_RATE)
    waveform = waveform.detach().to(device='cpu', dtype=torch.float32)
    if waveform.shape[1] == 1:
        waveform = waveform.repeat(1, 2, 1)
    return waveform


def compose_target(store, spec):
    """One bounded physical window; no looping, speech cutting or rate changes."""
    import torch
    key = preparation_identity(spec)
    existing = load_prepared_target(store, spec)
    if existing is not None:
        return existing
    recipe = spec['recipe']
    path = store.path(f'audio/{key}/{uuid.uuid4().hex}.wav')
    if recipe['kind'] == 'complete_wav':
        manifest = validate_target_audio(recipe, spec['window'], spec['dialogues'],
                                         spec['original_sources'], spec['forward_frames'])
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as dest, Path(manifest['path']).open('rb') as source:
            shutil.copyfileobj(source, dest, 1024 * 1024)
        copied = file_identity(path)
        if any(copied[field] != recipe[field] for field in ('sha256', 'bytes')):
            raise ValueError('Copied target audio hash/SHA-256 or bytes changed from the frozen supplied waveform')
        manifest.update(copied)
    else:
        background = recipe['background']
        sources = list(spec['original_sources'].values())
        if background['kind'] == 'silence':
            waveform = torch.zeros((1, 2, physical_samples(spec['window'])), dtype=torch.float32)
        else:
            waveform = _native_waveform(read_wave(background))
            sources.extend([background, *background['sources']])
        utterances = []
        cache = AudioCache(store.output_root)
        for line in spec['dialogues']:
            if line['id'] in spec['sentences']:
                source = cache.sentence(spec['sentences'][line['id']])
                if source is None:
                    raise ValueError('Whole sentence receipt must exist before target composition')
                generation = {'sentence_key': source['sentence_key'], **source['spec']['backend'],
                              'seed': source['seed'], 'director_seed': source['spec'].get('director_seed'),
                              'resolved_execution': source.get('resolved_execution'),
                              'reference_transcript': source['spec']['reference_transcript']}
            else:
                source = recipe['speech_sources'][line['id']]
                generation = deepcopy(source['provenance'])
            # Check the actual whole original duration before resampling, too.
            place_utterance(spec['window'], line, source['samples'], source['sample_rate'], spec['forward_frames'])
            speech = _native_waveform(read_wave(source))
            a, b = place_utterance(spec['window'], line, int(speech.shape[-1]), forward_frames=spec['forward_frames'])
            gain = recipe.get('speech_gain', 1.0)
            if type(gain) not in (int, float) or not 0 < gain <= 8:
                raise ValueError('Speech mix gain must be an explicit finite positive value at most eight')
            waveform[..., a:b] += speech * gain
            sources.append(source)
            utterances.append({**line, 'start_sample': a, 'end_sample': b, 'source_sha256': source['sha256'],
                               **({'original_reference_sha256': spec['original_sources'][str(line['audio_slot'])]['sha256']}
                                  if line.get('audio_slot') is not None else {}),
                               'source_duration': {'samples': source['samples'], 'sample_rate': source['sample_rate']},
                               'generation': generation, 'gain': gain})
        manifest = {'kind': 'complete_wav', **write_wave(path, {'waveform': waveform, 'sample_rate': SAMPLE_RATE}),
                    'window': spec['window'], 'sources': [verify_file(item) for item in sources], 'utterances': utterances,
                    'provenance': {'method': 'Whole scheduled utterances over explicit complete background before native video',
                                   'background': deepcopy(background), 'requested_voice': spec['requested_voice'],
                                   'seed': {item['id']: item['generation'].get('seed') for item in utterances},
                                   'no_speech_truncation_or_retiming': True}}
        manifest = validate_target_audio(manifest, spec['window'], spec['dialogues'],
                                         spec['original_sources'], spec['forward_frames'])
    # Receipt publication is serialized; losing concurrent workers retain their
    # unique diagnostic WAV rather than overwriting a winner or its evidence.
    with store.transaction():
        existing = load_prepared_target(store, spec)
        if existing is not None:
            return existing
        store.write_once(f'audio/{key}/receipt.json', {'preparation_key': key, 'target_audio': manifest})
    return manifest


class MiniMaxH3LongVideoAudioPrepare:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'request_json': ('STRING',)}}

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = 'prepare'
    CATEGORY = 'MiniMax H3 导演台 Plus/原生长视频'

    def prepare(self, request_json):
        import folder_paths
        from .long_video_store import ProjectStore
        request = json.loads(request_json)
        store = ProjectStore(folder_paths.get_output_directory(), request['project_name'])
        state = store.read()
        job = state.get('active_job', {})
        if (job.get('prompt_id') != request['prompt_id'] or job.get('stage') != 'audio_prepare'
                or job.get('preparation_key') != request['preparation_key']):
            raise ValueError('Audio preparation no longer owns the current queued reservation')
        spec = json.loads(store.path(f"audio/{request['preparation_key']}/request.json").read_text(encoding='utf-8'))
        if preparation_identity(spec) != request['preparation_key']:
            raise ValueError('Audio preparation request content hash changed')
        if request['operation'] == 'sentence':
            sentence = spec['sentences'][request['sentence_id']]
            _verify_backend(sentence)
            cache = AudioCache(store.output_root)
            if cache.sentence(sentence) is None:
                generated = _generate_fish(sentence)
                # Preserve the complete unapproved utterance even when this
                # window's deadline rejects it. A later timing correction can
                # reuse the same whole audio rather than re-running Fish.
                cache.publish_sentence(sentence, generated)
                line = next(line for line in spec['dialogues'] if line['id'] == request['sentence_id'])
                place_utterance(spec['window'], line, int(generated['waveform'].shape[-1]),
                                generated['sample_rate'], spec['forward_frames'])
        elif request['operation'] == 'compose':
            compose_target(store, spec)
        else:
            raise ValueError('Unknown audio preparation operation')
        return ()
