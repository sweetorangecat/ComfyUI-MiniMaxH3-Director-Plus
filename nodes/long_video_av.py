"""Native clean-AV history, explicit two-stream masks and tensor-only evidence.

Torch and ComfyUI are imported on use. No sampler recipe, model loading or media
approval lives here. Only one source window is needed at a time.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
from copy import deepcopy
from fractions import Fraction

from .long_video_plan import FPS, audio_ticks, video_steps, validate_window


def _torch():
    import torch
    return torch


def _factory(factory):
    if factory is not None:
        return factory
    from comfy.nested_tensor import NestedTensor
    return NestedTensor


def streams(latent):
    """Require exactly the native batch-one video 5D and audio 4D streams."""
    if not isinstance(latent, dict):
        raise ValueError('A nested H3 video/audio LATENT dictionary is required')
    value = latent.get('samples')
    if not getattr(value, 'is_nested', False) or not callable(getattr(value, 'unbind', None)):
        raise ValueError('A clean nested H3 video/audio latent is required')
    values = tuple(value.unbind())
    torch = _torch()
    if (len(values) != 2 or not all(isinstance(v, torch.Tensor) for v in values)
            or values[0].ndim != 5 or values[1].ndim != 4):
        raise ValueError('Unexpected H3 AV stream shape: require video 5D and audio 4D')
    if any(int(t.shape[0]) != 1 for t in values):
        raise ValueError('Native long-video continuation requires batch size one')
    if any(any(int(n) <= 0 for n in t.shape) for t in values):
        raise ValueError('H3 AV shapes must have positive dimensions')
    return values


def _native_geometry(video, audio, frames=None):
    steps = int(video.shape[2])
    if steps < 2 or (steps - 2) % 5:
        raise ValueError('Unexpected native video temporal shape')
    inferred = 5 + 17 * ((steps - 2) // 5)
    if frames is not None and inferred != frames:
        raise ValueError('Target video shape does not match the planned sample_frames')
    if int(audio.shape[-1]) != audio_ticks(inferred):
        raise ValueError('Audio temporal shape does not match native video geometry')
    return inferred


def _same_geometry(target_video, target_audio, source_video, source_audio, exact=False):
    video_axes = lambda t: (t.shape[0], t.shape[1], *t.shape[3:])
    if video_axes(target_video) != video_axes(source_video) or target_audio.shape[:3] != source_audio.shape[:3]:
        raise ValueError('Source and target AV spatial/channel geometry differs')
    if target_video.dtype != source_video.dtype or target_audio.dtype != source_audio.dtype:
        raise ValueError('Source and target AV dtype must agree to preserve clean history exactly')
    if exact and (target_video.shape != source_video.shape or target_audio.shape != source_audio.shape):
        raise ValueError('Repair source and target must have identical AV shapes')


def _forward(value):
    if type(value) is not int or value < 0 or value % 17:
        raise ValueError('forward_frames must be zero or a positive multiple of 17')
    return value


def conditioning_geometry(window, forward_frames=0, source_frames=None):
    """True native AV clocks for history plus an explicitly reviewed source tail.

    This validates geometry only. It cannot grant a tail review. The source
    starts at ev-cv / ea-ca; both end clocks advance by the actual frame count.
    """
    w = validate_window(window)
    f, c, e = _forward(forward_frames), w['context_frames'], w['source_endpoint']
    if f and (not c or f >= w['new_end'] - w['new_start']):
        raise ValueError('Forward conditioning needs history and must be shorter than the new interval')
    if source_frames is not None:
        video_steps(source_frames)
        if c and e + f > source_frames:
            raise ValueError('Reviewed forward source has insufficient actual unused tail')
    head = c + f
    return {'forward_frames': f, 'protected_frames': head,
            'video_steps': video_steps(head) if head else 0,
            'audio_ticks': audio_ticks(head) if head else 0,
            'source_video_range': [video_steps(e) - video_steps(c), video_steps(e + f)] if c else None,
            'source_audio_range': [audio_ticks(e) - audio_ticks(c), audio_ticks(e + f)] if c else None,
            'source_local_range': [e, e + f] if c else None,
            'consumed_local_range': [e - c, e + f] if c else None}


def _history_ranges(source, endpoint, context_frames, forward_frames=0):
    for value, label in ((context_frames, 'context_frames'), (endpoint, 'source_endpoint')):
        if type(value) is not int or value < 39 or (value - 39) % 51:
            raise ValueError(f'{label} must be a shared AV endpoint: 39+51k')
    if endpoint < context_frames:
        raise ValueError('Source endpoint cannot be shorter than the protected history')
    f = _forward(forward_frames)
    sv, sa = streams(source)
    _native_geometry(sv, sa)
    ev, cv = video_steps(endpoint + f), video_steps(context_frames)
    ea, ca = audio_ticks(endpoint + f), audio_ticks(context_frames)
    if ev > int(sv.shape[2]) or ea > int(sa.shape[-1]):
        raise ValueError('The requested history extends beyond the source AV')
    return sv, sa, [video_steps(endpoint) - cv, ev], [audio_ticks(endpoint) - ca, ea]


def _finite(tensor, label):
    if not bool(_torch().isfinite(tensor).all().item()):
        raise ValueError(f'{label} must contain only finite values')


def _repair_ranges(ranges, count, head, label):
    if not isinstance(ranges, list):
        raise ValueError(f'Repair {label} must be a list of integer pairs')
    canonical = []
    for pair in ranges:
        if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                or any(type(x) is not int for x in pair)):
            raise ValueError(f'Repair {label} must be integer pairs')
        a, b = pair
        if not head <= a < b <= count:
            raise ValueError(f'Repair {label} is outside the target or changes its protected history/forward prefix')
        canonical.append([a, b])
    canonical.sort()
    if any(a[1] > b[0] for a, b in zip(canonical, canonical[1:])):
        raise ValueError(f'Repair {label} ranges must not overlap')
    return canonical


def normalize_repair(window, repair, forward_frames=0):
    """Canonical bounded temporal recipe with an optional reviewed native reference.

    video_ranges/audio_ranges are end-exclusive integer native-step/tick pairs.
    video_weights is [[start_step,end_step,weight], ...], finite weights in
    [0,1], disjoint from other video spans. Weights are canonical float32 mask
    values; a requested positive weight that underflows to zero is rejected.
    Audio ranges require lock_audio=False.
    At least one stream must contain a positive open region. Internal source,
    index and seconds-mapping evidence may accompany this immutable recipe.
    """
    w = validate_window(window)
    geometry = conditioning_geometry(w, forward_frames)
    known = {'video_ranges', 'audio_ranges', 'video_weights', 'lock_audio',
             'source_take', 'index', 'conditioning_extent', 'video_reference'}
    if not isinstance(repair, dict) or set(repair) - known:
        raise ValueError('Unsupported repair fields; only bounded temporal video/audio repair is supported')
    lock = repair.get('lock_audio', True)
    if type(lock) is not bool:
        raise ValueError('Repair lock_audio must be a boolean')
    vr = _repair_ranges(repair.get('video_ranges', []), video_steps(w['sample_frames']),
                        geometry['video_steps'], 'video_ranges')
    ar = _repair_ranges(repair.get('audio_ranges', []), audio_ticks(w['sample_frames']),
                        geometry['audio_ticks'], 'audio_ranges')
    if (lock and ar) or (not lock and not ar):
        raise ValueError('Audio repair requires explicit audio_ranges and lock_audio=False')
    weights = repair.get('video_weights', [])
    if not isinstance(weights, list):
        raise ValueError('video_weights must be bounded temporal [start,end,weight] spans')
    canonical_weights = []
    for item in weights:
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            raise ValueError('video_weights must contain [start,end,weight] spans')
        a, b, weight = item
        _repair_ranges([[a, b]], video_steps(w['sample_frames']), geometry['video_steps'], 'video_weights')
        if type(weight) not in (int, float) or not 0 <= weight <= 1 or not math.isfinite(weight):
            raise ValueError('video_weights must be finite numbers in [0,1]')
        effective = struct.unpack('f', struct.pack('f', float(weight)))[0]
        if weight > 0 and effective == 0:
            raise ValueError('Positive video weight underflows the actual float32 noise mask')
        canonical_weights.append([a, b, effective])
    canonical_weights.sort()
    spans = sorted([*vr, *[item[:2] for item in canonical_weights]])
    if any(a[1] > b[0] for a, b in zip(spans, spans[1:])):
        raise ValueError('Video repair ranges and weights must not overlap')
    if not vr and not ar and not any(item[2] > 0 for item in canonical_weights):
        raise ValueError('Repair requires a positive open video or audio region')
    result = {'video_ranges': vr, 'lock_audio': lock}
    if ar:
        result['audio_ranges'] = ar
    if canonical_weights:
        result['video_weights'] = canonical_weights
    if 'video_reference' in repair:
        from .long_video_repair import normalize_video_reference
        result['video_reference'] = normalize_video_reference(
            repair['video_reference'], video_steps(w['sample_frames']), geometry['video_steps'],
            [*vr, *[item[:2] for item in canonical_weights if item[2] > 0]])
    for key in ('source_take', 'index', 'conditioning_extent'):
        if key in repair:
            result[key] = deepcopy(repair[key])
    return result


def validate_repair(window, repair, forward_frames=0):
    """Legacy list-returning video-range view; use normalize_repair for both AV."""
    return normalize_repair(window, repair, forward_frames)['video_ranges']


def _seconds_frame(value):
    """Remove only a float's representation noise at an integer frame boundary."""
    frames = Fraction(str(value)) * FPS
    if type(value) is float:
        nearest = round(frames)
        # Derived N/24 seconds can land just outside N after decimal roundtrip.
        # One actual input-float ULP is tiny; genuine sub-frame spans stay exact.
        if abs(frames - nearest) <= Fraction.from_float(math.ulp(value)) * FPS:
            return Fraction(nearest)
    return frames


def repair_seconds(window, start_seconds, end_seconds, stream='video', forward_frames=0):
    """Map film seconds outwards to native groups/ticks, never claim VAE isolation."""
    w = validate_window(window)
    g = conditioning_geometry(w, forward_frames)
    if stream not in {'video', 'audio', 'joint'}:
        raise ValueError('Repair stream must be video, audio or joint')
    for value in (start_seconds, end_seconds):
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError('Repair seconds must be finite numbers')
    a, b = (_seconds_frame(value) for value in (start_seconds, end_seconds))
    if not w['new_start'] <= a < b <= w['new_end']:
        raise ValueError('Repair seconds must lie inside the selected decoded delivery window')
    if a < w['new_start'] + forward_frames:
        raise ValueError('Requested interval overlaps protected forward AV; regenerate the accepted parent source to replace it')
    local_a, local_b = (value - w['new_start'] + w['context_frames'] for value in (a, b))
    floor = lambda value: value.numerator // value.denominator
    ceil = lambda value: -(-value.numerator // value.denominator)
    va = 0 if local_a < 5 else 2 + 5 * floor((local_a - 5) / 17)
    vb = 2 if local_b <= 5 else 2 + 5 * ceil((local_b - 5) / 17)
    aa, ab = floor(local_a * 40 / FPS), ceil(local_b * 40 / FPS)
    # Shared boundaries are native group boundaries, so outward snapping never
    # reopens a protected value when the requested interval starts after them.
    va, aa = max(va, g['video_steps']), max(aa, g['audio_ticks'])
    frame_at = lambda step: 0 if step == 0 else 5 + 17 * ((step - 2) // 5)
    extent = {'requested_seconds': [start_seconds, end_seconds], 'stream': stream,
              'video_local_frames': [frame_at(va), frame_at(vb)] if stream != 'audio' else None,
              'audio_local_ticks': [aa, ab] if stream != 'video' else None,
              'video_seconds': [float(Fraction(w['new_start'] - w['context_frames'] + frame_at(step), FPS))
                                for step in (va, vb)] if stream != 'audio' else None,
              'audio_seconds': [float(Fraction(w['new_start'] - w['context_frames'], FPS) + Fraction(tick, 40))
                                for tick in (aa, ab)] if stream != 'video' else None,
              'frame_isolation_guaranteed': False,
              'note': 'Outward native conditioning extent; VAE decoded frame isolation is not guaranteed.'}
    recipe = {'video_ranges': [[va, vb]] if stream != 'audio' else [], 'lock_audio': stream == 'video',
              'conditioning_extent': extent}
    if stream != 'video':
        recipe['audio_ranges'] = [[aa, ab]]
    return normalize_repair(w, recipe, forward_frames)


def prepare_av(target, window, source=None, repair=None, nested_factory=None, forward_frames=0,
               target_audio=None, target_audio_locked=False, video_reference_source=None):
    """Clone the target, copy true parent history, and always emit both masks.

    A repair source is the complete failed take with identical shape, rather
    than a parent window. The caller supplies expert-verified latent-step
    ranges; this function never guesses a per-frame repair mapping.
    """
    w = validate_window(window)
    tv, ta = streams(target)
    _native_geometry(tv, ta, w['sample_frames'])
    _finite(tv, 'Target video')
    _finite(ta, 'Target audio')
    torch = _torch()
    factory = _factory(nested_factory)
    c = w['context_frames']
    geometry = conditioning_geometry(w, forward_frames)
    reference = repair.get('video_reference') if isinstance(repair, dict) else None
    if (reference is not None) != (video_reference_source is not None):
        raise ValueError('Video reference requires its explicit matching actual source and repair')
    cv, ca = geometry['video_steps'], geometry['audio_ticks']
    if cv >= int(tv.shape[2]) or ca >= int(ta.shape[-1]):
        raise ValueError('The target must include new video and audio after its protected history')
    video, audio = tv.clone(), ta.clone()
    conditioned = target_audio is not None or target_audio_locked
    if target_audio is not None:
        encoded = target_audio.get('samples') if isinstance(target_audio, dict) else None
        if not isinstance(encoded, torch.Tensor) or encoded.shape != audio.shape:
            raise ValueError('Encoded target audio geometry must exactly match the native audio stream')
        _finite(encoded, 'Encoded target audio')
        audio.copy_(encoded.to(device=audio.device, dtype=audio.dtype))
        _finite(audio, 'Installed target audio')
    elif target_audio_locked and repair is None:
        raise ValueError('Target audio requires a complete encoded latent before native generation')
    if conditioned and repair is not None and (repair.get('lock_audio', True) is not True or repair.get('audio_ranges')):
        raise ValueError('Target-conditioned repair must inherit complete audio; open audio requires a new base generation')
    vm = torch.ones((1, 1, *video.shape[2:]), dtype=torch.float32, device=video.device)
    am = torch.ones((1, 1, *audio.shape[2:]), dtype=torch.float32, device=audio.device)
    report = {
        'mode': 'repair' if repair is not None else 'continuation' if c else 'first',
        'sample_frames': w['sample_frames'], 'context_frames': c,
        'source_endpoint': w['source_endpoint'], 'video_steps': cv, 'audio_ticks': ca,
        'target_video_steps': int(video.shape[2]), 'target_audio_ticks': int(audio.shape[-1]),
        'source_video_range': None, 'source_audio_range': None,
        'new_start': w['new_start'], 'new_end': w['new_end'],
        'forward_frames': forward_frames, 'protected_frames': geometry['protected_frames'],
        'video_ranges': [], 'audio_ranges': [], 'video_weights': [],
        'lock_audio': False, 'content_review': 'pending',
    }
    if repair is not None:
        if source is None:
            raise ValueError('Repair requires the complete failed source take')
        sv, sa = streams(source)
        _native_geometry(sv, sa, w['sample_frames'])
        _same_geometry(tv, ta, sv, sa, exact=True)
        _finite(sv, 'Source video')
        _finite(sa, 'Source audio')
        recipe = normalize_repair(w, repair, forward_frames)
        video.copy_(sv.to(device=video.device))
        audio.copy_(sa.to(device=audio.device))
        vm.zero_()
        am.zero_()
        for a, b in recipe['video_ranges']:
            vm[:, :, a:b] = 1
        for a, b, weight in recipe.get('video_weights', []):
            vm[:, :, a:b] = weight
        for a, b in recipe.get('audio_ranges', []):
            am[..., a:b] = 1
        if reference is not None:
            rv, ra = streams(video_reference_source)
            _native_geometry(rv, ra)
            _same_geometry(tv, ta, rv, ra)
            a, b = reference['source_steps']
            x, y = reference['target_steps']
            if b > int(rv.shape[2]):
                raise ValueError('Video reference source has insufficient actual native video')
            _finite(rv[:, :, a:b], 'Video reference')
            video[:, :, x:y].copy_(rv[:, :, a:b].to(device=video.device))
            vm[:, :, x:y] = 0
            report['video_reference'] = deepcopy(recipe['video_reference'])
        report.update(video_ranges=recipe['video_ranges'], audio_ranges=recipe.get('audio_ranges', []),
                      video_weights=recipe.get('video_weights', []), lock_audio=recipe['lock_audio'],
                      source_video_range=[0, int(sv.shape[2])], source_audio_range=[0, int(sa.shape[-1])])
    elif c:
        if source is None:
            raise ValueError('Continuation requires actual clean source AV history')
        sv, sa, vr, ar = _history_ranges(source, w['source_endpoint'], c, forward_frames)
        _same_geometry(tv, ta, sv, sa)
        _finite(sv[:, :, vr[0]:vr[1]], 'Source video history')
        _finite(sa[..., ar[0]:ar[1]], 'Source audio history')
        video[:, :, :cv].copy_(sv[:, :, vr[0]:vr[1]].to(device=video.device))
        audio[..., :ca].copy_(sa[..., ar[0]:ar[1]].to(device=audio.device))
        vm[:, :, :cv] = 0
        am[..., :ca] = 0
        report.update(source_video_range=vr, source_audio_range=ar)
    elif source is not None:
        raise ValueError('A first window accepts a source only for an explicit bounded repair')
    if conditioned:
        # Install the planned target first; the exact actual source/forward
        # history above remains authoritative. The full encoded audio latent,
        # including the known suffix, is now protected. VAE decoded WAV bytes
        # are not promised to equal the original target waveform.
        am.zero_()
        report.update(method='target_audio_conditioned_native_video', lock_audio=True)
    result = dict(target)
    result.update(samples=factory((video, audio)), noise_mask=factory((vm, am)), _h3_long_video_context=report)
    return result


def validate_result(result, prepared, require_change=False):
    """Check finite streams and every protected element. No media quality approval."""
    rv, ra = streams(result)
    pv, pa = streams(prepared)
    _same_geometry(pv, pa, rv, ra, exact=True)
    _native_geometry(rv, ra)
    context = prepared.get('_h3_long_video_context')
    if not isinstance(context, dict):
        raise ValueError('Prepared AV validation context is missing')
    if 'noise_mask' not in prepared:
        raise ValueError('Prepared AV must retain explicit video and audio noise masks')
    mv, ma = streams({'samples': prepared['noise_mask']})
    torch = _torch()
    exact, changed = [], []
    for name, actual, baseline, mask in [('video', rv, pv, mv), ('audio', ra, pa, ma)]:
        expected_mask_shape = (1, 1, *baseline.shape[2:])
        if tuple(mask.shape) != expected_mask_shape:
            raise ValueError(f'Prepared {name} mask shape is inconsistent with the stream')
        _finite(mask, 'Prepared ' + name + ' mask')
        if not bool(((mask >= 0) & (mask <= 1)).all().item()):
            raise ValueError(f'Prepared {name} mask must contain only finite values in [0,1]')
        _finite(actual, 'Result ' + name)
        baseline = baseline.to(device=actual.device)
        expanded = mask.to(device=actual.device).expand_as(actual)
        protected = expanded == 0
        preserved = torch.equal(actual[protected], baseline[protected])
        if not preserved:
            raise ValueError(f'Result changed protected clean {name} values')
        open_region = expanded > 0
        exact.append(preserved)
        changed.append(not torch.equal(actual[open_region], baseline[open_region]))
        if context.get('mode') == 'repair' and bool(open_region.any().item()) and not changed[-1]:
            raise ValueError(f'Repair did not change any open {name} latent element')
    if require_change and not any(changed):
        raise ValueError('Sampling did not change any open AV latent element')
    return {'status': 'pass', 'protected_video_exact': exact[0], 'protected_audio_exact': exact[1],
            'open_video_changed': changed[0], 'open_audio_changed': changed[1],
            'shapes': [list(rv.shape), list(ra.shape)], 'content_review': 'pending'}


def slice_fingerprint(latent, endpoint, context_frames=39, forward_frames=0):
    """SHA-256 of exactly the two parent slices consumed by a continuation.

    Changes outside those slices do not invalidate this fingerprint. Source
    device, dtype and slice shapes are included before contiguous tensor bytes.
    """
    video, audio, vr, ar = _history_ranges(latent, endpoint, context_frames, forward_frames)
    digest = hashlib.sha256()
    torch = _torch()
    for name, tensor in [('video', video[:, :, vr[0]:vr[1]]), ('audio', audio[..., ar[0]:ar[1]])]:
        _finite(tensor, 'Parent ' + name + ' history')
        metadata = {'stream': name, 'device': str(tensor.device), 'dtype': str(tensor.dtype),
                    'shape': list(tensor.shape)}
        digest.update(json.dumps(metadata, sort_keys=True, separators=(',', ':')).encode('utf-8'))
        digest.update(b'\0')
        # Viewing bytes also supports bfloat16, for which numpy has no scalar dtype.
        digest.update(tensor.detach().contiguous().to(device='cpu').view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def audio_sample_range(window, sample_rate, available_samples=None):
    """Map a delivery to cumulative global sample indices without rounding drift.

    The local count follows the cumulative global clock. Pass the decoded source
    length as available_samples to reject missing samples; never pad, copy or
    change the audio rate to make the plan fit.
    """
    w = validate_window(window)
    if type(sample_rate) is not int or sample_rate <= 0:
        raise ValueError('Audio sample_rate must be a positive integer')
    rounded = lambda frames: round(Fraction(frames * sample_rate, FPS))
    global_start, global_end = rounded(w['new_start']), rounded(w['new_end'])
    count = global_end - global_start
    local_start = rounded(w['context_frames'])
    local_end = local_start + count
    if available_samples is None:
        available_samples = rounded(w['sample_frames'])
    if type(available_samples) is not int or available_samples < local_end:
        raise ValueError('Source audio has insufficient local samples for the cumulative delivery range')
    return {'local_start': local_start, 'local_end': local_end, 'global_start': global_start,
            'global_end': global_end, 'count': count}
