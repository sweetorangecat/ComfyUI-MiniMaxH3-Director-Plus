"""Native clean-AV history, explicit two-stream masks and tensor-only evidence.

Torch and ComfyUI are imported on use. No sampler recipe, model loading or media
approval lives here. Only one source window is needed at a time.
"""
from __future__ import annotations

import hashlib
import json
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


def _history_ranges(source, endpoint, context_frames):
    for value, label in ((context_frames, 'context_frames'), (endpoint, 'source_endpoint')):
        if type(value) is not int or value < 39 or (value - 39) % 51:
            raise ValueError(f'{label} must be a shared AV endpoint: 39+51k')
    if endpoint < context_frames:
        raise ValueError('Source endpoint cannot be shorter than the protected history')
    sv, sa = streams(source)
    _native_geometry(sv, sa)
    ev, cv = video_steps(endpoint), video_steps(context_frames)
    ea, ca = audio_ticks(endpoint), audio_ticks(context_frames)
    if ev > int(sv.shape[2]) or ea > int(sa.shape[-1]):
        raise ValueError('The requested history extends beyond the source AV')
    return sv, sa, [ev - cv, ev], [ea - ca, ea]


def _finite(tensor, label):
    if not bool(_torch().isfinite(tensor).all().item()):
        raise ValueError(f'{label} must contain only finite values')


def _repair_ranges(repair, count, head):
    if not isinstance(repair, dict) or repair.get('lock_audio') is not True:
        raise ValueError('Bounded native video repair requires lock_audio=True')
    ranges = repair.get('video_ranges')
    if not isinstance(ranges, list) or not ranges:
        raise ValueError('Repair requires nonempty video_ranges in latent steps')
    canonical = []
    for pair in ranges:
        if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                or any(type(x) is not int for x in pair)):
            raise ValueError('Repair video_ranges must be integer latent-step pairs')
        a, b = pair
        if not head <= a < b <= count:
            raise ValueError('Repair video range is outside the target or changes its protected history')
        canonical.append([a, b])
    canonical.sort()
    if any(a[1] > b[0] for a, b in zip(canonical, canonical[1:])):
        raise ValueError('Repair video ranges must not overlap')
    return canonical


def validate_repair(window, repair):
    """Return canonical open video ranges using only the native window plan."""
    w = validate_window(window)
    head = video_steps(w['context_frames']) if w['context_frames'] else 0
    return _repair_ranges(repair, video_steps(w['sample_frames']), head)


def prepare_av(target, window, source=None, repair=None, nested_factory=None):
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
    cv, ca = (video_steps(c), audio_ticks(c)) if c else (0, 0)
    if cv >= int(tv.shape[2]) or ca >= int(ta.shape[-1]):
        raise ValueError('The target must include new video and audio after its protected history')
    video, audio = tv.clone(), ta.clone()
    vm = torch.ones((1, 1, *video.shape[2:]), dtype=torch.float32, device=video.device)
    am = torch.ones((1, 1, *audio.shape[2:]), dtype=torch.float32, device=audio.device)
    report = {
        'mode': 'repair' if repair is not None else 'continuation' if c else 'first',
        'sample_frames': w['sample_frames'], 'context_frames': c,
        'source_endpoint': w['source_endpoint'], 'video_steps': cv, 'audio_ticks': ca,
        'target_video_steps': int(video.shape[2]), 'target_audio_ticks': int(audio.shape[-1]),
        'source_video_range': None, 'source_audio_range': None,
        'new_start': w['new_start'], 'new_end': w['new_end'],
        'video_ranges': [], 'lock_audio': False, 'content_review': 'pending',
    }
    if repair is not None:
        if source is None:
            raise ValueError('Repair requires the complete failed source take')
        sv, sa = streams(source)
        _native_geometry(sv, sa, w['sample_frames'])
        _same_geometry(tv, ta, sv, sa, exact=True)
        _finite(sv, 'Source video')
        _finite(sa, 'Source audio')
        ranges = validate_repair(w, repair)
        video.copy_(sv.to(device=video.device))
        audio.copy_(sa.to(device=audio.device))
        vm.zero_()
        am.zero_()
        for a, b in ranges:
            vm[:, :, a:b] = 1
        report.update(video_ranges=ranges, lock_audio=True,
                      source_video_range=[0, int(sv.shape[2])], source_audio_range=[0, int(sa.shape[-1])])
    elif c:
        if source is None:
            raise ValueError('Continuation requires actual clean source AV history')
        sv, sa, vr, ar = _history_ranges(source, w['source_endpoint'], c)
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
        if not bool(((mask == 0) | (mask == 1)).all().item()):
            raise ValueError(f'Prepared {name} mask must contain only hard zero/one values')
        _finite(actual, 'Result ' + name)
        baseline = baseline.to(device=actual.device)
        expanded = mask.to(device=actual.device).expand_as(actual)
        protected = expanded == 0
        preserved = torch.equal(actual[protected], baseline[protected])
        if not preserved:
            raise ValueError(f'Result changed protected clean {name} values')
        open_region = expanded == 1
        exact.append(preserved)
        changed.append(not torch.equal(actual[open_region], baseline[open_region]))
    if context.get('mode') == 'repair' and not changed[0]:
        raise ValueError('Repair did not change any open video latent element')
    if require_change and not any(changed):
        raise ValueError('Sampling did not change any open AV latent element')
    return {'status': 'pass', 'protected_video_exact': exact[0], 'protected_audio_exact': exact[1],
            'open_video_changed': changed[0], 'open_audio_changed': changed[1],
            'shapes': [list(rv.shape), list(ra.shape)], 'content_review': 'pending'}


def slice_fingerprint(latent, endpoint, context_frames=39):
    """SHA-256 of exactly the two parent slices consumed by a continuation.

    Changes outside those slices do not invalidate this fingerprint. Source
    device, dtype and slice shapes are included before contiguous tensor bytes.
    """
    video, audio, vr, ar = _history_ranges(latent, endpoint, context_frames)
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
