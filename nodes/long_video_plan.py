"""Pure frame planning and official Ref2VA text for native H3 AV windows.

Intervals are end-exclusive frames at 24 fps. A physical window is never an
editorial cut. These functions do not authorize assets or approve generated media.
"""
from __future__ import annotations

import copy
import json
import math
import re
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction

FPS = 24
_MIN_SAMPLE_FRAMES = 107  # First 17k+5 length inside the native 4–15 second budget.


def _integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{label} must be an integer >= {minimum}')
    return value


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{label} must be nonempty text')
    return value


def target_frame_count(seconds):
    """Round positive finite seconds half-up; the actual duration is frames / FPS."""
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
        raise ValueError('Target seconds must be a positive finite number')
    if seconds <= 0 or (isinstance(seconds, float) and not math.isfinite(seconds)):
        raise ValueError('Target seconds must be a positive finite number')
    product = seconds * FPS
    # Preserve the normal seconds-to-frame multiplication at representable ties,
    # while still accepting finite inputs whose product overflows a float.
    value = Decimal(str(product)) if not isinstance(product, float) or math.isfinite(product) else Decimal(str(seconds)) * FPS
    frames = int(value.to_integral_value(rounding=ROUND_HALF_UP))
    if frames < 1:
        raise ValueError('Target seconds round to zero frames')
    return frames


def video_steps(frames):
    """Native video geometry; accept only actual 17k+5 frame lengths."""
    _integer(frames, 'Native frame count', 5)
    if (frames - 5) % 17:
        raise ValueError('Native frame count must be 17k+5')
    return 2 + 5 * ((frames - 5) // 17)


def audio_ticks(frames):
    """The actual 40 Hz H3 audio clock, rounded using exact rational arithmetic."""
    _integer(frames, 'Audio frame count')
    return round(Fraction(frames * 40, FPS))


def _shared(endpoint, label):
    _integer(endpoint, label, 39)
    if (endpoint - 39) % 51:
        raise ValueError(f'{label} must be a shared AV endpoint: 39+51k')
    return endpoint


def _sample_at_least(frames):
    return max(_MIN_SAMPLE_FRAMES, 5 + 17 * max(0, (frames - 5 + 16) // 17))


def _intervals(intervals, total):
    result = []
    try:
        for interval in intervals:
            if not isinstance(interval, (tuple, list)) or len(interval) != 2:
                raise ValueError('Each protected interval must be [start, end)')
            a = _integer(interval[0], 'Protected interval start')
            b = _integer(interval[1], 'Protected interval end', 1)
            if not a < b <= total:
                raise ValueError('Protected interval must lie inside the target timeline')
            result.append((a, b))
    except TypeError as exc:
        raise ValueError('protected_intervals must be an iterable of frame intervals') from exc
    return sorted(set(result))


def plan_windows(total_frames, max_sample_frames=328, context_frames=39,
                 safety_tail_frames=34, protected_intervals=()):
    """Plan positive new intervals, retaining the two-clock phase at every handoff.

    ``endpoint`` is local to this window, ``source_endpoint`` to its parent.
    The final local delivery end may precede endpoint; only generated surplus is
    trimmed. Short deliveries still sample a valid native window without freezing
    or duplicating any delivered frames.
    """
    total = _integer(total_frames, 'total_frames', 1)
    maximum = _integer(max_sample_frames, 'max_sample_frames', _MIN_SAMPLE_FRAMES)
    if maximum > 15 * FPS:
        raise ValueError('A physical sample window must remain inside the native 4–15 second budget')
    _shared(context_frames, 'context_frames')
    tail = _integer(safety_tail_frames, 'safety_tail_frames')
    protected = _intervals(protected_intervals, total)
    endpoints = [e for e in range(39, maximum + 1, 51) if _sample_at_least(e + tail) <= maximum]
    if not endpoints:
        raise ValueError('Sample budget cannot fit a shared AV endpoint and safety tail')
    cap = endpoints[-1]
    windows = []
    start = 0
    while start < total:
        head = context_frames if windows else 0
        if cap <= head:
            raise ValueError('Sample budget cannot make positive progress after the protected history')
        if total - start <= cap - head:
            end = total
            endpoint = next(e for e in endpoints if e >= head + end - start)
        else:
            choices = [(e, start + e - head) for e in reversed(endpoints) if e > head]
            feasible = [(e, b) for e, b in choices if not any(x < b < y for x, y in protected)]
            if not feasible:
                raise ValueError(f'Cannot fit a protected dialogue or indivisible action after frame {start} '
                                 f'inside the {cap - head}-frame new-content budget at a shared AV endpoint')
            endpoint, end = feasible[0]
        windows.append({
            'index': len(windows), 'sample_frames': _sample_at_least(endpoint + tail),
            'context_frames': head, 'new_start': start, 'new_end': end,
            'endpoint': endpoint, 'source_endpoint': windows[-1]['local_new_end'] if windows else 0,
            'local_new_start': head, 'local_new_end': head + end - start,
        })
        start = end
    return windows


def _object(raw):
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f'Long-video contract JSON: {exc.msg}') from exc
    if not isinstance(raw, dict):
        raise ValueError('Long-video contract must be a JSON object')
    return copy.deepcopy(raw)


def _item_interval(item, total, label):
    if not isinstance(item, dict):
        raise ValueError(f'{label} must be an object')
    _text(item.get('id'), label + '.id')
    a = _integer(item.get('start_frame'), label + '.start_frame')
    b = _integer(item.get('end_frame'), label + '.end_frame', 1)
    if not a < b <= total:
        raise ValueError(f'{label} interval must lie inside [0, {total})')
    return a, b


def validate_contract(raw, total_frames):
    """Return a detached canonical scene contract with a fully covered action line.

    Reference slot declarations are metadata, never proof of mounting or review.
    Beats may overlap for simultaneous action; their union must cover the target.
    """
    total = _integer(total_frames, 'total_frames', 1)
    s = _object(raw)
    for key, expected in [('schema_version', 1), ('fps', FPS)]:
        if type(s.get(key)) is not int or s[key] != expected:
            raise ValueError(f'{key} must be {expected}')
    if 'total_frames' in s and (type(s['total_frames']) is not int or s['total_frames'] != total):
        raise ValueError('Contract total_frames must match the requested target frame count')
    s['total_frames'] = total
    for key in ('subject_definitions', 'summary', 'retention_analysis', 'overall_soundscape',
                'non_diegetic_music', 'scale_contract'):
        _text(s.get(key), key)
    if not re.search(r'\d', s['scale_contract']):
        raise ValueError('scale_contract must state the numeric dimensions and relative scale')
    beats = s.get('beats')
    if not isinstance(beats, list) or not beats:
        raise ValueError('Contract requires action beats covering the target timeline')
    seen = set()
    ranges = []
    for beat in beats:
        a, b = _item_interval(beat, total, 'beat')
        if beat['id'] in seen:
            raise ValueError('Beat ids must be unique')
        seen.add(beat['id'])
        _text(beat.get('action'), 'beat.action')
        if '<d' in beat['action'].lower():
            raise ValueError('Place complete dialogue in the top-level dialogues list')
        for key in ('camera', 'sound'):
            if key in beat:
                _text(beat[key], 'beat.' + key)
        if type(beat.get('indivisible', False)) is not bool:
            raise ValueError('beat.indivisible must be boolean')
        beat.setdefault('indivisible', False)
        ranges.append((a, b))
    cursor = 0
    for a, b in sorted(ranges):
        if a > cursor:
            raise ValueError(f'Action beats leave an uncovered interval starting at frame {cursor}')
        cursor = max(cursor, b)
    if cursor != total:
        raise ValueError('Action beats must cover the entire target timeline')
    dialogues = s.setdefault('dialogues', [])
    if not isinstance(dialogues, list):
        raise ValueError('dialogues must be a list of complete utterances')
    seen = set()
    for dialogue in dialogues:
        _item_interval(dialogue, total, 'dialogue')
        if dialogue['id'] in seen:
            raise ValueError('Dialogue ids must be unique')
        seen.add(dialogue['id'])
        _text(dialogue.get('text'), 'dialogue.text')
        if re.search(r'</?d(?:\s|>)', dialogue['text'], re.I):
            raise ValueError('dialogue.text must contain spoken words without dialogue markup')
        speaker = _text(dialogue.get('speaker'), 'dialogue.speaker')
        if not re.search(r'\bS[1-9]\d*\b', speaker) or not re.search(r'<Subject\s+[1-9]\d*>', speaker):
            raise ValueError('dialogue.speaker requires both Sx and <Subject N> identity markers')
        if 'language' in dialogue:
            _text(dialogue['language'], 'dialogue.language')
            if any(c in dialogue['language'] for c in '[]<>\n\r'):
                raise ValueError('dialogue.language must be a plain language name')
        if 'audio_slot' in dialogue:
            _integer(dialogue['audio_slot'], 'dialogue.audio_slot', 1)
    s['beats'] = sorted(beats, key=lambda b: (b['start_frame'], b['end_frame'], b['id']))
    s['dialogues'] = sorted(dialogues, key=lambda d: (d['start_frame'], d['end_frame'], d['id']))
    return s


def protected_intervals(contract):
    """Return the indivisible actions and complete dialogue intervals for planning."""
    s = _object(contract)
    if 'total_frames' not in s:
        raise ValueError('Validate the contract with a target total_frames before extracting protection')
    s = validate_contract(s, s['total_frames'])
    return sorted(set((item['start_frame'], item['end_frame'])
                      for item in s['dialogues'] + [b for b in s['beats'] if b['indivisible']]))


def validate_window(window):
    """Validate a physical window in isolation; also used by the AV adapter."""
    if not isinstance(window, dict):
        raise ValueError('Window must be an object')
    w = copy.deepcopy(window)
    for key in ('index', 'sample_frames', 'context_frames', 'new_start', 'new_end', 'endpoint',
                'source_endpoint', 'local_new_start', 'local_new_end'):
        _integer(w.get(key), 'window.' + key)
    video_steps(w['sample_frames'])
    if not _MIN_SAMPLE_FRAMES <= w['sample_frames'] <= 15 * FPS:
        raise ValueError('Physical sample length must be inside the native 4–15 second budget')
    a, b, c = w['new_start'], w['new_end'], w['context_frames']
    if not a < b or w['local_new_start'] != c or w['local_new_end'] != c + b - a:
        raise ValueError('Window new-content and local intervals disagree or make no progress')
    _shared(w['endpoint'], 'window.endpoint')
    if not w['local_new_end'] <= w['endpoint'] <= w['sample_frames']:
        raise ValueError('Window delivery extends beyond its generated endpoint or sample length')
    if c:
        _shared(c, 'window.context_frames')
        _shared(w['source_endpoint'], 'window.source_endpoint')
        if c > w['source_endpoint'] or c > a or w['index'] == 0:
            raise ValueError('Window history is longer than the available source or global past')
    elif a or w['index'] or w['source_endpoint']:
        raise ValueError('Only the first window may omit its protected AV history')
    return w


def _time(frame):
    return f'{frame / FPS:.3f}'


def compile_window(contract, window, incoming_state=''):
    """Compile six official Ref2VA sections and only this window's new content.

    Continuation requires a caller-supplied *observed* state from the accepted
    source. No contract ending state or hypothetical future is substituted.
    """
    raw = _object(contract)
    total = raw.get('total_frames')
    s = validate_contract(raw, total)
    w = validate_window(window)
    a, b, c = w['new_start'], w['new_end'], w['context_frames']
    if b > total:
        raise ValueError('Window extends beyond the contract')
    if isinstance(incoming_state, dict):
        incoming_state = json.dumps(incoming_state, ensure_ascii=False, sort_keys=True) if incoming_state else ''
    if c and (not isinstance(incoming_state, str) or not incoming_state.strip()):
        raise ValueError('Continuation requires the observed incoming_state from accepted clean AV')
    for x, y in protected_intervals(s):
        if x < a < y or x < b < y:
            raise ValueError('Window cuts an indivisible action or complete dialogue; re-plan at a shared AV endpoint')
    detail = [s['scale_contract'], 'All times below are local seconds at 24 fps; frame intervals are end-exclusive.']
    if c:
        detail.extend([
            f'Local 0.000–{_time(c)} seconds are existing protected audiovisual history. '
            'Continue the ongoing action phase and camera movement from its actual state.',
            'Observed incoming state: ' + incoming_state,
            'Inherit the same speaker identities from the actual audiovisual history. Preserve the '
            'contract (Sx) and <Subject N> assignments; audio slot numbers do not redefine speaker order.',
        ])
    else:
        detail.append('Establish the scene according to the first action and camera directions.')
    detail.append(f'New visible content occupies local {_time(c)}–{_time(w["local_new_end"])} seconds. '
                  'The physical window boundary creates no camera cut or new shot.')
    mapped = []
    for beat in s['beats']:
        x, y = max(a, beat['start_frame']), min(b, beat['end_frame'])
        if x >= y:
            continue
        local_a, local_b = c + x - a, c + y - a
        item = {'id': beat['id'], 'global_start': x, 'global_end': y,
                'local_start': local_a, 'local_end': local_b,
                'action': beat['action'], 'partial_start': x > beat['start_frame'],
                'partial_end': y < beat['end_frame']}
        phase = 'Continue the ongoing portion without replaying its onset: ' if item['partial_start'] else ''
        lines = [f'{_time(local_a)}–{_time(local_b)} seconds (local frames {local_a}–{local_b}): {phase}{beat["action"]}']
        for key in ('camera', 'sound'):
            if key in beat:
                item[key] = beat[key]
                lines.append(f'{key.capitalize()} in this interval: {beat[key]}')
        if item['partial_end']:
            lines.append('This action continues beyond the visible interval; keep its natural motion in progress.')
        detail.append('\n'.join(lines))
        mapped.append(item)
    owned = []
    for dialogue in s['dialogues']:
        x, y = dialogue['start_frame'], dialogue['end_frame']
        if x >= b or y <= a:
            continue
        if not a <= x < y <= b:
            raise ValueError('Window cuts a complete dialogue; re-plan before compilation')
        owned.append(dialogue['id'])
        voice = f' using <Audio {dialogue["audio_slot"]}> for voice timbre only' if 'audio_slot' in dialogue else ''
        language = f'[{dialogue["language"]}] ' if 'language' in dialogue else ''
        detail.append(f'{_time(c + x - a)}–{_time(c + y - a)} seconds: {dialogue["speaker"]}{voice} '
                      f'says exactly <d>{language}{dialogue["text"]}</d>')
    detail.append('Only the scheduled utterances in the new visible interval are newly spoken. '
                  'Do not replay historical dialogue or speak future lines in the unused tail. '
                  'Continue natural motion and sound through the unused generated remainder; do not loop or freeze to fill time.')
    sections = [('subject_definitions', s['subject_definitions']), ('summary', s['summary']),
                ('retention_analysis', s['retention_analysis']), ('detailed_description', '\n'.join(detail)),
                ('overall_soundscape', s['overall_soundscape']), ('non_diegetic_music', s['non_diegetic_music'])]
    return {'prompt': '\n\n'.join(name + ':\n' + text for name, text in sections),
            'dialogue_ids': owned, 'mapped_beats': mapped}
