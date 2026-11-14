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

from .long_video_references import (directing_text, reference_maps, remap_references,
                                    validate_reference_syntax, validate_selection, without_local_voice_bindings,
                                    validate_parent_reference)

FPS = 24
_MIN_SAMPLE_FRAMES = 107  # First 17k+5 length inside the native 4–15 second budget.
SECTIONS = ('subject_definitions', 'summary', 'retention_analysis', 'detailed_description',
            'overall_soundscape', 'non_diegetic_music')


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


def _known(item, keys, label):
    unknown = set(item) - set(keys)
    if unknown:
        raise ValueError(f'{label} has unknown fields: {", ".join(sorted(unknown))}')


def _phase_prose(item, label, start, end):
    _text(item.get('action'), label + '.action')
    if re.search(r'</?d(?:\s|>)', item['action'], re.I):
        raise ValueError('Place complete dialogue in the top-level dialogues list')
    for key in ('camera', 'sound'):
        if key in item:
            _text(item[key], label + '.' + key)
    if 'timed_actions' in item:
        actions = item['timed_actions']
        if not isinstance(actions, list) or not actions:
            raise ValueError(label + '.timed_actions must be a nonempty list')
        for action in actions:
            if not isinstance(action, dict):
                raise ValueError(label + '.timed_actions entries must be objects')
            _known(action, ('start_frame', 'end_frame', 'action', 'camera', 'sound'), label + '.timed_actions')
            a = _integer(action.get('start_frame'), label + '.timed_actions.start_frame')
            b = _integer(action.get('end_frame'), label + '.timed_actions.end_frame', 1)
            if not start <= a < b <= end:
                raise ValueError(label + '.timed_actions interval must lie inside its beat or remaining window')
            _phase_prose(action, label + '.timed_actions', a, b)


def _voice_policy(value):
    if not isinstance(value, str) or value not in {'inherit', 'dialogue_only'}:
        raise ValueError('voice_policy must be inherit or dialogue_only')
    return value


def _validate_overrides(s, total):
    if 'voice_policy' in s:
        _voice_policy(s['voice_policy'])
    if 'reference_selection' in s:
        s['reference_selection'] = validate_selection(s['reference_selection'])
    if 'parent_reference' in s:
        s['parent_reference'] = validate_parent_reference(s['parent_reference'])
        if s['parent_reference']['mode'] == 'append' and s.get('reference_selection') is not None:
            raise ValueError('parent_reference append cannot use explicit reference_selection; designate a selected Picture with replace')
    misplaced = {'forward_frames', 'remaining_beats', 'sections', 'anchors'} & set(s)
    if misplaced:
        raise ValueError('Unsupported top-level phase fields; use exact window_overrides: ' + ', '.join(sorted(misplaced)))
    overrides = s.get('window_overrides', [])
    if not isinstance(overrides, list):
        raise ValueError('window_overrides must be a list of exact new_start/new_end objects')
    intervals = []
    by_id = {beat['id']: beat for beat in s['beats']}
    for override in overrides:
        if not isinstance(override, dict):
            raise ValueError('window_overrides entries must be objects')
        _known(override, ('new_start', 'new_end', 'remaining_beats', 'sections',
                          'reference_selection', 'voice_policy', 'forward_frames', 'target_audio', 'parent_reference'), 'window_override')
        a = _integer(override.get('new_start'), 'window_override.new_start')
        b = _integer(override.get('new_end'), 'window_override.new_end', 1)
        if not a < b <= total:
            raise ValueError('window_override interval must lie inside the contract timeline')
        if any(a < y and x < b for x, y in intervals):
            raise ValueError('Duplicate or overlapping window_overrides are ambiguous')
        intervals.append((a, b))
        if len(override) == 2:
            raise ValueError('window_override requires an explicit phase, reference or forward override')
        if 'forward_frames' in override:
            forward = _integer(override['forward_frames'], 'window_override.forward_frames')
            if forward % 17 or forward >= b - a:
                raise ValueError('window_override.forward_frames must be 0 or a positive multiple of 17 shorter than its new interval')
        if 'voice_policy' in override:
            _voice_policy(override['voice_policy'])
        if 'reference_selection' in override:
            override['reference_selection'] = validate_selection(override['reference_selection'])
        if 'parent_reference' in override:
            override['parent_reference'] = validate_parent_reference(override['parent_reference'])
        parent_policy = override.get('parent_reference', s.get('parent_reference'))
        if (parent_policy and parent_policy['mode'] == 'append'
                and override.get('reference_selection', s.get('reference_selection')) is not None):
            raise ValueError('parent_reference append cannot use explicit reference_selection; designate a selected Picture with replace')
        if 'sections' in override:
            sections = override['sections']
            if not isinstance(sections, dict) or not sections:
                raise ValueError('window_override.sections must be a nonempty object')
            _known(sections, SECTIONS, 'window_override.sections')
            for key, value in sections.items():
                _text(value, 'window_override.sections.' + key)
        if 'remaining_beats' in override:
            remaining = override['remaining_beats']
            if not isinstance(remaining, list) or not remaining:
                raise ValueError('remaining_beats must be a nonempty list keyed by existing beat id')
            seen = set()
            for beat in remaining:
                if not isinstance(beat, dict):
                    raise ValueError('remaining_beats entries must be objects')
                _known(beat, ('id', 'action', 'camera', 'sound', 'timed_actions'), 'remaining_beat')
                identity = _text(beat.get('id'), 'remaining_beat.id')
                if identity in seen or identity not in by_id:
                    raise ValueError('remaining_beats ids must be unique known beat IDs')
                seen.add(identity)
                original = by_id[identity]
                x, y = max(a, original['start_frame']), min(b, original['end_frame'])
                if x >= y:
                    raise ValueError('remaining_beat does not intersect its exact override window')
                _phase_prose(beat, 'remaining_beat', x, y)
    return overrides


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
        _phase_prose(beat, 'beat', a, b)
        if '<d' in beat['action'].lower():
            raise ValueError('Place complete dialogue in the top-level dialogues list')
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
        if 'delivery' in dialogue:
            _text(dialogue['delivery'], 'dialogue.delivery')
            if any(char in dialogue['delivery'] for char in '<>\n\r'):
                raise ValueError('dialogue.delivery must be plain speech direction without markup')
        if 'audio_slot' in dialogue:
            _integer(dialogue['audio_slot'], 'dialogue.audio_slot', 1)
            if dialogue['audio_slot'] > 3:
                raise ValueError('dialogue.audio_slot must refer to an original Audio slot in 1..3')
    s['beats'] = sorted(beats, key=lambda b: (b['start_frame'], b['end_frame'], b['id']))
    s['dialogues'] = sorted(dialogues, key=lambda d: (d['start_frame'], d['end_frame'], d['id']))
    _validate_overrides(s, total)
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


def validate_window_overrides(contract, windows):
    """Validate every exact override against the actual plan before queueing.

    This preflight needs no hypothetical incoming AV state. Individual window
    compilation repeats its own exact-boundary check as a defense on resume.
    """
    raw = _object(contract)
    s = validate_contract(raw, raw.get('total_frames'))
    try:
        planned = [validate_window(window) for window in windows]
    except TypeError as exc:
        raise ValueError('windows must be an iterable of physical window objects') from exc
    pairs = {(window['new_start'], window['new_end']) for window in planned}
    if any(window['new_end'] > s['total_frames'] for window in planned):
        raise ValueError('Planned window extends beyond the contract')
    for override in s.get('window_overrides', []):
        if (override['new_start'], override['new_end']) not in pairs:
            raise ValueError('window_overrides must match exact planned new_start/new_end boundaries')
    for window in planned:
        override = next((item for item in s.get('window_overrides', [])
                         if (item['new_start'], item['new_end']) == (window['new_start'], window['new_end'])), {})
        policy = override.get('parent_reference', s.get('parent_reference'))
        if (window['context_frames'] and policy and policy['frame_policy'] == 'forward_last'
                and not override.get('forward_frames', 0)):
            raise ValueError('parent_reference forward_last requires an exact-window forward_frames range and actual forward review')
    return copy.deepcopy(s.get('window_overrides', []))


def _time(frame):
    return f'{frame / FPS:.3f}'


_CLOCK_PREFIX = r'\b(?:at|by|until|through)\s+|(?:在|于|到|至|截至|直到)\s*'
_CLOCK_NUMBER = r'-?\d+(?:\.\d+)?'
_CLOCK_UNIT = r'\s*(?:(?:seconds?|secs?|s)\b|秒(?:钟)?|时)'
_CLOCK = re.compile(
    # Paired range endpoints both need explicit time units. Bare from/to/and
    # numbers remain ordinary counts, dimensions or relative durations.
    r'(?P<range>(?P<range_prefix>\b(?:from|between)\s+|从\s*)'
    r'(?P<range_a>' + _CLOCK_NUMBER + r')(?P<range_unit_a>' + _CLOCK_UNIT + r')'
    r'(?P<range_join>\s+(?:to|and)\s+|\s*(?:到|至)\s*)'
    r'(?P<range_b>' + _CLOCK_NUMBER + r')(?P<range_unit_b>' + _CLOCK_UNIT + r'))|'
    r'(?P<frame>\{\{frame:(?P<frame_number>\d+)\}\})|'
    r'(?P<prefix>' + _CLOCK_PREFIX + r')(?P<number>' + _CLOCK_NUMBER + r')'
    r'(?P<unit>' + _CLOCK_UNIT + r')|'
    r'(?P<bare_prefix>' + _CLOCK_PREFIX + r')(?P<bare>-?\d+\.\d{3})'
    r'(?=\s*(?:[.,;:!?。；，！？\n]|$))(?!\.\d)', re.I)


def _local_clocks(text, start, end, offset):
    """Remap explicit global event clocks, never measurements or spoken words.

    Arbitrary prose is not semantically inferred. A known past/future event in
    active prose requires an explicit current-phase replacement instead.
    """
    def transform(part):
        if re.search(r'\{\{frame', _CLOCK.sub('', part), re.I):
            raise ValueError('Malformed frame clock; use {{frame:N}} with a nonnegative integer global frame')
        def local_time(frame):
            if not start <= frame <= end:
                raise ValueError('An explicit action clock lies outside the current phase; use remaining_beats '
                                 'or a current phase sections override instead of issuing completed/future actions')
            return f'{float((frame + offset) / FPS):.3f}'
        def clock(match):
            if match.group('range'):
                first = local_time(Fraction(Decimal(match.group('range_a'))) * FPS)
                last = local_time(Fraction(Decimal(match.group('range_b'))) * FPS)
                return (match.group('range_prefix') + first + match.group('range_unit_a')
                        + match.group('range_join') + last + match.group('range_unit_b'))
            if match.group('frame'):
                frame = Fraction(int(match.group('frame_number')))
            else:
                frame = Fraction(Decimal(match.group('number') or match.group('bare'))) * FPS
            local = local_time(frame)
            return (match.group('prefix') or match.group('bare_prefix') or '') + local + (match.group('unit') or '')
        return _CLOCK.sub(clock, part)
    return directing_text(text, transform)


def _mapped_phase(beat, x, y, offset, free_start, window_end):
    item = {'id': beat['id'], 'global_start': x, 'global_end': y,
            'local_start': offset + x, 'local_end': offset + y,
            'partial_start': x > beat['start_frame'], 'partial_end': y < beat['end_frame']}
    if 'timed_actions' in beat:
        timed = []
        for action in beat['timed_actions']:
            start, end = max(x, action['start_frame']), min(y, action['end_frame'])
            if start >= end:
                continue
            phase = {'global_start': start, 'global_end': end,
                     'local_start': offset + start, 'local_end': offset + end,
                     'partial_start': start > action['start_frame'],
                     'partial_end': end < action['end_frame']}
            for key in ('action', 'camera', 'sound'):
                if key in action:
                    text = action[key]
                elif key != 'action' and key in beat:
                    text = beat[key]
                else:
                    continue
                phase[key] = _local_clocks(text, free_start, window_end, offset)
            timed.append(phase)
        if not timed:
            return None
        timed.sort(key=lambda action: (action['global_start'], action['global_end']))
        end = max(action['global_end'] for action in timed)
        item.update(global_start=timed[0]['global_start'], global_end=end,
                    local_start=timed[0]['local_start'], local_end=offset + end, timed_actions=timed,
                    action='\n'.join(action['action'] for action in timed))
    else:
        for key in ('action', 'camera', 'sound'):
            if key in beat:
                item[key] = _local_clocks(beat[key], free_start, window_end, offset)
    return item


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
    override = None
    for candidate in s.get('window_overrides', []):
        if candidate['new_start'] < b and a < candidate['new_end']:
            if (candidate['new_start'], candidate['new_end']) != (a, b):
                raise ValueError('window_overrides must match exact planned new_start/new_end boundaries')
            override = candidate
    override = override or {}
    forward = override.get('forward_frames', 0)
    if forward and not c:
        raise ValueError('forward_frames requires a continuation with actual source history; compilation does not approve a source tail')
    parent_policy = override.get('parent_reference', s.get('parent_reference'))
    if c and parent_policy and parent_policy['frame_policy'] == 'forward_last' and not forward:
        raise ValueError('parent_reference forward_last requires an explicit forward_frames range and actual forward review')
    free_start, free_local, offset = a + forward, c + forward, c - a
    for x, y in protected_intervals(s):
        if x < free_start < y:
            raise ValueError('forward_frames cuts an indivisible action or complete dialogue; supply a safe remaining phase')
    selection = override.get('reference_selection', s.get('reference_selection'))
    selection = copy.deepcopy(selection)
    policy = override.get('voice_policy', s.get('voice_policy', 'inherit'))
    local_sections = override.get('sections', {})
    replacements = {beat['id']: beat for beat in override.get('remaining_beats', [])}
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
    if forward:
        detail.append(f'Local {_time(c)}–{_time(free_local)} seconds are supplied forward conditioning, '
                      'not free action instructions. This compiler does not authorize or approve their source tail.')
    detail.append(f'New visible content occupies local {_time(free_local)}–{_time(w["local_new_end"])} seconds. '
                  'The physical window boundary creates no camera cut or new shot.')
    if 'detailed_description' in local_sections:
        detail.append(_local_clocks(local_sections['detailed_description'], free_start, b, offset))
    mapped, timeline = [], []
    for beat in s['beats']:
        x, y = max(free_start, beat['start_frame']), min(b, beat['end_frame'])
        if x >= y:
            continue
        original = beat
        if beat['id'] in replacements:
            # Replace the complete prose group: omitted old camera/sound are
            # deliberately absent, so completed pickup/sip cues cannot leak.
            beat = {key: original[key] for key in ('id', 'start_frame', 'end_frame')}
            beat.update(replacements[original['id']])
        item = _mapped_phase(beat, x, y, offset, free_start, b)
        if item is None:
            continue
        parts = item.get('timed_actions', [item])
        for part in parts:
            lines = []
            local_a, local_b = part['local_start'], part['local_end']
            phase = 'Continue the ongoing portion without replaying its onset: ' if part['partial_start'] else ''
            lines.append(f'From 00:{local_a / FPS:06.3f} to 00:{local_b / FPS:06.3f}, {phase}{part["action"]}')
            for key in ('camera', 'sound'):
                if key in part:
                    lines.append(f'{key.capitalize()} in this interval: {part[key]}')
            if part['partial_end']:
                lines.append('This action continues beyond the visible interval; keep its natural motion in progress.')
            timeline.append((local_a, 0, '\n'.join(lines)))
        mapped.append(item)
    owned = []
    for dialogue in s['dialogues']:
        x, y = dialogue['start_frame'], dialogue['end_frame']
        if x >= b or y <= free_start:
            continue
        if not free_start <= x < y <= b:
            raise ValueError('Window cuts a complete dialogue; re-plan before compilation')
        owned.append(dialogue['id'])
        voice = (f' using <Audio {dialogue["audio_slot"]}> for voice timbre only'
                 if 'audio_slot' in dialogue and (selection is None or selection['audios']) else '')
        language = f'[{dialogue["language"]}] ' if 'language' in dialogue else ''
        delivery = _local_clocks(dialogue.get('delivery', 'says exactly'), free_start, b, offset)
        local_a, local_b = c + x - a, c + y - a
        timeline.append((local_a, 1,
                         f'From 00:{local_a / FPS:06.3f} to 00:{local_b / FPS:06.3f}, '
                         f'{dialogue["speaker"]}{voice} {delivery} <d>{language}{dialogue["text"]}</d>'))
    # Actions and utterances share one chronological lane. Appending all speech
    # after the final silent action caused H3 to delay otherwise valid dialogue.
    detail.extend(text for _, _, text in sorted(timeline, key=lambda item: item[:2]))
    detail.append('Only the scheduled utterances in the new visible interval are newly spoken. '
                  'Do not replay historical dialogue or speak future lines in the unused tail. '
                  'Continue natural motion and sound through the unused generated remainder; do not loop or freeze to fill time.')
    source_sections = {key: local_sections.get(key, s[key]) for key in SECTIONS if key != 'detailed_description'}
    if replacements and 'summary' not in local_sections:
        source_sections['summary'] = ('Perform only this window\'s current action phase and scheduled dialogue, '
                                      'as specified in detailed_description.')
    sections = {key: _local_clocks(text, free_start, b, offset) for key, text in source_sections.items()}
    sections['detailed_description'] = '\n'.join(detail)
    quiet = policy == 'dialogue_only' and not owned
    omit_voice = quiet or (selection is not None and not selection['audios'])
    if omit_voice:
        sections = {key: without_local_voice_bindings(text) for key, text in sections.items()}
    if quiet and selection is not None:
        selection['audios'] = []
    maps = reference_maps(selection) if selection is not None else None
    if maps is not None:
        sections = {key: remap_references(text, maps) for key, text in sections.items()}
    for item in mapped:
        for part in [item, *item.get('timed_actions', [])]:
            for key in ('action', 'camera', 'sound'):
                if key in part:
                    text = without_local_voice_bindings(part[key]) if omit_voice else part[key]
                    part[key] = remap_references(text, maps) if maps is not None else text
    audit = {
        'original_sections': {key: s[key] for key in SECTIONS if key in s},
        'original_beats': [copy.deepcopy(beat) for beat in s['beats'] if beat['start_frame'] < b and a < beat['end_frame']],
        'completed_beats': [copy.deepcopy(beat) for beat in s['beats'] if beat['end_frame'] <= a],
        'original_dialogues': [copy.deepcopy(line) for line in s['dialogues'] if line['start_frame'] < b and a < line['end_frame']],
        'window_override': copy.deepcopy(override) or None,
        'original_reference_selection': copy.deepcopy(override.get('reference_selection', s.get('reference_selection'))),
        'reference_maps': maps, 'prompt_reference_space': 'window' if maps is not None else 'original',
        'free_global_start': free_start, 'free_local_start': free_local,
    }
    prompt = '\n\n'.join(name + ':\n' + sections[name] for name in SECTIONS)
    validate_reference_syntax(prompt)
    result = {'prompt': prompt,
            'dialogue_ids': owned, 'mapped_beats': mapped, 'reference_selection': selection,
            'voice_policy': policy, 'forward_frames': forward, 'audit': audit}
    if parent_policy is not None:
        result['parent_reference'] = copy.deepcopy(parent_policy)
    target = override.get('target_audio', s.get('target_audio'))
    if target is not None:
        if not isinstance(target, dict):
            raise ValueError('target_audio must be an explicit complete audio recipe object')
        result['target_audio_recipe'] = copy.deepcopy(target)
    return result
