"""Editorial long videos made from the unchanged original short-video graph.

An authored cut, rather than a physical continuation window, resets generation.
No generated image, latent or audio from another clip is a conditioning input.
"""
from copy import deepcopy
import json
import math
import re

from .long_video_plan import FPS, SECTIONS, target_frame_count
from .long_video_prompt import (_sections, _ranges, _mask_speech, _SHOT, _identities,
                                _dialogue, _reference_counts)
from .long_video_references import validate_local_references, validate_reference_syntax
from .long_video_graph import ancestors, OUTPUT_TYPES
from .long_video_store import fingerprint

ENTRY_MODE = 'independent_shots'


def _clock(frames):
    return f'{frames / FPS:.6f}'.rstrip('0').rstrip('.')


def _opening_view(text):
    """Express the already-applied editorial cut as this clip's opening state."""
    text = re.sub(
        r'^(?:(?:a|the)\s+)?(?:(?:deliberate|hard|straight|match)\s+)?'
        r'cut\s+(?:back\s+)?to\s+',
        'The shot opens directly on ', text, count=1, flags=re.I)
    return re.sub(r'^(?:(?:镜头|画面)\s*)?(?:直接|立即|硬)?切(?:回(?:到)?|到|至)\s*',
                  '开场即为', text, count=1)


def _speakers(sections):
    """Sx follows this independent request's speech, without renumbering assets."""
    detail = sections['detailed_description']
    order = []
    for match in re.finditer(r'\(S([1-9]\d*)\)(?:(?!\(S[1-9]\d*\)).)*?<d>', detail, re.S):
        if match[1] not in order:
            order.append(match[1])
    declared = re.findall(r'\bS([1-9]\d*)\b', _mask_speech(sections['subject_definitions']))
    order.extend(value for value in dict.fromkeys(declared) if value not in order)
    mapping = {value: str(index + 1) for index, value in enumerate(order)}
    for name, value in sections.items():
        parts = re.split(r'(<d(?:\s+[^>]*)?>.*?</d\s*>)', value, flags=re.I | re.S)
        sections[name] = ''.join(part if index % 2 else re.sub(
            r'\bS([1-9]\d*)\b', lambda m: 'S' + mapping.get(m[1], m[1]), part)
            for index, part in enumerate(parts))
    return sections


def plan_shots(prompt, target_seconds, settings=None):
    """Pack whole authored shots into short requests; never invent a cut."""
    total = target_frame_count(target_seconds)
    sections = _sections(prompt)
    validate_reference_syntax(prompt)
    if settings is not None:
        validate_local_references(prompt, _reference_counts(settings))
    ranges = _ranges(sections['detailed_description'], total)
    identities = _identities(sections['subject_definitions'])
    cuts = [0]
    for index, (a, b, text) in enumerate(ranges):
        # Validate speech before queueing, including complete per-interval lines.
        _dialogue(text, a, b, identities, f'line_{index + 1}')
        labels = list(_SHOT.finditer(_mask_speech(text)))
        if len(labels) > 1 or labels and _mask_speech(text)[:labels[0].start()].strip():
            raise ValueError('分镜式长视频：请把每个 [Shot N] 放在切镜时间段之前。')
        if labels and a:
            cuts.append(a)
    cuts.append(total)
    # Iterative reachability handles arbitrarily many shots without a recursion
    # depth limit. Prefer the longest valid clip while keeping the tail valid.
    following = {total: None}
    for a in reversed(cuts[:-1]):
        candidates = [b for b in cuts if 4 * FPS <= b - a <= 15 * FPS and b in following]
        if candidates:
            following[a] = max(candidates)
    if 0 not in following:
        raise ValueError('分镜式长视频：请在剧情的自然切镜处安排 4–15 秒的完整片段；'
                         '单镜超过 15 秒时请增加有叙事依据的 [Shot N]，或使用连续镜头续拍。')
    result, a = [], 0
    while a < total:
        b = following[a]
        detail, dialogue = [], []
        for start, end, text in ranges:
            if not a <= start < b:
                continue
            if end > b:
                raise ValueError('切镜不能截断同一个动作时间段或完整台词。')
            label = _SHOT.match(text)
            if label:
                detail.append(label.group())
                text = text[label.end():].strip()
            if start == a and a:
                text = _opening_view(text)
            detail.append(f'[{_clock(start-a)}-{_clock(end-a)}s] {text}')
            dialogue.extend(re.findall(r'<d(?:\s+[^>]*)?>(.*?)</d\s*>', text, re.I | re.S))
        local = _speakers({**sections, 'detailed_description': '\n'.join(detail)})
        seconds = math.ceil((b-a) / FPS)
        frames = seconds * FPS
        result.append({'kind': ENTRY_MODE, 'index': len(result), 'new_start': a, 'new_end': b,
            'context_frames': 0, 'source_endpoint': 0, 'local_new_start': 0, 'local_new_end': b-a,
            'sample_frames': frames + ((5-frames) % 17), 'duration': seconds,
            'prompt': '\n\n'.join(f'{key}:\n{local[key]}' for key in SECTIONS),
            'dialogue': dialogue})
        a = b
    return result


def automatic_shots(base, director, seconds):
    """Select only explicit editorial stories; preserve continuous/short routes."""
    if seconds <= 15:
        return None
    graph = deepcopy(base)
    settings = graph[director]['inputs']
    text = settings.get('prompt', '')
    if len(_SHOT.findall(_mask_speech(text))) < 2:
        return None
    if settings.get('mode', 'REF2VA') != 'REF2VA':
        raise ValueError('分镜式长视频请使用 REF2VA 的普通人物/场景参考；'
                         'FL2VA 的全片硬首尾帧不能自动充当每个独立镜头的首尾帧。')
    if settings.get('motion_smoothing', 'off') not in {'off', 'auto'}:
        raise ValueError('分镜式长视频使用原生 24FPS，请关闭插帧后排队。')
    outputs = [key for key, node in graph.items() if node.get('class_type') in OUTPUT_TYPES]
    if len(outputs) != 1:
        raise ValueError('分镜式长视频需要一个原 U11 视频输出。')
    output = graph[outputs[0]]['inputs']
    if (float(output.get('frame_rate', 24)) != 24 or output.get('pingpong', False)
            or output.get('crop_to_audio', False) or not output.get('save_output', True)):
        raise ValueError('分镜式长视频需要保存 24FPS 原时长视频；请关闭往返播放和按音频裁切。')
    from .long_video_media import resolve_encoding
    resolve_encoding(output)
    if settings.get('voice_mode', 'none') == 'none':
        from .prompting import without_voice_references
        text = settings['prompt'] = without_voice_references(text)
        for field in list(settings):
            if field.startswith('voice_reference_audio'):
                settings.pop(field)
    plan = plan_shots(text, seconds, settings)
    graph[director].setdefault('_meta', {})['h3_long_video_entry'] = ENTRY_MODE
    project = 'u11-shots-' + fingerprint({'graph': graph, 'seconds': seconds})[:20]
    return graph, {'enabled': True, 'target_seconds': seconds, 'project_name': project,
                   'stop_condition': 'target', 'entry_mode': ENTRY_MODE, 'shot_plan': plan}


def build_shot_graph(base, request):
    """Copy the actual short graph, changing only this clip's text/clock/seed/output."""
    graph = deepcopy(base)
    output = next(key for key, node in graph.items() if node.get('class_type') in OUTPUT_TYPES)
    graph = ancestors(graph, output)
    directors = [node for node in graph.values() if node['class_type'] == 'MiniMaxH3DirectorPlus']
    if len(directors) != 1:
        raise ValueError('分镜式长视频需要一个原导演台。')
    settings = directors[0]['inputs']
    settings.update(prompt=request['prompt'], duration=request['window']['duration'], seed=request['seed'])
    if settings.get('voice_mode') == 'fish_lock':
        settings['target_dialogue'] = '\n'.join(request['window']['dialogue'])
    for number in range(1, 4):
        settings.setdefault(f'voice_reference_name_{number}', '')
    for node in graph.values():
        if node['class_type'] == 'RandomNoise':
            node['inputs']['noise_seed'] = request['seed']
    final = graph[output]
    final['class_type'] = 'MiniMaxH3LongVideoShotOutput'
    final['inputs'].pop('long_video', None)
    final['inputs'].update(request_json=json.dumps(request, ensure_ascii=False, allow_nan=False),
        filename_prefix=f"h3_native_av/{request['project_name']}/takes/{request['take_id']}/shot")
    return graph
