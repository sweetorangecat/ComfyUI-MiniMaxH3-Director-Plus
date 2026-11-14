"""Compile the original Director's full-film H3 prose without loading any media.

This adapter only translates explicit authoring facts. Physical AV windows,
observed incoming state, file loading and media review remain runtime concerns.
"""
from __future__ import annotations

import re
from decimal import Decimal, ROUND_HALF_UP

from .long_video_plan import (FPS, SECTIONS, plan_windows, protected_intervals,
                              target_frame_count, validate_contract)
from .long_video_references import (AUDIO_SLOTS, PICTURE_SLOTS,
                                    validate_local_references, validate_reference_syntax)


_HEADINGS = re.compile(r'^[ \t]*(?:#{1,6}[ \t]+)?(' + '|'.join(SECTIONS) + r')[ \t]*[:：][ \t]*', re.I | re.M)
_SPEECH = re.compile(r'<d(?:\s+[^>]*)?>(.*?)</d\s*>', re.I | re.S)
_SPEECH_TAG = re.compile(r'</?d(?:\s|>)', re.I)
_SUBJECT = re.compile(r'<Subject\s+([1-9]\d*)>')
_SPEAKER = re.compile(r'\bS([1-9]\d*)\b')
_AUDIO = re.compile(r'<Audio\s+([1-9]\d*)>')
_CLOCK = r'\d+(?::\d{1,2}){0,2}(?:\.\d+)?'
_UNIT = r'(?:s(?:ec(?:ond)?s?)?|秒)'
_RANGE_BODY = (r'(?P<a>' + _CLOCK + r')[ \t]*(?:' + _UNIT + r')?[ \t]*[-–—][ \t]*'
               r'(?P<b>' + _CLOCK + r')[ \t]*(?:' + _UNIT + r')?')
_BRACKET_RANGE = re.compile(r'\[[ \t]*' + _RANGE_BODY + r'[ \t]*\]', re.I)
_BARE_RANGE = re.compile(r'^[ \t]*' + _RANGE_BODY + r'(?=[ \t]|$)', re.I | re.M)
_RANGE_HINT = re.compile(r'\[[ \t]*-?\d[^\]\n]*[-–—][^\]\n]*(?:\]|$)', re.M)
_BARE_HINT = re.compile(r'^[ \t]*\d+:\d[^\n]*[-–—]', re.M)
_SHOT = re.compile(r'\[Shot\s+[1-9]\d*\]', re.I)
_SEQUENCE = re.compile(r'\b(?:then|afterwards?|finally|next)\b|随后|然后|接着|最后|先[^。\n]*再', re.I)
# These are ambiguity guards, not a semantic event/timing inference engine.
# Finite interactions need author-supplied phases even without words like "then".
_FINITE_ACTION = re.compile(
    r'\b(?:pick(?:s|ed|ing)?\s+up|put(?:s|ting)?|plac(?:e[sd]?|ing)|tak(?:e[sn]?|ing)|took|'
    r'drink(?:s|ing)?|drank|sipp?(?:s|ed|ing)?|eat(?:s|ing)?|ate|swallow(?:s|ed|ing)?|'
    r'lift(?:s|ed|ing)?|rais(?:e[sd]?|ing)|lower(?:s|ed|ing)?|pour(?:s|ed|ing)?|'
    r'insert(?:s|ed|ing)?|remov(?:e[sd]?|ing)|press(?:es|ed|ing)?|touch(?:es|ed|ing)?|'
    r'grab(?:s|bed|bing)?|releas(?:e[sd]?|ing)|fold(?:s|ed|ing)?|drop(?:s|ped|ping)?|'
    r'open(?:s|ed|ing)|clos(?:es|ed|ing)|turn(?:s|ed|ing)|start(?:s|ed|ing)|stop(?:s|ped|ping)|'
    r'arriv(?:e[sd]?|ing)|leav(?:e[sd]?|ing)|enter(?:s|ed|ing)?|exit(?:s|ed|ing)?|'
    r'finish(?:es|ed|ing)?|complet(?:e[sd]?|ing)|execut(?:e[sd]?|ing)|discover(?:s|ed|ing)?)\b|'
    r'\b(?:open|close|turn|start|stop)\b(?=\s+(?:the|a|an|her|his|their)\b)|'
    r'\b(?:sit\s+down|stand\s+up|set(?:s|ting)?\s+down)\b|'
    r'拿起|拾起|取出|放下|放回|放到|举起|抬起|伸手|接过|交给|递给|插入|收起|'
    r'走出|走进|站起|坐下|起身|转身|打开|关上|按下|喝|吃|吞下|松开|停下|开始|结束', re.I)
_SUSTAINED_SCENE = re.compile(
    r'\b(?:walk(?:s|ing)?|run(?:s|ning)?|stand(?:s|ing)?|sit(?:s|ting)?|wait(?:s|ing)?|'
    r'hold(?:s|ing)?|breath(?:es?|ing)|track(?:s|ing)?|rain(?:s|ing)?|snow(?:s|ing)?|'
    r'flow(?:s|ing)?|glow(?:s|ing)?|static|motionless|fixed|stationary|unchanging)\b|'
    r'行走|走动|跟拍|站立|静立|坐着|等待|手持|握着|呼吸|静态|静景|雨景|雪景', re.I)
_UNTIMED_CLAUSES = re.compile(r'[,;，；]+|\s+(?:and|while|as)\s+|并且|同时|一边|并(?!排|肩)', re.I)
_UNKNOWN_DYNAMIC = re.compile(
    r'\b(?P<verb>[a-z]+(?:s|ed|ing))\s+(?:a|an|the|her|his|its|their|one|another)\b', re.I)
_QUOTED_SPEECH = re.compile(
    r'(?:\b(?:says?|said|speaks?|whispers?|shouts?|asks?|replies|narrates?)\b|说|问|回答|喊|旁白|对白|台词)'
    r'[^\n。.!?！？]{0,35}[:：]?[ \t]*[“"‘\']|\bS[1-9]\d*\b[^\n]{0,35}[:：][ \t]*[“"‘\']|'
    r'(?:\b(?:says?|said|speaks?|whispers?|shouts?|asks?|replies|narrates?|dialogue|narration|voice[- ]?over)\b|'
    r'说|问|回答|喊|旁白|对白|台词)[ \t]*[:：][ \t]*\S', re.I)
_UNMARKED_SPEECH = re.compile(_QUOTED_SPEECH.pattern + r'|\(?\bS[1-9]\d*\b\)?[ \t]*[:：][ \t]*\S', re.I)
_SPEECH_CUE = re.compile(
    r'(?:\b(?:says?|said|speaks?|whispers?|shouts?|asks?|replies|narrates?)'
    r'\b[^<>\n.!?;:：]*|(?:轻声|低声|大声)?(?:说|问|回答|喊|旁白))'
    r'[ \t]*[:：]?[ \t]*$', re.I)


def _error(message):
    return ValueError('长视频提示词：' + message)


def _mask_speech(text):
    """Keep offsets and line boundaries while protecting literal spoken content."""
    def mask(match):
        return ''.join('\n' if char == '\n' else ' ' for char in match.group())
    masked = _SPEECH.sub(mask, text)
    if _SPEECH_TAG.search(masked):
        raise _error('请用配对的 <d>...</d> 标记每句完整台词。')
    return masked


def _has_finite_events(text):
    # Negative constraints do not schedule the prohibited action. Keep this
    # deliberately clause-local, rather than guessing at a sentence's meaning.
    active = re.sub(r'\b(?:do not|don\'t|never)\b[^.;!?\n]*|(?:不要|不得|禁止)[^。；！？\n]*', '', text, flags=re.I)
    # Unknown inflected predicates with an object cannot borrow a static camera
    # cue elsewhere in the clause. Closed-class prepositions/copulas are not
    # action predicates; recognised sustained verbs remain supported.
    ambiguous = False
    for match in _UNKNOWN_DYNAMIC.finditer(active):
        word = match['verb'].lower()
        if word in {'is', 'was', 'has', 'as', 'across', 'towards', 'during', 'following', 'using'}:
            continue
        if _SUSTAINED_SCENE.fullmatch(word) and word not in {'fixed', 'static', 'motionless', 'stationary', 'unchanging'}:
            continue
        ambiguous = True
        break
    return bool(_SEQUENCE.search(active) or _FINITE_ACTION.search(active) or ambiguous)


def _continuous_scene(text):
    # Positive sustained/static cues are required. Unknown prose and multi-step
    # descriptions get a format request, not an invented 0..total action chain.
    sentences = [part for part in re.split(r'[.!?。！？]+(?:\s|$)', text.strip()) if part.strip()]
    clauses = [part.strip() for sentence in sentences for part in _UNTIMED_CLAUSES.split(sentence) if part.strip()]
    return bool(clauses) and not _has_finite_events(text) and all(_SUSTAINED_SCENE.search(part) for part in clauses)


def _sections(prompt):
    if not isinstance(prompt, str) or not prompt.strip():
        raise _error('请在原提示词框填写场景描述或完整 H3 六段正文。')
    masked = _mask_speech(prompt)
    headings = list(_HEADINGS.finditer(masked))
    names = [match.group(1).lower() for match in headings]
    if not headings:
        return {
            'subject_definitions': 'Use only the subjects and reference roles specified in the scene description.',
            'summary': 'A continuous scene described in detailed_description.',
            'retention_analysis': 'Preserve the supplied identities, scene, reference roles and relative proportions.',
            'detailed_description': prompt.strip(),
            'overall_soundscape': 'Use only the ambience and action sounds called for by the description; no additional speech.',
            'non_diegetic_music': 'N/A',
        }
    missing = [key for key in SECTIONS if key not in names]
    if missing:
        raise _error('H3 六段正文缺少字段：' + ', '.join(missing) + '。')
    if len(names) != len(set(names)):
        raise _error('H3 六段字段各写一次，删除重复字段。')
    if masked[:headings[0].start()].strip():
        raise _error('请把场景描述放入对应 H3 字段，从六段字段名开始。')
    result = {}
    for index, match in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(prompt)
        value = prompt[match.end():end].strip()
        if not value:
            raise _error(f'{names[index]} 为空；请填写说明，无配乐可写 N/A。')
        if names[index] != 'detailed_description':
            if _SPEECH.search(value) or _QUOTED_SPEECH.search(value):
                raise _error('请把完整台词放进 detailed_description 的全片时间段，其他字段只写稳定约束。')
            if _BRACKET_RANGE.search(value) or _BARE_RANGE.search(value):
                raise _error('请把带时间的事件放进 detailed_description；其他字段只写稳定身份、环境或风格。')
            if names[index] == 'summary' and _has_finite_events(value):
                raise _error('summary 含有先后动作或有限事件；请把这些内容移入 detailed_description 并标明全片时间，保留 summary 中的稳定视觉约束。')
        result[names[index]] = value
    return result


def _frame(clock):
    parts = clock.split(':')
    if len(parts) > 1 and any(Decimal(part) >= 60 for part in parts[1:]):
        raise _error(f'时间戳 {clock} 无效；冒号后的分、秒必须小于 60。')
    seconds = Decimal(0)
    for part in parts:
        seconds = seconds * 60 + Decimal(part)
    return int((seconds * FPS).to_integral_value(rounding=ROUND_HALF_UP))


def _ranges(detail, total):
    masked = _mask_speech(detail)
    matches = list(_BRACKET_RANGE.finditer(masked))
    for match in _BARE_RANGE.finditer(masked):
        # A bare dimension/count is not a timestamp. Bare clocks require a colon
        # or an explicit time unit; bracketed film ranges are unambiguous.
        if ':' in match.group() or re.search(_UNIT + r'\s*$', match.group(), re.I):
            matches.append(match)
    matches.sort(key=lambda match: match.start())
    for hint in _RANGE_HINT.finditer(masked):
        if not any(match.start() <= hint.start() < match.end() for match in matches):
            raise _error('时间段格式无效；请用 [0-8s]、[00:08-00:16] 或 [16-30秒]。')
    for hint in _BARE_HINT.finditer(masked):
        if not any(match.start() <= hint.start() < match.end() for match in matches):
            raise _error('时间戳格式无效；请用 [00:00-00:08] 标明完整全片时间段。')
    if not matches:
        if _SPEECH.search(detail) or _UNMARKED_SPEECH.search(detail):
            raise _error('请在 detailed_description 中为每句完整 <d>台词</d> 标明全片起止时间。')
        if not _continuous_scene(detail):
            raise _error('无时间段仅适用于明确的单一持续场景；拿取、吃喝、放回等有限动作或不明确的变化，请在原提示词中分阶段标明全片时间。')
        return [(0, total, detail.strip())]
    # A Shot label immediately before a range belongs to that range's cut,
    # including when the label occupies its own line.
    entries = []
    for match in matches:
        shot = re.search(r'(\[Shot\s+[1-9]\d*\])\s*$', masked[:match.start()], re.I)
        entries.append((match, shot.start() if shot else match.start(), shot.group(1) if shot else ''))
    prefix = detail[:entries[0][1]].strip()
    if _SPEECH.search(prefix) or _UNMARKED_SPEECH.search(prefix):
        raise _error('请把未标时间的台词放入 detailed_description 的完整全片时间段。')
    if _has_finite_events(prefix):
        raise _error('请把先后发生的动作阶段写入各自的全片时间段。')
    result = []
    cursor = 0
    for index, (match, _, shot) in enumerate(entries):
        a, b = _frame(match.group('a')), _frame(match.group('b'))
        if a >= b:
            raise _error('每个时间段的终点须晚于起点，且至少包含一帧（24 fps）。')
        if b > total:
            raise _error('时间段超出导演台设定的目标总时长，请修正全片终点。')
        if a < cursor:
            raise _error('时间段重叠或倒序；请按全片时间顺序连续排列。')
        if a > cursor:
            raise _error(f'时间轴在 {cursor / FPS:g} 秒处有空缺；请从 0 秒起连续覆盖全片。')
        end = entries[index + 1][1] if index + 1 < len(entries) else len(detail)
        text = detail[match.end():end].strip()
        if shot:
            text = shot + '\n' + text
        if index == 0 and prefix:
            text = prefix + '\n' + text
        if not text:
            raise _error('每个时间段都需填写动作描述或一句完整台词，不能留空。')
        result.append((a, b, text))
        cursor = b
    if cursor != total:
        raise _error(f'时间轴未覆盖目标终点 {total / FPS:g} 秒，请补全末段。')
    return result


def _identities(subjects):
    """Read declared mappings; never derive Sx from a Subject or Audio number."""
    result = []
    for clause in re.split(r'\n|(?<=;)\s*|(?<=；)\s*', subjects):
        subject, speaker = list(_SUBJECT.finditer(clause)), list(_SPEAKER.finditer(clause))
        if len(subject) == len(speaker) == 1:
            first, last = sorted([subject[0], speaker[0]], key=lambda match: match.start())
            # Include parentheses belonging to the Sx token without altering it.
            start = first.start() - (first is speaker[0] and first.start() > 0 and clause[first.start() - 1] == '(')
            end = last.end() + (last is speaker[0] and clause[last.end():last.end() + 1] == ')')
            audios = _AUDIO.findall(clause)
            result.append({'subject': subject[0].group(1), 'speaker_id': speaker[0].group(1),
                           'speaker': clause[start:end], 'audio_slot': int(audios[0]) if len(set(audios)) == 1 else None})
    return result


def _dialogue(text, start, end, identities, identity):
    matches = list(_SPEECH.finditer(text))
    if len(matches) > 1:
        raise _error('请为每句完整台词分别标明全片时间段，避免猜测多句之间的分配。')
    if not matches:
        if _UNMARKED_SPEECH.search(text):
            raise _error('请在对应全片时间段中用 <d>完整台词</d> 标记发言。')
        return text, None
    match = matches[0]
    cue = text[:match.start()]
    # Identity on the same line is authoritative. A declared mapping can supply
    # the omitted half when the author uses only (Sx) or only <Subject N>.
    local = cue.rsplit('\n', 1)[-1]
    subjects, speakers = list(_SUBJECT.finditer(local)), list(_SPEAKER.finditer(local))
    if len(subjects) > 1 or len(speakers) > 1:
        raise _error('每句台词旁只标一位发声者，使用固定的 <Subject N> 和 (Sx)。')
    subject = subjects[0].group(1) if subjects else None
    speaker_id = speakers[0].group(1) if speakers else None
    declared = [item for item in identities
                if (subject is None or item['subject'] == subject)
                and (speaker_id is None or item['speaker_id'] == speaker_id)] if subject or speaker_id else []
    if subjects and speakers:
        first, last = sorted([subjects[0], speakers[0]], key=lambda item: item.start())
        a = first.start() - (first is speakers[0] and first.start() > 0 and local[first.start() - 1] == '(')
        b = last.end() + (last is speakers[0] and local[last.end():last.end() + 1] == ')')
        speaker = local[a:b]
        # Prevent a cue from silently contradicting its declared identity.
        if any(item['speaker_id'] == speaker_id and item['subject'] != subject
               or item['subject'] == subject and item['speaker_id'] != speaker_id for item in identities):
            raise _error('台词的 Subject/Sx 归属与 subject_definitions 冲突，请保留原身份映射。')
    elif len(declared) == 1:
        speaker = declared[0]['speaker']
    else:
        raise _error('台词须有 <Subject N> 和 (Sx) 发声者标记；可在 subject_definitions 声明映射。')
    words = match.group(1)
    language = re.match(r'^\[([^\[\]<>\r\n]+)\][ \t]?', words)
    result = {'id': identity, 'start_frame': start, 'end_frame': end,
              'speaker': speaker, 'text': words[language.end():] if language else words}
    if not result['text'].strip() or _SPEECH_TAG.search(result['text']):
        raise _error('每个 <d>...</d> 必须包含一句非空的完整台词。')
    if language:
        result['language'] = language.group(1)
    audios = _AUDIO.findall(local)
    if len(set(audios)) > 1:
        raise _error('每句台词最多绑定一个原有 <Audio N> 音色参考。')
    audio = int(audios[0]) if audios else declared[0]['audio_slot'] if len(declared) == 1 else None
    if audio is not None:
        result['audio_slot'] = audio
    delivery = _SPEECH_CUE.search(cue)
    if delivery:
        # Keep the author's language and delivery next to the spoken words.
        # Leaving a dangling "says naturally in Mandarin:" in the action lane
        # asks the model to speak before it encounters the actual utterance.
        result['delivery'] = delivery.group().strip()
    action = (cue[:delivery.start()] if delivery else cue) + text[match.end():]
    action = action.strip()
    # No new pose, silence, camera or contact is invented for a dialogue-only
    # block; its identity cue is harmless prose alongside the scheduled line.
    return action or speaker, result


def _reference_counts(settings):
    if not isinstance(settings, dict):
        raise _error('原导演台设置格式无效。')
    def selected(field):
        value, filename = settings.get(field), settings.get(field + '_file')
        if value is not None and value != '':
            if (not isinstance(value, list) or len(value) != 2 or not isinstance(value[0], str)
                    or type(value[1]) is not int or value[1] < 0):
                raise _error(f'{field} 须使用原导演台已选文件或有效图连接。')
            return True
        if filename is not None and not isinstance(filename, str):
            raise _error(f'{field}_file 须为原导演台已选文件名。')
        return bool(filename and filename.strip())
    images = [selected(field) for field in PICTURE_SLOTS]
    audios = [selected(field) for field in AUDIO_SLOTS] if settings.get('voice_mode', 'none') != 'none' else []
    for kind, values in (('image', images[2:]), ('audio', audios)):
        if any(values) and not all(values[:max(i for i, value in enumerate(values) if value) + 1]):
            raise _error(f'原有 {kind} 槽位中间有空缺，请使用连续的原导演台输入。')
    return {'pictures': sum(images), 'audios': sum(audios)}


def contract_from_prompt(prompt, target_seconds, settings=None):
    """Return validated schema-v1 from one normal, globally timed H3 prompt.

    ``settings`` is the original Director inputs, with already selected filename
    and/or graph-link values. It only checks existing reference counts; it does
    not resolve a graph, open files, allocate slots, remap IDs or approve assets.
    """
    try:
        total = target_frame_count(target_seconds)
        sections = _sections(prompt)
        validate_reference_syntax(prompt)
        if settings is not None:
            validate_local_references(prompt, _reference_counts(settings))
        identities = _identities(sections['subject_definitions'])
        beats, dialogues = [], []
        for index, (start, end, text) in enumerate(_ranges(sections['detailed_description'], total)):
            action, line = _dialogue(text, start, end, identities, f'line_{index + 1:04d}')
            beats.append({'id': f'phase_{index + 1:04d}', 'start_frame': start, 'end_frame': end,
                          'action': action, 'indivisible': False})
            if line is not None:
                dialogues.append(line)
        contract = {key: value for key, value in sections.items() if key != 'detailed_description'}
        contract.update(schema_version=1, fps=FPS, total_frames=total, beats=beats, dialogues=dialogues,
                        source_prompt_sections={key: value for key, value in sections.items() if key != 'detailed_description'},
                        scale_contract='Scale instruction: preserve reference-relative proportions 1:1, including perspective; '
                                       'retain any explicitly supplied size constraints without adding physical dimensions.')
        contract = validate_contract(contract, total)
        try:
            windows = plan_windows(total, protected_intervals=protected_intervals(contract))
        except ValueError as exc:
            raise _error('完整台词的时间段无法放入原生续接窗口；请标出更准确的发声起止，或整句移到可容纳的全片时间段。') from exc
        overrides = []
        for window in windows:
            remaining = []
            for beat in contract['beats']:
                if beat['start_frame'] < window['new_start'] < beat['end_frame'] and _SHOT.search(beat['action']):
                    remaining.append({'id': beat['id'], 'action': _SHOT.sub('', beat['action']).strip()})
            if remaining:
                overrides.append({'new_start': window['new_start'], 'new_end': window['new_end'],
                                  'remaining_beats': remaining, 'sections': {'summary': contract['summary']}})
        if overrides:
            contract['window_overrides'] = overrides
            contract = validate_contract(contract, total)
        return contract
    except ValueError as exc:
        if str(exc).startswith('长视频提示词：'):
            raise
        marker = re.search(r'<(?:Picture|Audio)\s+\d+>', str(exc))
        if marker:
            raise _error(f'{marker.group()} 没有对应的原导演台输入或标记无效，请核对原有素材编号。') from exc
        raise _error('时间、身份或参考格式无效：' + str(exc)) from exc
