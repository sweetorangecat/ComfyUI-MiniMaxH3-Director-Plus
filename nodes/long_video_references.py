"""Detached, text-only reference policies for U11 physical windows.

IDs in contracts refer to the original Director's actual flattened media order.
No asset loading, approval, speaker renumbering, or process-global state lives here.
"""
from copy import deepcopy
import re


PICTURE_SLOTS = ('first_image', 'last_image', *(f'reference_image_{i}' for i in range(1, 10)))
AUDIO_SLOTS = ('voice_reference_audio', 'voice_reference_audio_2', 'voice_reference_audio_3')
SOURCE_FILES_META = 'h3_long_video_reference_source_files'
_SPEECH = re.compile(r'(<d(?:\s+[^>]*)?>.*?</d>)', re.I | re.S)
_MARKER = re.compile(r'<(Picture|Audio)\s+([1-9]\d*)>', re.I)


def _check_marker_syntax(part):
    for match in re.finditer(r'</?(?:Picture|Audio)', part, re.I):
        end = part.find('>', match.start())
        candidate = part[match.start():end + 1] if end >= 0 else part[match.start():]
        if end < 0 or not _MARKER.fullmatch(candidate):
            raise ValueError(f'Malformed directing reference marker: {candidate}')


def directing_text(text, transform):
    """Apply a text transform outside literal spoken markup only."""
    return ''.join(part if i % 2 else transform(part)
                   for i, part in enumerate(_SPEECH.split(text)))


def validate_reference_syntax(text):
    def check(part):
        _check_marker_syntax(part)
        return part
    directing_text(text, check)


def validate_selection(value):
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {'pictures', 'audios'}:
        raise ValueError('reference_selection requires only pictures and audios lists')
    result = {}
    for key, maximum in (('pictures', 11), ('audios', 3)):
        ids = value[key]
        if (not isinstance(ids, list) or any(type(i) is not int or not 1 <= i <= maximum for i in ids)
                or len(ids) != len(set(ids))):
            raise ValueError(f'reference_selection.{key} must be unique ordered original IDs in 1..{maximum}')
        result[key] = list(ids)
    return result


def reference_maps(selection):
    return {key: {str(original): local for local, original in enumerate(ids, 1)}
            for key, ids in selection.items()}


def remap_references(text, maps):
    def convert(part):
        _check_marker_syntax(part)
        def marker(match):
            kind, number = match.group(1).capitalize(), match.group(2)
            key = 'pictures' if kind == 'Picture' else 'audios'
            if number not in maps[key]:
                raise ValueError(f'Dangling <{kind} {number}>; select its original binding or supply local section prose')
            return f'<{kind} {maps[key][number]}>'
        return _MARKER.sub(marker, part)
    return directing_text(text, convert)


def validate_local_references(text, counts):
    def check(part):
        _check_marker_syntax(part)
        for match in _MARKER.finditer(part):
            key = 'pictures' if match.group(1).lower() == 'picture' else 'audios'
            if int(match.group(2)) > counts[key]:
                raise ValueError(f'Dangling {match.group(0)} has no actual local binding')
        return part
    directing_text(text, check)


def without_local_voice_bindings(text):
    """Remove known reference-only clauses without inventing a replacement voice.

    Mixed identity/binding prose is intentionally rejected. The caller can supply
    an exact-window subject_definitions override that states identity separately.
    """
    def clean(part):
        _check_marker_syntax(part)
        # The compiler's own binding suffix is explicit and independent of
        # identity and utterance text. Remove it without dropping either.
        part = re.sub(r'\s+using\s+<Audio\s+\d+>\s+for\s+voice\s+timbre\s+only', '', part, flags=re.I)
        # Zero-width splits retain all original spaces and line boundaries.
        clauses = re.split(r'(?<=[;；。!?！？])|(?<=\.)(?=\s+[A-Z<])|(?<=\n)', part)
        kept = []
        for clause in clauses:
            if not clause.strip():
                kept.append(clause)
                continue
            if re.search(r'<Audio\s+\d+>', clause, re.I):
                # A separate Audio declaration is safe to remove only when the
                # rest is an explicit reference/timbre declaration.
                words = re.sub(r'<Audio\s+\d+>', '', clause, flags=re.I).strip().strip(' .;；。')
                if re.match(r'^(?:is\s+(?:(?:his|her|their)\s+)?(?:voice\s+)?(?:timbre\s+)?reference\s+only|'
                            r'(?:is\s+)?(?:voice\s+)?timbre\s+reference\s+only|'
                            r'(?:is\s+)?(?:a\s+)?voice\s+reference\s+only|'
                            r'仅(?:作为|用于).*(?:音色|声音).*(?:参考|对齐)|(?:是|为).*(?:音色参考|声音参考))$',
                            words, re.I):
                    kept.append(re.search(r'\s*$', clause).group())
                    continue
                identity = re.fullmatch(r'\s*(<Subject\s+[1-9]\d*>\s*\(S[1-9]\d*\))\s+'
                    r'(?:uses?|using)\s+<Audio\s+\d+>\s+(?:for\s+)?(?:voice\s+)?timbre(?:\s+only)?[.;；。]?\s*', clause, re.I)
                if identity:
                    kept.append(identity.group(1) + ': inherit the established speaker identity from accepted audiovisual history. ')
                    continue
                raise ValueError('Cannot safely remove a mixed voice binding; provide an exact-window '
                                 'sections.subject_definitions (and affected local sections) override separating identity from audio references')
            if re.fullmatch(r'\s*(?:Audio references define timbre only\.|'
                            r'Use each audio solely for its assigned speaker\x27s timbre\.|'
                            r'Do not reproduce the reference recording\x27s words\.)\s*', clause, re.I):
                continue
            kept.append(clause)
        return ''.join(kept)
    return directing_text(text, clean)


def _binding(settings, field, graph):
    value, file = settings.get(field), settings.get(field + '_file')
    if file is not None and not isinstance(file, str):
        raise ValueError(f'{field}_file must be a filename string')
    if value is not None and value != '':
        if (not isinstance(value, list) or len(value) != 2 or not isinstance(value[0], str)
                or type(value[1]) is not int or value[1] < 0 or value[0] not in graph):
            raise ValueError(f'{field} has a missing or malformed graph binding')
        return {'link': deepcopy(value)}
    return {'file': file} if file and file.strip() else None


def snapshot_bindings(settings, graph):
    source_files = {}
    for node in graph.values():
        if node.get('class_type') == 'MiniMaxH3DirectorPlus':
            source_files = node.get('_meta', {}).get(SOURCE_FILES_META, {})
    result = {}
    for kind, fields in (('pictures', PICTURE_SLOTS), ('audios', AUDIO_SLOTS)):
        bindings = [_binding(settings, field, graph) for field in fields]
        if kind == 'audios' and settings.get('voice_mode', 'none') == 'none':
            bindings = [None] * len(fields)
        # Endpoints are independently optional in the actual flattened Ref2VA
        # order. Explicit selection below compacts them into a Director-valid
        # contiguous destination sequence; ordinary/image and audio gaps are
        # always errors, rather than an implicit renumbering of original IDs.
        active = bindings if kind == 'audios' else bindings[2:]
        if any(active):
            last = max(i for i, binding in enumerate(active) if binding)
            if any(binding is None for binding in active[:last + 1]):
                raise ValueError(f'Original {kind} bindings contain a gap; use contiguous Director slots before selecting references')
        records = []
        for field, binding in zip(fields, bindings):
            if binding is not None:
                record = {'id': len(records) + 1, 'slot': field, 'binding': binding, 'role': field}
                if kind == 'pictures':
                    source = settings.get(field + '_file', source_files.get(field, '')) or ''
                    if not isinstance(source, str):
                        raise ValueError(f'{field}_file preprocessing source must be text')
                    record['preprocessing_source'] = source
                else:
                    record['name'] = settings.get(f'voice_reference_name_{len(records) + 1}', '')
                    record['transcript'] = settings.get('reference_transcript', '') if not records else ''
                records.append(record)
        result[kind] = records
    return result


def apply_reference_policy(settings, request, graph, *, preprocessing_validator=None):
    """Apply static binding transforms with an optional adapter-owned validator.

    The default is deliberately pure: it checks no actual Director crop/filter
    policy and does not authorize video. Production callers must use the graph
    adapter's apply_window_reference_policy wrapper before saving a take.
    """
    if preprocessing_validator is not None and not callable(preprocessing_validator):
        raise ValueError('preprocessing_validator must be callable or None')
    selection = validate_selection(request.get('reference_selection'))
    policy = request.get('voice_policy', 'inherit')
    if not isinstance(policy, str) or policy not in {'inherit', 'dialogue_only'}:
        raise ValueError('voice_policy must be inherit or dialogue_only')
    if policy == 'dialogue_only':
        ids = request.get('dialogue_ids')
        if not isinstance(ids, list) or any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
            raise ValueError('dialogue_only requires an explicit unique dialogue_ids list')
    quiet = policy == 'dialogue_only' and not request['dialogue_ids']
    original = snapshot_bindings(settings, graph)
    audit = request.get('audit', {})
    if not isinstance(audit, dict):
        raise ValueError('Compiled reference audit must be an object')
    declared = validate_selection(audit.get('original_reference_selection'))
    if declared is not None:
        expected = {**declared, 'audios': []} if quiet else declared
        if selection != expected:
            raise ValueError('Compiled reference selection disagrees with its original binding audit')
        for key, ids in declared.items():
            if any(number > len(original[key]) for number in ids):
                raise ValueError(f'Original audited {key} selection names a missing binding')
    chosen = selection or {key: [record['id'] for record in rows] for key, rows in original.items()}
    chosen = deepcopy(chosen)
    for key, ids in chosen.items():
        available = {record['id'] for record in original[key]}
        if any(i not in available for i in ids):
            raise ValueError(f'reference_selection.{key} names an original binding that is missing')
    if quiet:
        chosen['audios'] = []
    if quiet or (selection is not None and not selection['audios']):
        settings['prompt'] = without_local_voice_bindings(settings['prompt'])
    maps = reference_maps(chosen)
    if selection is not None:
        if audit.get('prompt_reference_space') == 'window':
            if audit.get('reference_maps') != maps:
                raise ValueError('Compiled prompt reference maps disagree with the actual window selection')
        else:
            settings['prompt'] = remap_references(settings['prompt'], maps)
        if preprocessing_validator is not None:
            preprocessing_validator(settings['prompt'], original['pictures'], chosen['pictures'])
        for field in PICTURE_SLOTS:
            settings.pop(field, None)
            settings.pop(field + '_file', None)
        for local, original_id in enumerate(chosen['pictures']):
            record = original['pictures'][original_id - 1]
            binding = record['binding']
            settings[PICTURE_SLOTS[local] + ('_file' if 'file' in binding else '')] = deepcopy(next(iter(binding.values())))
            if 'link' in binding and record['preprocessing_source']:
                settings[PICTURE_SLOTS[local] + '_file'] = record['preprocessing_source']
    if selection is not None or quiet:
        for key in list(settings):
            if key.startswith('voice_reference_'):
                settings.pop(key)
        settings['reference_transcript'] = (original['audios'][0]['transcript']
                                            if chosen['audios'] and chosen['audios'][0] == 1 else '')
        for local, original_id in enumerate(chosen['audios']):
            record = original['audios'][original_id - 1]
            binding = record['binding']
            settings[AUDIO_SLOTS[local] + ('_file' if 'file' in binding else '')] = deepcopy(next(iter(binding.values())))
            settings[f'voice_reference_name_{local + 1}'] = record['name']
        if quiet or not chosen['audios']:
            settings['voice_mode'] = 'none'
    validate_local_references(settings['prompt'], {key: len(ids) for key, ids in chosen.items()})
    selected = {key: [{'original_id': number, 'local_id': local,
                      'destination': (PICTURE_SLOTS if key == 'pictures' else AUDIO_SLOTS)[local - 1]
                      if selection is not None else original[key][number - 1]['slot'],
                      'source_slot': original[key][number - 1]['slot']}
                     for local, number in enumerate(ids, 1)] for key, ids in chosen.items()}
    validation = ('not_applicable_no_selection' if selection is None else
                  'callback_checked' if preprocessing_validator is not None else 'not_performed_static')
    return {'original': original, 'selection': chosen, 'maps': maps, 'selected': selected,
            'voice_policy': policy, 'quiet': quiet, 'preprocessing_validation': validation}
