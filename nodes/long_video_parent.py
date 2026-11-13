"""A frozen, observed native parent image for one physical continuation window.

Policy validation is pure. Media imports stay inside CPU-only functions so the
planner and static preflight never load Comfy, models, tensors or parent media.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
import re
import uuid

from .long_video_store import ProjectStore, StaleCoordination
from .long_video_references import validate_parent_reference, snapshot_bindings, PICTURE_SLOTS, SOURCE_FILES_META


def _frame_index(request, policy):
    from .long_video_av import conditioning_geometry
    geometry = conditioning_geometry(request['window'], request.get('forward_frames', 0))
    if policy['frame_policy'] == 'forward_last':
        if not geometry['forward_frames']:
            raise ValueError('parent_reference forward_last requires an explicit reviewed forward range')
        return request['window']['source_endpoint'] + geometry['forward_frames'] - 1
    return request['window']['source_endpoint'] - 1


def _source(store, request, state=None):
    """Bind only the accepted adjacent full native preview and its real review."""
    from .long_video_plan import validate_window
    window = validate_window(request['window'])
    policy = validate_parent_reference(request['parent_reference'])
    parent = request.get('parent')
    if (not window['context_frames'] or not isinstance(parent, dict) or not parent.get('take_id')
            or not parent.get('history_sha256')):
        raise ValueError('Parent reference requires the current accepted adjacent native source')
    current = store.active(state if state is not None else store.read(), parent['take_id'])
    previous = current['window']
    if (current['status'] != 'accepted' or previous['index'] != window['index'] - 1
            or previous['new_end'] != window['new_start']
            or previous['local_new_end'] != window['source_endpoint']):
        raise StaleCoordination('Parent reference must use the current accepted adjacent native window')
    store.verify_accepted(current)
    artifacts = current['artifacts']
    if not {'native', 'native_receipt', 'request'}.issubset(artifacts):
        raise ValueError('Parent reference requires the full native preview and its immutable native receipt/request')
    receipt = json.loads(store.path(artifacts['native_receipt']['name']).read_text(encoding='utf-8'))
    if receipt.get('artifacts', {}).get('native') != artifacts['native']:
        raise ValueError('Native preview differs from its immutable native receipt')
    saved = json.loads(store.path(artifacts['request']['name']).read_text(encoding='utf-8'))
    if (saved.get('take_id') != current['take_id'] or saved.get('project_name') != store.name
            or saved.get('window') != previous):
        raise ValueError('Parent native request differs from the accepted window')
    frame = _frame_index(request, policy)
    if not 0 <= frame < previous['sample_frames']:
        raise ValueError('Parent native reference frame is outside the actual physical source range')
    info = {'schema_version': 1, 'project_name': store.name, 'window': deepcopy(window),
            'policy': policy, 'parent_take_id': current['take_id'], 'parent_window': deepcopy(previous),
            'source_native': deepcopy(artifacts['native']), 'source_native_receipt': deepcopy(artifacts['native_receipt']),
            'source_review': deepcopy(current['review']), 'source_history_sha256': parent['history_sha256'],
            'frame_policy': policy['frame_policy'], 'frame_index': frame}
    if policy['frame_policy'] == 'forward_last':
        forward = parent.get('forward_review')
        if not forward or current.get('forward_review') != forward:
            raise ValueError('Parent forward frame requires the exact explicit forward review proof')
        proof = store.verify_forward_review(current, forward)
        if (proof['requirement']['next_window'] != window
                or proof['requirement']['forward_frames'] != request.get('forward_frames', 0)
                or proof['consumed_av_sha256'] != parent.get('history_sha256')):
            raise ValueError('Parent forward review scope does not match this window, range and consumed history')
        info['source_forward_review'] = deepcopy(forward)
    return info


def _header(source, frames):
    if len(source.streams.video) != 1:
        raise ValueError('Parent native preview requires exactly one video stream')
    stream = source.streams.video[0]
    if stream.frames != frames or stream.average_rate != 24 or stream.width <= 0 or stream.height <= 0:
        raise ValueError('Parent native preview geometry/frame range differs from its physical native window')
    return stream


def _hash_stream(source):
    source.seek(0)
    digest = hashlib.sha256()
    for chunk in iter(lambda: source.read(1024 * 1024), b''):
        digest.update(chunk)
    source.seek(0)
    return digest.hexdigest()


def decode_native_frame(path, index, frames, sha256):
    """Materialize just one RGB image, retaining at most one bounded CPU frame.

    Read and hash the same open file that PyAV decodes. This prevents a pathname
    replacement from changing the source between its hash and the decoder open.
    A native physical clip has at most 360 frames, independent of film duration.
    """
    import av
    if type(index) is not int or not 0 <= index < frames <= 360:
        raise ValueError('Parent native frame must be inside one bounded physical window')
    image = None
    with open(path, 'rb') as data:
        if _hash_stream(data) != sha256:
            raise ValueError('Parent native bytes changed before frame extraction')
        with av.open(data) as source:
            stream = _header(source, frames)
            stream.codec_context.thread_count = 1
            for number, frame in enumerate(source.decode(stream)):
                if number == index:
                    if (frame.width, frame.height) != (stream.width, stream.height):
                        raise ValueError('Decoded parent frame geometry changed')
                    image = frame.to_image().convert('RGB')
                    break
        if _hash_stream(data) != sha256:
            raise ValueError('Parent native bytes changed during frame extraction')
    if image is None:
        raise ValueError('Parent native preview ends before the requested source frame')
    return image


def _verify_png(store, artifact, dimensions):
    from PIL import Image
    store.verify({'parent_png': artifact})
    with Image.open(store.path(artifact['name'])) as image:
        if image.format != 'PNG' or image.mode != 'RGB' or image.size != dimensions:
            raise ValueError('Frozen parent PNG dimensions or format differ from the native source')
        image.load()


def resolve_parent_reference(store, request, *, expected_snapshot):
    """Publish one complete immutable PNG+receipt under a coordination lock.

    Callers reuse existing frozen descriptors through verify_parent_reference;
    this function is for a new base generation only, never recovery or repair.
    """
    if request.get('project_name', store.name) != store.name:
        raise ValueError('Parent reference project identity differs from its journal')
    state = store.read()
    store.check_coordination(state, request['window']['index'], expected_snapshot)
    source = _source(store, request, state)
    native = store.path(source['source_native']['name'])
    image = decode_native_frame(native, source['frame_index'], source['parent_window']['sample_frames'],
                                source['source_native']['sha256'])
    store.verify({'source_native': source['source_native']})
    identity = 'parent-' + uuid.uuid4().hex
    relative = 'references/' + identity
    staging = store.path('references/.' + identity + '.tmp')
    staging.mkdir(parents=True)
    published = False
    try:
        png = staging / 'frame.png'
        image.save(png, format='PNG')
        with png.open('r+b') as data:
            os.fsync(data.fileno())
        artifact = {**store.artifact(png), 'name': relative + '/frame.png'}
        receipt = {**source, 'width': image.width, 'height': image.height, 'png': artifact}
        staged_artifact = {**artifact, 'name': png.relative_to(store.root).as_posix()}
        _verify_png(store, staged_artifact, image.size)
        receipt_path = store.write_once((staging / 'receipt.json').relative_to(store.root), receipt)
        descriptor = {**receipt, 'receipt': {**store.artifact(receipt_path), 'name': relative + '/receipt.json'}}
        with store.transaction() as current:
            store.check_coordination(current, request['window']['index'], expected_snapshot)
            if current['stop_requested'] or current.get('active_job'):
                raise StaleCoordination('Parent frame publication lost its coordinator reservation')
            if _source(store, request, current) != source:
                raise StaleCoordination('Actual parent reference source changed during extraction')
            # A directory rename exposes the complete pair together. No reader
            # can discover a usable receipt before its PNG has been published.
            os.rename(staging, store.path(relative))
            published = True
        return descriptor
    finally:
        if not published:
            for name in ('frame.png', 'receipt.json'):
                (staging / name).unlink(missing_ok=True)
            staging.rmdir()


def verify_parent_reference(store, request, *, state=None):
    """Verify frozen ownership and bytes without decoding/replacing its source."""
    policy = request.get('parent_reference')
    frozen = request.get('parent_reference_image')
    if policy is None or not request['window']['context_frames']:
        if frozen is not None:
            raise ValueError('An inactive parent reference cannot carry a frozen image')
        return None
    validate_parent_reference(policy)
    if not isinstance(frozen, dict) or frozen.get('project_name') != store.name:
        raise ValueError('Continuation requires its frozen bound project parent PNG')
    png, receipt_artifact = frozen.get('png', {}), frozen.get('receipt', {})
    match = re.fullmatch(r'references/(parent-[0-9a-f]{32})/frame\.png', str(png.get('name', '')))
    if not match or receipt_artifact.get('name') != 'references/' + match[1] + '/receipt.json':
        raise ValueError('Parent PNG must be a bound project reference artifact')
    store.verify({'parent_receipt': receipt_artifact})
    receipt = json.loads(store.path(receipt_artifact['name']).read_text(encoding='utf-8'))
    if receipt != {key: value for key, value in frozen.items() if key != 'receipt'}:
        raise ValueError('Frozen parent descriptor differs from its immutable receipt')
    actual = _source(store, request, state)
    if {key: frozen.get(key) for key in actual} != actual:
        raise ValueError('Frozen parent reference no longer binds the current adjacent accepted source')
    import av
    with av.open(str(store.path(actual['source_native']['name']))) as media:
        stream = _header(media, actual['parent_window']['sample_frames'])
        dimensions = (stream.width, stream.height)
    if (frozen.get('width'), frozen.get('height')) != dimensions:
        raise ValueError('Frozen parent reference geometry differs from the native source')
    _verify_png(store, png, dimensions)
    return frozen


def apply_parent_overlay(settings, request, graph, *, parent_store=None, preflight=False):
    """Apply the same private image binding before either reference handshake.

    The only static substitute is conspicuously unobserved and exists solely in
    a detached preflight graph. Production always verifies a frozen descriptor.
    """
    policy = request.get('parent_reference')
    if policy is None:
        if request.get('parent_reference_image') is not None:
            raise ValueError('A frozen parent image requires its parent_reference policy')
        return None
    policy = validate_parent_reference(policy)
    original = snapshot_bindings(settings, graph)['pictures']
    selection = request.get('reference_selection')
    if policy['mode'] == 'append':
        if selection is not None:
            raise ValueError('parent_reference append cannot use explicit reference_selection; use designated replace')
        if len(original) > 8:
            raise ValueError('Native continuity needs one existing picture slot: use at most 8 static images '
                             'so the generated parent can occupy the ninth continuity slot')
    if not request['window']['context_frames']:
        if request.get('parent_reference_image') is not None:
            raise ValueError('The first window cannot bind a parent image')
        return None
    if policy['mode'] == 'replace':
        original_id = policy['picture_id']
        if original_id > len(original):
            raise ValueError('parent_reference designates a missing original Picture binding')
        chosen = selection['pictures'] if selection is not None else [item['id'] for item in original]
        if original_id not in chosen:
            raise ValueError('parent_reference Picture must be explicitly selected in this window')
        if len(chosen) > 9:
            raise ValueError('parent_reference cannot exceed the Director maximum of 9 active pictures')
        slot = original[original_id - 1]['slot']
    else:
        original_id = len(original) + 1
        position = PICTURE_SLOTS.index(original[-1]['slot']) + 1 if original else 0
        if position >= len(PICTURE_SLOTS):
            raise ValueError('No existing Director picture position remains for the continuity slot')
        slot = PICTURE_SLOTS[position]
    if preflight:
        frozen = {'observation': 'UNOBSERVED_STATIC_PREFLIGHT_ONLY_UNAPPROVED',
                  'window': deepcopy(request['window']), 'policy': policy,
                  'frame_policy': policy['frame_policy'], 'frame_index': _frame_index(request, policy)}
    else:
        if parent_store is None:
            import folder_paths
            parent_store = ProjectStore(folder_paths.get_output_directory(), request['project_name'])
        frozen = verify_parent_reference(parent_store, request)
    node_id = 'h3long_parent_reference'
    graph[node_id] = {'class_type': 'MiniMaxH3LongVideoParentImage', 'inputs': {
        'request_json': json.dumps({**request, 'stage': request.get('stage', 'native')},
                                  ensure_ascii=False, allow_nan=False)}}
    settings[slot] = [node_id, 0]
    # File widgets are upload names, not project artifact paths. The connected
    # private IMAGE needs no upload filename and no storyboard/five-view hint.
    settings.pop(slot + '_file', None)
    for node in graph.values():
        if node.get('class_type') == 'MiniMaxH3DirectorPlus':
            node.setdefault('_meta', {}).setdefault(SOURCE_FILES_META, {})[slot] = ''
    return {'mode': policy['mode'], 'original_picture_id': original_id, 'source_slot': slot,
            'binding': {'link': [node_id, 0]}, 'preprocessing_source': '', 'image': deepcopy(frozen)}


def retain_parent_instruction(prompt, local_picture_id):
    """Add nonspoken direction within the existing official retention field."""
    instruction = (f'<Picture {local_picture_id}> is the observed accepted native parent at the continuation boundary. '
        'Use it as a soft visual reference for the current pose, expression, costume, scene, prop ownership, '
        'contact and relative scale while the actual clean audiovisual history retains motion and speaker identity. '
        'Continue the current action without replaying its onset; this reference does not impose a hard first or last frame.')
    # Match only official section delimiters, preserving literal dialogue and
    # all original Subject/Sx/Picture/Audio declarations byte for byte.
    control = re.sub(r'<d(?:\s+[^>]*)?>.*?</d>', lambda match: ' ' * len(match[0]), prompt, flags=re.I | re.S)
    marker = 'retention_analysis:\n'
    if marker not in control:
        raise ValueError('parent_reference requires the official retention_analysis prompt field')
    start = control.index(marker) + len(marker)
    end = control.find('\n\ndetailed_description:\n', start)
    if end < 0:
        raise ValueError('parent_reference requires the existing six-field prompt')
    section = control[start:end]
    if instruction in section:
        return prompt
    return prompt[:end] + '\n' + instruction + prompt[end:]


class MiniMaxH3LongVideoParentImage:
    """Private loader: only an immutable take's project-owned frozen PNG."""
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'request_json': ('STRING',)}}

    RETURN_TYPES = ('IMAGE',)
    FUNCTION = 'load'
    CATEGORY = 'MiniMax H3 导演台 Plus/原生长视频'

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float('nan')

    def load(self, request_json):
        from .long_video_runtime import _request, _bound_store
        from PIL import Image
        import numpy as np
        import torch
        request = _request(request_json)
        store = _bound_store(request)
        frozen = verify_parent_reference(store, request)
        if frozen is None:
            raise ValueError('Private parent IMAGE loader requires an active frozen continuation image')
        with Image.open(store.path(frozen['png']['name'])) as image:
            pixels = np.array(image, dtype=np.uint8, copy=True)
        # Catch file or source replacement during the CPU read, before sampling.
        verify_parent_reference(store, request)
        return (torch.from_numpy(pixels).to(dtype=torch.float32).unsqueeze(0) / 255.0,)
