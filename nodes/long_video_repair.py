"""Reviewed unused native video for bounded repair; never a media approval."""
from __future__ import annotations

from copy import deepcopy
import json
import re
import time
import uuid

from .long_video_plan import validate_window, video_steps


def _steps(value, label):
    if (not isinstance(value, list) or len(value) != 2
            or any(type(step) is not int for step in value)
            or not 2 <= value[0] < value[1]
            or any((step - 2) % 5 for step in value)):
        raise ValueError(f'{label} must align to complete native five-step video groups')
    return list(value)


def normalize_video_reference(value, count, head, positive_spans):
    if not isinstance(value, dict) or set(value) != {'source_take', 'source_steps', 'target_steps', 'review'}:
        raise ValueError('Video reference requires a source take, native ranges and its actual review')
    if not isinstance(value['source_take'], str) or not re.fullmatch(r'[\w-]{1,128}', value['source_take']):
        raise ValueError('Invalid video reference source take')
    source = _steps(value['source_steps'], 'Reference source_steps')
    target = _steps(value['target_steps'], 'Reference target_steps')
    if source[1] - source[0] != target[1] - target[0] or not head <= target[0] < target[1] <= count:
        raise ValueError('Video reference lengths must match and preserve protected history')
    review = value['review']
    if (not isinstance(review, dict) or set(review) != {'name', 'sha256', 'bytes'}
            or not isinstance(review['name'], str)
            or not re.fullmatch(r'reviews/[\w-]+\.json', review['name'])
            or not isinstance(review['sha256'], str) or not re.fullmatch(r'[0-9a-f]{64}', review['sha256'])
            or type(review['bytes']) is not int or review['bytes'] <= 0):
        raise ValueError('Video reference needs an immutable project review artifact')
    cursor = target[0]
    for a, b in sorted(positive_spans):
        if a <= cursor < b:
            cursor = min(target[1], b)
    if cursor != target[1]:
        raise ValueError('Video reference must be wholly inside positive video repair spans')
    if sum(b - a for a, b in positive_spans) <= target[1] - target[0]:
        raise ValueError('Video reference must leave free video to sample')
    return deepcopy(value)


def _source_identity(store, source_take, source_steps, state):
    source_steps = _steps(source_steps, 'Reference source_steps')
    source = store.active(state, source_take)
    if source['status'] != 'accepted':
        raise ValueError('Video reference must use a current accepted earlier window')
    store.verify_accepted(source)
    window = validate_window(source['window'])
    artifacts = source['artifacts']
    required = {'clean_video', 'clean_audio', 'clean_receipt', 'native', 'native_receipt', 'request'}
    if not required.issubset(artifacts):
        raise ValueError('Video reference requires actual clean AV, native preview and immutable receipts')
    start, end = [5 + 17 * ((step - 2) // 5) for step in source_steps]
    delivered_end = window['local_new_end']
    # An accepted child may have delivered reviewed forward surplus too.
    for child in state['windows']:
        if child.get('status') == 'accepted' and (child.get('parent') or {}).get('take_id') == source_take:
            delivered_end = max(delivered_end, window['local_new_end'] + child.get('forward_frames', 0))
    if start < delivered_end or end > window['sample_frames'] or source_steps[1] > video_steps(window['sample_frames']):
        raise ValueError('Video reference must use genuinely unused native tail, never delivered frames')
    saved = json.loads(store.path(artifacts['request']['name']).read_text(encoding='utf-8'))
    if (saved.get('take_id') != source_take or saved.get('project_name') != store.name
            or saved.get('window') != window):
        raise ValueError('Reference source differs from its immutable native request')
    native = json.loads(store.path(artifacts['native_receipt']['name']).read_text(encoding='utf-8'))
    clean = json.loads(store.path(artifacts['clean_receipt']['name']).read_text(encoding='utf-8'))
    if (native.get('artifacts', {}).get('native') != artifacts['native']
            or any(clean.get('artifacts', {}).get(key) != artifacts[key] for key in ('clean_video', 'clean_audio', 'request'))):
        raise ValueError('Reference clean AV or native preview differs from its immutable receipts')
    return {'schema_version': 1, 'project_name': store.name, 'source_take': source_take,
            'source_steps': source_steps, 'source_native_frames': [start, end],
            'source_window': deepcopy(window), 'source_artifacts': deepcopy(artifacts),
            'source_review': deepcopy(source['review'])}


def review_video_reference(store, source_take, source_steps, scope, observed_state):
    """Record an explicit observation of actual unused media, separately from delivery."""
    if any(not isinstance(value, str) or not value.strip() for value in (scope, observed_state)):
        raise ValueError('Record actual reference review scope and observed pose/prop state')
    with store.transaction() as state:
        if state.get('active_job'):
            raise ValueError('Wait for the running window before reviewing video reference media')
        receipt = {**_source_identity(store, source_take, source_steps, state),
                   'scope': scope, 'observed_state': observed_state, 'time': time.time()}
        path = store.write_once(f'reviews/video-reference-{uuid.uuid4().hex}.json', receipt)
        return store.artifact(path)


def verify_video_reference(store, reference, target_window, state=None):
    """Recheck the current source and exact review before revocation or execution."""
    state = store.read() if state is None else state
    if not isinstance(reference, dict):
        raise ValueError('Invalid video reference')
    review = reference.get('review')
    if not isinstance(review, dict) or not isinstance(review.get('name'), str):
        raise ValueError('Missing actual video reference review')
    store.verify({'video_reference_review': review})
    path = store.path(review['name'])
    if store.artifact(path) != review:
        raise ValueError('Video reference review size/hash changed')
    receipt = json.loads(path.read_text(encoding='utf-8'))
    actual = _source_identity(store, reference.get('source_take'), reference.get('source_steps'), state)
    if (actual['source_window']['index'] >= validate_window(target_window)['index']
            or any(receipt.get(key) != value for key, value in actual.items())
            or any(not isinstance(receipt.get(key), str) or not receipt[key].strip() for key in ('scope', 'observed_state'))):
        raise ValueError('Video reference review does not bind this current accepted earlier unused source')
    return receipt
