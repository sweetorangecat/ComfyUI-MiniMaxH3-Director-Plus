"""Journal mutation guards and immutable actual-tail review receipts."""
import pytest

from nodes.long_video_av import slice_fingerprint
from nodes.long_video_plan import plan_windows
from nodes.long_video_store import ProjectStore
from test_long_video_runtime import environment, native_take
from test_long_video_store import spec, native


def forward_parent(store, forward=17):
    windows = plan_windows(720)
    store.configure({'total_frames': 720, 'scene_contract': {'window_overrides': [
        {'new_start': windows[1]['new_start'], 'new_end': windows[1]['new_end'], 'forward_frames': forward}
    ]}}, windows)
    request, clean = native_take(store)
    store.review(request['take_id'], True, 'Delivery endpoint: right hand on cup.', 'Delivered native AV checked.')
    return store.read()['windows'][0], clean


def test_forward_review_derives_exact_source_tail_and_binds_actual_clean_preview(environment):
    record, clean = forward_parent(environment)
    required = environment.required_forward_review(record)
    assert required['next_index'] == 1 and required['forward_frames'] == 17
    assert required['source_local_range'] == [294, 311]
    assert required['consumed_local_range'] == [255, 311]
    assert required['protected_frames'] == 56
    environment.review_forward(record['take_id'], 'Saw local tail through frame 311 with sound.',
                               'Forward endpoint: right fingers releasing cup, camera moving left.')
    record = environment.read()['windows'][0]
    assert record['status'] == 'accepted'
    assert record['observed_outgoing_state'] == 'Delivery endpoint: right hand on cup.'
    proof = environment.verify_forward_review(record, record['forward_review'])
    assert proof['source_artifacts'] == record['artifacts']
    assert proof['source_review'] == record['review']
    assert proof['requirement'] == required
    assert proof['consumed_av_sha256'] == slice_fingerprint(clean, 294, 39, 17)
    environment.verify({'forward_review': record['forward_review']})
    path = environment.path(record['forward_review']['name'])
    path.write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='hash'):
        environment.verify_forward_review(record)


@pytest.mark.parametrize('bad', ['missing', 'not_accepted', 'no_actual_preview', 'active_job', 'wrong_take',
                                'empty_scope', 'null_observation', 'expected_state', 'unavailable'])
def test_invalid_forward_review_preserves_valid_journal(environment, bad):
    record, _ = forward_parent(environment, forward=51 if bad == 'unavailable' else 17)
    take_id, scope, observed = record['take_id'], 'Actual tail and sound checked.', 'Actual free-phase state.'
    if bad == 'missing':
        with pytest.raises(ValueError):
            environment.verify_forward_review(record)
        return
    if bad in {'not_accepted', 'no_actual_preview', 'active_job'}:
        with environment.transaction() as state:
            if bad == 'not_accepted':
                state['windows'][0]['status'] = 'awaiting_review'
            elif bad == 'no_actual_preview':
                state['windows'][0]['artifacts'].pop('native')
            else:
                state['active_job'] = {'prompt_id': 'still-running'}
    if bad == 'wrong_take':
        take_id = 'not-current'
    if bad == 'empty_scope':
        scope = ' '
    if bad == 'null_observation':
        observed = None
    if bad == 'expected_state':
        observed = {'expected_outgoing_state': 'planned pose'}
    before = environment.path('state.json').read_bytes()
    with pytest.raises(ValueError):
        environment.review_forward(take_id, scope, observed)
    assert environment.path('state.json').read_bytes() == before


def test_changed_forward_proof_invalidates_descendants_and_final_but_keeps_parent(environment):
    record, _ = forward_parent(environment)
    environment.review_forward(record['take_id'], 'Tail checked', 'First observed endpoint')
    first = environment.read()['windows'][0]['forward_review']
    child = environment.create_take(spec(1))
    native(environment, child)
    environment.review(child['take_id'], True, 'Child endpoint', 'Reviewed child')
    with environment.transaction() as state:
        state['output'] = 'previous film'
    environment.review_forward(record['take_id'], 'Tail checked again', 'Corrected actual endpoint')
    state = environment.read()
    assert state['windows'][0]['status'] == 'accepted'
    assert state['windows'][1]['status'] == 'invalidated' and state['output'] == ''
    assert state['windows'][0]['forward_review'] != first
    environment.verify({'old_forward_review': first})


@pytest.mark.parametrize('kind', ['queued', 'sampled', 'missing_preview', 'unknown_field', 'protected_forward'])
def test_repair_requires_decoded_current_source_and_validates_all_bounds_before_invalidation(tmp_path, kind):
    store = ProjectStore(tmp_path, 'repair')
    for i in range(2 if kind == 'protected_forward' else 1):
        request = store.create_take({**spec(i), 'forward_frames': 34 if i else 0})
        native(store, request)
        store.review(request['take_id'], True, 'Observed state', 'Full native review')
    with store.transaction() as state:
        record = state['windows'][-1]
        if kind in {'queued', 'sampled'}:
            record['status'] = kind
        if kind == 'missing_preview':
            record['artifacts'].pop('native')
    before = store.path('state.json').read_bytes()
    with pytest.raises((ValueError, TypeError)):
        if kind == 'unknown_field':
            store.request_repair(0, [[20, 24]], anchors=[])
        else:
            store.request_repair(len(store.read()['windows']) - 1, [[20, 24]])
    assert store.path('state.json').read_bytes() == before


def test_seconds_audio_repair_and_generated_rejected_progress_do_not_grant_acceptance(tmp_path):
    store = ProjectStore(tmp_path, 'repair')
    request = store.create_take(spec())
    native(store, request)
    store.review(request['take_id'], False, '', 'Rejected observed lip timing.')
    assert store.read()['generated_frames'] == 294 and store.read()['completed_frames'] == 0
    store.request_repair_seconds(0, 2, 2.4, 'audio')
    repair = store.read()['repair']
    assert repair['video_ranges'] == [] and repair['audio_ranges'] == [[80, 96]]
    assert repair['lock_audio'] is False
    assert repair['conditioning_extent']['requested_seconds'] == [2, 2.4]
    assert store.read()['windows'][0]['status'] == 'invalidated'


def test_accepted_retry_requires_explicit_matching_replacement_and_increments_content_attempt(tmp_path):
    store = ProjectStore(tmp_path, 'repair')
    request = store.create_take(spec())
    native(store, request)
    store.review(request['take_id'], True, 'Actual state', 'Full native review')
    before = store.path('state.json').read_bytes()
    for kwargs in ({}, {'replace_accepted': True}, {'replace_accepted': True, 'take_id': 'stale'},
                   {'replace_accepted': 1, 'take_id': request['take_id']}):
        with pytest.raises(ValueError):
            store.request_retry(0, **kwargs)
        assert store.path('state.json').read_bytes() == before
    store.request_retry(0, replace_accepted=True, take_id=request['take_id'])
    assert store.read()['windows'][0]['status'] == 'invalidated'
    assert store.read()['content_attempts']['0'] == 1
    store.verify_accepted({**store.read()['windows'][0], 'status': 'accepted'})


@pytest.mark.parametrize('observation', [None, {'expected_state': 'planned pose'}, ' '])
def test_acceptance_requires_explicit_actual_observation_without_coercing_source_metadata(tmp_path, observation):
    store = ProjectStore(tmp_path, 'review')
    request = store.create_take(spec())
    native(store, request)
    before = store.path('state.json').read_bytes()
    with pytest.raises(ValueError):
        store.review(request['take_id'], True, observation, 'Actual native review')
    assert store.path('state.json').read_bytes() == before


def test_observed_delivery_state_cannot_drift_from_immutable_acceptance_receipt(tmp_path):
    store = ProjectStore(tmp_path, 'review')
    request = store.create_take(spec())
    native(store, request)
    store.review(request['take_id'], True, 'Actual observed hand position', 'Native review')
    with store.transaction() as state:
        state['windows'][0]['observed_outgoing_state'] = 'Expected future pose'
    with pytest.raises(ValueError, match='acceptance'):
        store.verify_accepted(store.read()['windows'][0])


def test_bound_take_creation_checks_forward_proof_under_same_journal_lock(environment):
    record, clean = forward_parent(environment)
    environment.review_forward(record['take_id'], 'Tail checked', 'Old observed free state')
    frozen = environment.read()['windows'][0]['forward_review']
    pending = {**spec(1), 'forward_frames': 17, 'parent': {
        'take_id': record['take_id'], 'forward_review': frozen,
        'history_sha256': slice_fingerprint(clean, 294, 39, 17)}}
    environment.review_forward(record['take_id'], 'Tail checked again', 'Corrected observed free state')
    before = environment.path('state.json').read_bytes()
    takes = sorted(path.name for path in environment.path('takes').iterdir())
    with pytest.raises(ValueError, match='forward|Forward'):
        environment.create_take(pending, require_current_parent=True,
            expected_snapshot=environment.coordination_snapshot(environment.read(), 1))
    assert environment.path('state.json').read_bytes() == before
    assert sorted(path.name for path in environment.path('takes').iterdir()) == takes


@pytest.mark.parametrize('joint', [False, True])
def test_underflowing_positive_video_weight_preserves_accepted_journal(tmp_path, joint):
    store = ProjectStore(tmp_path, 'repair')
    request = store.create_take(spec())
    native(store, request)
    store.review(request['take_id'], True, 'Observed actual hand position', 'Full native AV review')
    before = store.path('state.json').read_bytes()
    with pytest.raises(ValueError, match='float32'):
        store.request_repair(0, [], lock_audio=not joint, video_weights=[[20, 24, 1e-50]],
                             audio_ranges=[[100, 120]] if joint else None)
    assert store.path('state.json').read_bytes() == before
    store.verify_accepted(store.read()['windows'][0])
