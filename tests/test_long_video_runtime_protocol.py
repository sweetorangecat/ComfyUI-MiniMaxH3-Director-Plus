"""Runtime rechecks the immutable actual-forward proof before any conditioning."""
import json

import pytest
import torch

from nodes.long_video_av import slice_fingerprint
from nodes.long_video_plan import plan_windows
from test_long_video_runtime import (environment, native_take, request, latent, guide, model,
                                     runtime, sample, sampled_guide, VideoVAE)
from test_long_video_store_protocol import forward_parent


def child_request(store, record, clean, *, reviewed=True, forward=17, **changes):
    parent = {'take_id': record['take_id'], 'history_sha256': slice_fingerprint(clean, 294, 39, forward)}
    if reviewed:
        parent['forward_review'] = store.read()['windows'][0]['forward_review']
    return request(store, plan_windows(720)[1], parent=parent, forward_frames=forward, **changes)


def test_runtime_copies_reviewed_tail_and_checks_receipt_hash_again(environment):
    record, clean = forward_parent(environment)
    environment.review_forward(record['take_id'], 'Actual tail AV checked', 'Fingers just release at forward endpoint')
    req = child_request(environment, record, clean)
    prepared = runtime().MiniMaxH3LongVideoContext().prepare(latent(), guide(), json.dumps(req), model())[0]
    sv, sa = clean['samples'].unbind()
    pv, pa = prepared['samples'].unbind()
    assert torch.equal(pv[:, :, :17], sv[:, :, 75:92])
    assert torch.equal(pa[..., :93], sa[..., 425:518])
    # The receipt remains mandatory even when a request supplies the right hash.
    environment.path(req['parent']['forward_review']['name']).write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='hash'):
        runtime()._source(environment, req)


@pytest.mark.parametrize('problem', ['missing', 'wrong', 'wrong_frames'])
def test_runtime_never_treats_delivery_review_as_source_tail_review(environment, problem):
    record, clean = forward_parent(environment)
    environment.review_forward(record['take_id'], 'Actual tail AV checked', 'Observed free-phase state')
    req = child_request(environment, record, clean, reviewed=problem != 'missing')
    if problem == 'wrong':
        req['parent']['forward_review'] = record['review']
    if problem == 'wrong_frames':
        req['forward_frames'] = 34
    with pytest.raises(ValueError, match='forward|Forward'):
        runtime()._source(environment, req)


def test_forward_repair_rejects_stale_tail_even_when_delivery_history_is_identical(environment):
    record, old_parent_av = forward_parent(environment)
    environment.review_forward(record['take_id'], 'Tail checked', 'Old free state')
    old_child = child_request(environment, record, old_parent_av)
    prepared = runtime().MiniMaxH3LongVideoContext().prepare(latent(), guide(), json.dumps(old_child), model())[0]
    runtime().MiniMaxH3LongVideoCheckpoint().save(sample(prepared), prepared, sampled_guide(), VideoVAE(), object(), json.dumps(old_child))
    environment.review(old_child['take_id'], False, '', 'Repair child free interval')
    environment.invalidate_from(0, 'Replace actual source tail')
    parent = request(environment, seed=8)
    base = runtime().MiniMaxH3LongVideoContext().prepare(latent(), guide(), json.dumps(parent), model())[0]
    new_parent_av = sample(base)
    new_parent_av['samples'].unbind()[0][:, :, 90] += 1
    runtime().MiniMaxH3LongVideoCheckpoint().save(new_parent_av, base, sampled_guide(), VideoVAE(), object(), json.dumps(parent))
    environment.review(parent['take_id'], True, record['observed_outgoing_state'], 'New actual delivery checked')
    environment.review_forward(parent['take_id'], 'New actual tail checked', 'New free state')
    assert slice_fingerprint(old_parent_av, 294) == slice_fingerprint(new_parent_av, 294)
    new_record = environment.read()['windows'][0]
    req = child_request(environment, new_record, new_parent_av, repair={
        'source_take': old_child['take_id'], 'video_ranges': [[20, 24]], 'lock_audio': True})
    with pytest.raises(ValueError, match='Repair source protected AV history.*current parent'):
        runtime().MiniMaxH3LongVideoContext().prepare(latent(), guide(), json.dumps(req), model())
    assert not environment.path(f"takes/{req['take_id']}/prepared_context.json").exists()
