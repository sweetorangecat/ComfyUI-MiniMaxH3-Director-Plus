"""Small CPU tensors exercise real copying, protection, repair and provenance."""
import importlib
import json
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


class Nested:
    is_nested = True

    def __init__(self, values):
        self.values = tuple(values)

    def unbind(self):
        return self.values


def module():
    assert (ROOT / 'nodes/long_video_av.py').is_file(), 'Native AV preparation is missing'
    return importlib.import_module('nodes.long_video_av')


def plan():
    return importlib.import_module('nodes.long_video_plan')


def latent(frames=328, offset=0, dtype=torch.float32):
    p = plan()
    v = torch.arange(4 * p.video_steps(frames) * 2 * 3, dtype=dtype).reshape(1, 4, -1, 2, 3) + offset
    a = torch.arange(8 * 2 * p.audio_ticks(frames), dtype=dtype).reshape(1, 8, 2, -1) + offset
    return {'samples': Nested((v, a)), 'some_metadata': 'kept'}


def clone_latent(value):
    return {'samples': Nested(tuple(t.clone() for t in value['samples'].unbind()))}


def test_first_window_still_preserves_two_explicit_all_open_masks():
    av = module()
    target = latent()
    out = av.prepare_av(target, plan().plan_windows(720)[0], nested_factory=Nested)
    assert out is not target and out['some_metadata'] == 'kept'
    vm, am = out['noise_mask'].unbind()
    assert torch.all(vm == 1) and torch.all(am == 1)
    assert vm.shape == (1, 1, 97, 2, 3)
    assert am.shape == (1, 1, 2, 547)
    json.dumps(out['_h3_long_video_context'])
    report = av.validate_result(clone_latent(out), out)
    assert report['protected_video_exact'] and report['protected_audio_exact']
    assert report['content_review'] == 'pending'


def test_continuation_copies_the_actual_two_parent_slices_and_does_not_mutate_input():
    av = module()
    source, target = latent(offset=100), latent(offset=-500)
    tv, ta = [t.clone() for t in av.streams(target)]
    out = av.prepare_av(target, plan().plan_windows(720)[1], source=source, nested_factory=Nested)
    v, a = av.streams(out)
    sv, sa = av.streams(source)
    assert torch.equal(v[:, :, :12], sv[:, :, 75:87])
    assert torch.equal(a[..., :65], sa[..., 425:490])
    assert torch.equal(v[:, :, 12:], tv[:, :, 12:])
    assert torch.equal(a[..., 65:], ta[..., 65:])
    assert torch.equal(av.streams(target)[0], tv)
    vm, am = out['noise_mask'].unbind()
    assert torch.all(vm[:, :, :12] == 0) and torch.all(vm[:, :, 12:] == 1)
    assert torch.all(am[..., :65] == 0) and torch.all(am[..., 65:] == 1)
    context = out['_h3_long_video_context']
    assert context['source_video_range'] == [75, 87]
    assert context['source_audio_range'] == [425, 490]
    assert context['video_steps'] == 12 and context['audio_ticks'] == 65


@pytest.mark.parametrize('which', ['video', 'audio'])
def test_any_changed_protected_element_is_a_hard_error(which):
    av = module()
    out = av.prepare_av(latent(), plan().plan_windows(720)[1], source=latent(offset=2), nested_factory=Nested)
    sampled = clone_latent(out)
    v, a = av.streams(sampled)
    (v if which == 'video' else a).flatten()[0] += 1
    with pytest.raises(ValueError, match='protected'):
        av.validate_result(sampled, out)


def test_repair_copies_entire_failed_take_locks_audio_and_demands_changed_open_video():
    av = module()
    source = latent(offset=8)
    w = plan().plan_windows(720)[1]
    out = av.prepare_av(latent(), w, source=source,
                        repair={'video_ranges': [[20, 24], [28, 30]], 'lock_audio': True}, nested_factory=Nested)
    assert all(torch.equal(x, y) for x, y in zip(av.streams(out), av.streams(source)))
    vm, am = out['noise_mask'].unbind()
    assert int(vm.sum()) == 6 * 2 * 3 and torch.all(am == 0)
    assert out['_h3_long_video_context']['video_ranges'] == [[20, 24], [28, 30]]
    with pytest.raises(ValueError, match='chang'):
        av.validate_result(clone_latent(out), out)
    sampled = clone_latent(out)
    av.streams(sampled)[0][:, :, 20:24] += 1
    assert av.validate_result(sampled, out)['open_video_changed']
    av.streams(sampled)[1][..., -1] += 1
    with pytest.raises(ValueError, match='protected'):
        av.validate_result(sampled, out)


@pytest.mark.parametrize('repair', [
    {'video_ranges': [[11, 14]], 'lock_audio': True},
    {'video_ranges': [[20, 98]], 'lock_audio': True},
    {'video_ranges': [[20, 24], [22, 25]], 'lock_audio': True},
    {'video_ranges': [], 'lock_audio': True},
    {'video_ranges': [[20.0, 24]], 'lock_audio': True},
    {'video_ranges': [[20, 24]], 'lock_audio': False},
])
def test_invalid_or_history_touching_repair_is_rejected(repair):
    with pytest.raises(ValueError):
        module().prepare_av(latent(), plan().plan_windows(720)[1], source=latent(), repair=repair,
                            nested_factory=Nested)


def test_bad_source_geometry_endpoint_or_missing_history_is_rejected():
    av = module()
    w = plan().plan_windows(720)[1]
    with pytest.raises(ValueError):
        av.prepare_av(latent(), w, nested_factory=Nested)
    with pytest.raises(ValueError):
        av.prepare_av(latent(), {**w, 'source_endpoint': 295}, source=latent(), nested_factory=Nested)
    with pytest.raises(ValueError):
        av.prepare_av(latent(), w, source=latent(277), nested_factory=Nested)
    source = latent()
    source['samples'] = Nested((av.streams(source)[0][..., :1], av.streams(source)[1]))
    with pytest.raises(ValueError, match='geometry|shape'):
        av.prepare_av(latent(), w, source=source, nested_factory=Nested)


def test_nested_two_stream_batch_one_and_native_temporal_shape_are_required():
    av = module()
    for value in [None, {}, {'samples': torch.zeros(1)}, {'samples': Nested((torch.zeros(1),))},
                  {'samples': Nested((torch.zeros(2, 4, 97, 2, 3), torch.zeros(1, 8, 2, 547)))},
                  {'samples': Nested((torch.zeros(1, 4, 97, 2, 3), torch.zeros(1, 8, 547)))}]:
        with pytest.raises(ValueError):
            av.streams(value)
    bad = latent()
    bad['samples'] = Nested((av.streams(bad)[0], av.streams(bad)[1][..., :-1]))
    with pytest.raises(ValueError):
        av.prepare_av(bad, plan().plan_windows(720)[0], nested_factory=Nested)


def test_result_rejects_nonfinite_wrong_shape_and_silent_no_change_when_requested():
    av = module()
    out = av.prepare_av(latent(), plan().plan_windows(720)[0], nested_factory=Nested)
    with pytest.raises(ValueError, match='chang'):
        av.validate_result(clone_latent(out), out, require_change=True)
    for which in [0, 1]:
        result = clone_latent(out)
        av.streams(result)[which].flatten()[100] = float('nan')
        with pytest.raises(ValueError, match='finite'):
            av.validate_result(result, out)
    v, a = av.streams(out)
    with pytest.raises(ValueError, match='shape|geometry'):
        av.validate_result({'samples': Nested((v[:, :, :-1], a))}, out)


def test_parent_fingerprint_only_depends_on_the_two_consumed_slices():
    av = module()
    source = latent()
    original = av.slice_fingerprint(source, 294)
    assert len(original) == 64
    altered = clone_latent(source)
    av.streams(altered)[0][:, :, :20] += 1
    av.streams(altered)[1][..., :100] += 1
    assert av.slice_fingerprint(altered, 294) == original
    av.streams(altered)[1][..., 425] += 1
    assert av.slice_fingerprint(altered, 294) != original
    altered = clone_latent(source)
    av.streams(altered)[0][:, :, 75] += 1
    assert av.slice_fingerprint(altered, 294) != original
    half = {'samples': Nested(tuple(t.half() for t in av.streams(source)))}
    assert av.slice_fingerprint(half, 294) != original


def test_cumulative_audio_rounding_avoids_long_timeline_drift_and_never_pads():
    av = module()
    windows = plan().plan_windows(2880)
    cursor = 0
    counts = []
    for w in windows:
        r = av.audio_sample_range(w, 1001)
        assert r['global_start'] == cursor
        assert r['local_end'] - r['local_start'] == r['count']
        counts.append(r['count'])
        cursor = r['global_end']
    assert sum(counts) == round(2880 * 1001 / 24)
    r = av.audio_sample_range(windows[1], 1001)
    with pytest.raises(ValueError, match='audio|samples'):
        av.audio_sample_range(windows[1], 1001, available_samples=r['local_end'] - 1)
