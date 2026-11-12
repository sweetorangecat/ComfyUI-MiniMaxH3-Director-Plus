"""CPU evidence for bounded two-stream repair and true-clock forward AV."""
from copy import deepcopy

import pytest
import torch

from nodes import long_video_av as av
from nodes.long_video_plan import plan_windows
from test_long_video_av import Nested, latent, clone_latent


@pytest.mark.parametrize('total,index,forward', [(2160,2,17),(2332,-1,0)])
def test_film_seconds_accept_exact_derived_float_free_and_delivery_boundaries(total,index,forward):
    window = plan_windows(total)[index]
    start = (window['new_start'] + forward) / 24
    end = window['new_end'] / 24
    recipe = av.repair_seconds(window,start,end,forward_frames=forward)
    assert recipe['conditioning_extent']['requested_seconds'] == [start,end]
    assert recipe['conditioning_extent']['frame_isolation_guaranteed'] is False


def test_seconds_roundoff_tolerance_does_not_accept_real_protected_overlap_or_overshoot():
    window = plan_windows(2160)[2]
    free = (window['new_start']+17)/24
    with pytest.raises(ValueError,match='protected|source'):
        av.repair_seconds(window,free-1e-8,window['new_end']/24,forward_frames=17)
    window = plan_windows(2332)[-1]
    with pytest.raises(ValueError,match='inside|interval|delivery'):
        av.repair_seconds(window,window['new_start']/24,window['new_end']/24+1e-8)


@pytest.mark.parametrize('mode', ['video', 'audio', 'joint', 'soft'])
def test_bounded_repair_changes_each_requested_stream_and_locks_every_other_value(mode):
    recipe = {'video_ranges': [], 'lock_audio': mode in {'video', 'soft'}}
    if mode in {'video', 'joint'}:
        recipe['video_ranges'] = [[20, 24]]
    if mode in {'audio', 'joint'}:
        recipe['audio_ranges'] = [[100, 120]]
    if mode == 'soft':
        recipe['video_weights'] = [[20, 24, .25]]
    source = latent(offset=8)
    before = deepcopy(recipe)
    prepared = av.prepare_av(latent(), plan_windows(720)[1], source, recipe, Nested)
    assert recipe == before
    vm, am = prepared['noise_mask'].unbind()
    if mode == 'soft':
        assert torch.all(vm[:, :, 20:24] == .25)
    changed = clone_latent(prepared)
    for tensor, mask in zip(av.streams(changed), (vm, am)):
        tensor.add_(mask.expand_as(tensor))
    result = av.validate_result(changed, prepared)
    assert result['open_video_changed'] == (mode != 'audio')
    assert result['open_audio_changed'] == (mode in {'audio', 'joint'})
    unchanged = clone_latent(prepared)
    with pytest.raises(ValueError, match='Repair did not change'):
        av.validate_result(unchanged, prepared)
    if mode == 'joint':
        av.streams(unchanged)[0][:, :, 20] += 1
        with pytest.raises(ValueError, match='audio'):
            av.validate_result(unchanged, prepared)


@pytest.mark.parametrize('extra', [
    {'anchors': []}, {'source_path': 'x'}, {'quiet_source': {}},
    {'video_weights': [[20, 24, float('nan')]]},
    {'video_weights': [[20, 24, 1.01]]},
    {'video_weights': [[20, 24, -1]]},
    {'video_weights': [[20, 24, 0]]},
    {'video_ranges': [[20, 24]], 'video_weights': [[23, 25, .5]]},
    {'video_ranges': [], 'audio_ranges': [[64, 70]], 'lock_audio': False},
    {'video_ranges': [], 'audio_ranges': [[100, 110]], 'lock_audio': True},
])
def test_invalid_advanced_repair_is_never_ignored(extra):
    recipe = {'video_ranges': [], 'lock_audio': True, **extra}
    with pytest.raises(ValueError):
        av.normalize_repair(plan_windows(720)[1], recipe)


@pytest.mark.parametrize('forward,vr,ar,head_v,head_a', [
    (17, [75, 92], [425, 518], 17, 93),
    (34, [75, 97], [425, 547], 22, 122),
])
def test_forward_conditions_both_streams_with_true_native_clocks(forward, vr, ar, head_v, head_a):
    source, target = latent(offset=19), latent(offset=-2)
    prepared = av.prepare_av(target, plan_windows(720)[1], source=source,
                             forward_frames=forward, nested_factory=Nested)
    pv, pa = av.streams(prepared)
    sv, sa = av.streams(source)
    assert torch.equal(pv[:, :, :head_v], sv[:, :, vr[0]:vr[1]])
    assert torch.equal(pa[..., :head_a], sa[..., ar[0]:ar[1]])
    vm, am = prepared['noise_mask'].unbind()
    assert torch.all(vm[:, :, :head_v] == 0) and torch.all(vm[:, :, head_v:] == 1)
    assert torch.all(am[..., :head_a] == 0) and torch.all(am[..., head_a:] == 1)
    fingerprint = av.slice_fingerprint(source, 294, 39, forward_frames=forward)
    # Every temporal slice in either consumed stream participates in the hash.
    for stream, start, end in ((0, *vr), (1, *ar)):
        for tick in range(start, end):
            changed = clone_latent(source)
            value = av.streams(changed)[stream]
            if stream == 0:
                value[:, :, tick] += 1
            else:
                value[..., tick] += 1
            assert av.slice_fingerprint(changed, 294, 39, forward_frames=forward) != fingerprint
    changed = clone_latent(source)
    av.streams(changed)[0][:, :, :vr[0]] += 1
    av.streams(changed)[1][..., :ar[0]] += 1
    assert av.slice_fingerprint(changed, 294, 39, forward_frames=forward) == fingerprint


@pytest.mark.parametrize('forward', [-17, 1, True, 51, 255])
def test_unavailable_or_invalid_forward_never_copies_a_tail(forward):
    with pytest.raises(ValueError):
        av.prepare_av(latent(), plan_windows(720)[1], latent(), nested_factory=Nested,
                      forward_frames=forward)


def test_forward_prefix_is_protected_from_hard_and_soft_repair():
    window = plan_windows(720)[1]
    for recipe in ({'video_ranges': [[20, 24]], 'lock_audio': True},
                   {'video_ranges': [], 'video_weights': [[20, 24, .5]], 'lock_audio': True},
                   {'video_ranges': [], 'audio_ranges': [[120, 125]], 'lock_audio': False}):
        with pytest.raises(ValueError, match='protected'):
            av.normalize_repair(window, recipe, forward_frames=34)


def test_zero_weight_transition_is_closed_when_another_span_is_open():
    recipe = {'video_weights': [[20, 22, 0], [22, 24, .25]], 'lock_audio': True}
    canonical = av.normalize_repair(plan_windows(720)[1], recipe)
    assert canonical['video_weights'] == [[20, 22, 0.0], [22, 24, .25]]
    prepared = av.prepare_av(latent(), plan_windows(720)[1], latent(offset=4), recipe, Nested)
    vm, am = prepared['noise_mask'].unbind()
    assert torch.all(vm[:, :, 20:22] == 0) and torch.all(vm[:, :, 22:24] == .25)
    changed = clone_latent(prepared)
    av.streams(changed)[0][:, :, 23] += 1
    assert av.validate_result(changed, prepared)['open_video_changed']
    av.streams(changed)[0][:, :, 20] += 1
    with pytest.raises(ValueError, match='protected'):
        av.validate_result(changed, prepared)


@pytest.mark.parametrize('joint', [False, True])
def test_positive_video_weight_cannot_underflow_to_closed_mask_even_in_joint_repair(joint):
    recipe = {'video_weights': [[20, 24, 1e-50]], 'lock_audio': not joint}
    if joint:
        recipe['audio_ranges'] = [[100, 120]]
    with pytest.raises(ValueError, match='float32'):
        av.normalize_repair(plan_windows(720)[1], recipe)
    with pytest.raises(ValueError, match='float32'):
        av.prepare_av(latent(), plan_windows(720)[1], latent(), recipe, Nested)


def test_seconds_repair_snaps_outwards_on_native_groups_and_keeps_requested_extent():
    window = plan_windows(720)[1]
    repair = av.repair_seconds(window, 15, 15.5, 'joint')
    assert repair['video_ranges'] == [[27, 37]]
    assert repair['audio_ranges'] == [[175, 195]]
    assert repair['lock_audio'] is False
    extent = repair['conditioning_extent']
    assert extent['requested_seconds'] == [15, 15.5]
    assert extent['video_local_frames'] == [90, 124]
    assert extent['audio_local_ticks'] == [175, 195]
    assert extent['video_seconds'] == pytest.approx([14.375, 15.791666666666666])
    assert extent['audio_seconds'] == [15, 15.5]
    assert extent['frame_isolation_guaranteed'] is False


def test_seconds_repair_rejects_requested_overlap_with_forward_instead_of_clipping():
    window = plan_windows(720)[1]
    with pytest.raises(ValueError, match='protected.*parent'):
        av.repair_seconds(window, 12.25, 13.5, 'video', forward_frames=17)
    repair = av.repair_seconds(window, (294 + 17) / 24 + .01, 13.5, 'video', forward_frames=17)
    assert repair['video_ranges'][0][0] >= 17


@pytest.mark.parametrize('start,end,mode', [(float('nan'), 15, 'video'), (15, float('inf'), 'audio'),
    (True, 15, 'video'), (12, 15, 'video'), (15, 13, 'joint'), (15, 100, 'video'), (15, 16, 'spatial')])
def test_seconds_repair_rejects_nonfinite_outside_and_unknown_modes(start, end, mode):
    with pytest.raises(ValueError):
        av.repair_seconds(plan_windows(720)[1], start, end, mode)
