"""Native phase references protect real unused video, never approve a movie."""
from copy import deepcopy

import pytest
import torch

from nodes import long_video_av as av
from nodes.long_video_plan import plan_windows
from test_long_video_av import Nested, clone_latent, latent


def reference(**changes):
    return {'source_take': 'reviewed-tail', 'source_steps': [87, 97], 'target_steps': [77, 87],
            'review': {'name': 'reviews/unused-tail.json', 'sha256': 'a' * 64, 'bytes': 1000}, **changes}


def recipe(**changes):
    return {'video_ranges': [[22, 97]], 'lock_audio': True, 'video_reference': reference(), **changes}


def test_reference_is_part_of_the_canonical_immutable_repair_recipe():
    original = recipe()
    before = deepcopy(original)
    normalized = av.normalize_repair(plan_windows(720)[1], original)
    assert normalized == original == before
    normalized['video_reference']['source_steps'][0] = 92
    assert original == before


def test_reference_copies_only_declared_native_video_before_sampling_and_locks_it():
    failed, target, actual = latent(offset=10), latent(offset=-4), latent(offset=5000)
    before = [t.clone() for t in av.streams(actual)]
    prepared = av.prepare_av(target, plan_windows(720)[1], source=failed, repair=recipe(),
                             nested_factory=Nested, video_reference_source=actual)
    video, audio = av.streams(prepared)
    vm, am = prepared['noise_mask'].unbind()
    assert torch.equal(video[:, :, :77], av.streams(failed)[0][:, :, :77])
    assert torch.equal(video[:, :, 77:87], av.streams(actual)[0][:, :, 87:97])
    assert torch.equal(video[:, :, 87:], av.streams(failed)[0][:, :, 87:])
    assert torch.equal(audio, av.streams(failed)[1]) and torch.all(am == 0)
    assert torch.all(vm[:, :, :22] == 0) and torch.all(vm[:, :, 77:87] == 0)
    assert torch.all(vm[:, :, 22:77] == 1) and torch.all(vm[:, :, 87:] == 1)
    assert prepared['_h3_long_video_context']['video_reference'] == reference()
    assert all(torch.equal(a, b) for a, b in zip(before, av.streams(actual)))
    sampled = clone_latent(prepared)
    av.streams(sampled)[0].add_(vm.expand_as(video))
    result = av.validate_result(sampled, prepared)
    assert result['protected_video_exact'] and result['protected_audio_exact']
    assert result['content_review'] == 'pending'
    av.streams(sampled)[0][:, :, 77] += 1
    with pytest.raises(ValueError, match='protected'):
        av.validate_result(sampled, prepared)


@pytest.mark.parametrize('change', [
    {'target_steps': [7, 17]}, {'target_steps': [78, 88]},
    {'source_steps': [88, 98]}, {'source_steps': [0, 10]},
    {'source_steps': [87, 92]}, {'target_steps': [92, 102]},
    {'source_steps': [True, 11]}, {'source_take': ''},
    {'review': {'name': 'reviews/x.json', 'sha256': 'a', 'bytes': 1}},
    {'source_path': 'arbitrary/video.latent'},
])
def test_bad_reference_geometry_or_identity_is_rejected(change):
    with pytest.raises(ValueError):
        av.normalize_repair(plan_windows(720)[1], recipe(video_reference=reference(**change)))


def test_reference_must_be_wholly_inside_positive_video_repair_and_leave_free_video():
    for ranges in [[[22, 77]], [[22, 82]], [[77, 87]]]:
        with pytest.raises(ValueError):
            av.normalize_repair(plan_windows(720)[1], recipe(video_ranges=ranges))
    normalized = av.normalize_repair(plan_windows(720)[1], recipe(
        video_ranges=[[22, 77]], video_weights=[[77, 87, .5], [87, 97, .25]]))
    assert normalized['video_reference'] == reference()


@pytest.mark.parametrize('kind', ['missing', 'extra_without_repair', 'wrong_dtype', 'short_source'])
def test_reference_requires_explicit_matching_actual_source(kind):
    actual = latent(dtype=torch.float16) if kind == 'wrong_dtype' else latent(277) if kind == 'short_source' else latent()
    kwargs = {'repair': recipe(), 'video_reference_source': actual}
    if kind == 'missing':
        kwargs.pop('video_reference_source')
    if kind == 'extra_without_repair':
        kwargs['repair'] = None
    with pytest.raises(ValueError):
        av.prepare_av(latent(), plan_windows(720)[1], source=latent(), nested_factory=Nested, **kwargs)
