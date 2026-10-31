import json

import pytest
import torch

from nodes.color_guard import MiniMaxH3ColorGuard
from nodes.execution_report import new_report, report_from


def _scene():
    y, x = torch.meshgrid(torch.linspace(0, 1, 48), torch.linspace(0, 1, 64), indexing="ij")
    luma = 0.16 + 0.62 * (0.5 + 0.3 * torch.sin(x * 12) + 0.2 * torch.cos(y * 9))
    return torch.stack((luma + 0.04 * x, luma - 0.015 * x,
                        luma + 0.045 * y), dim=-1).unsqueeze(0)


def _drift(reference):
    return (reference.pow(1.22) * torch.tensor([1.025, 0.99, 0.98])).clamp(0, 1)


def test_reference_grade_restores_tone_and_cast_using_the_matching_scene():
    reference = _scene()
    generated = _drift(reference).repeat(4, 1, 1, 1)
    unrelated_identity = reference.flip(2).pow(0.6)
    guide = {"mode": "REF2VA", "ref_images": {
        "ref_image_1": unrelated_identity, "ref_image_2": reference,
    }}
    output, message = MiniMaxH3ColorGuard().apply(generated, guide)
    before = (generated - reference).abs().mean()
    after = (output - reference).abs().mean()
    assert after < before * 0.4
    assert output.shape == generated.shape
    assert torch.equal(output[0], output[-1])
    evidence = report_from(output)["color_guard"]
    assert evidence["reference"] == "ref_image_2"
    assert evidence["method"] == "scene_reference_v1"
    assert "场景" in message
    assert new_report(guide, output)["color_guard"] == evidence
    json.dumps(evidence, allow_nan=False)


def test_reference_grade_is_fixed_across_lighting_changes_and_does_not_mix_frames():
    reference = _scene()
    matching = _drift(reference)
    later = (matching.flip(2) * 0.55 + torch.tensor([0.08, 0.025, 0.01])).clamp(0, 1)
    clip = torch.cat((matching, matching, later, matching, later))
    output, _ = MiniMaxH3ColorGuard().apply(clip, {"mode": "REF2VA", "ref_images": {"scene": reference}})
    assert torch.equal(output[0], output[1]) and torch.equal(output[0], output[3])
    assert torch.equal(output[2], output[4])
    assert output[2].mean() < output[0].mean() - 0.1
    assert output[2, ..., 0].mean() > output[2, ..., 2].mean()
    assert torch.equal(clip[0], matching[0])  # The cached decode is immutable.


def test_scene_grade_preserves_black_white_and_alpha():
    reference = _scene()
    generated = _drift(reference).repeat(3, 1, 1, 1)
    generated[:, 0, 0] = 0
    generated[:, 0, 1] = 1
    rgba = torch.cat((generated, torch.full_like(generated[..., :1], 0.37)), dim=-1)
    output, _ = MiniMaxH3ColorGuard().apply(rgba, {"ref_images": {"scene": reference}})
    assert torch.equal(output[:, 0, 0, :3], torch.zeros(3, 3))
    assert torch.equal(output[:, 0, 1, :3], torch.ones(3, 3))
    assert torch.equal(output[..., 3], rgba[..., 3])


def test_reference_statistics_downsample_before_gathering_frames(monkeypatch):
    # A 24-frame full-resolution index_select was previously allocated before
    # downsampling: this reproduces the costly operation without needing 8GB.
    def reject_full_batch(*args, **kwargs):
        raise AssertionError("full-resolution gather")
    monkeypatch.setattr(torch.Tensor, "index_select", reject_full_batch)
    images = _scene().repeat(30, 1, 1, 1)
    assert MiniMaxH3ColorGuard._chroma_metric(images) > 0


@pytest.mark.parametrize("strength", [0.0, 1.0])
def test_matching_scene_does_not_add_a_grade_when_colors_already_match(strength):
    reference = _scene()
    frames = reference.repeat(3, 1, 1, 1)
    output, _ = MiniMaxH3ColorGuard().apply(frames, {"ref_images": {"scene": reference}}, strength=strength)
    assert torch.allclose(output, frames, atol=2e-6)
