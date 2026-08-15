import torch
import pytest

from nodes.color_guard import MiniMaxH3ColorGuard


def test_color_guard_corrects_temporal_exposure_drift_against_first_frame():
    anchor = torch.full((1, 8, 8, 3), 0.25)
    bright = torch.full((1, 8, 8, 3), 0.5)
    frames = torch.cat((anchor, bright), dim=0)
    result, message = MiniMaxH3ColorGuard().apply(
        frames,
        {"first_frame": anchor},
        enabled=True,
        strength=1.0,
    )
    assert torch.allclose(result[0], anchor)
    assert result[1].mean() < bright.mean()
    assert "曝光" in message


def test_color_guard_does_not_treat_ref2va_reference_as_first_frame():
    frames = torch.stack((torch.full((8, 8, 3), 0.25), torch.full((8, 8, 3), 0.5)))
    result, _ = MiniMaxH3ColorGuard().apply(
        frames,
        {"mode": "REF2VA", "ref_images": {"ref_image_1": torch.full((1, 8, 8, 3), 0.1)}},
        enabled=True,
        strength=1.0,
    )
    assert torch.allclose(result[0], frames[0])


def test_color_guard_is_noop_when_disabled():
    frames = torch.rand(3, 8, 8, 3)
    result, message = MiniMaxH3ColorGuard().apply(frames, {}, enabled=False)
    assert torch.equal(result, frames)
    assert "未启用" in message


@pytest.mark.parametrize("mode", ["T2VA", "FL2VA", "REF2VA"])
def test_color_guard_uses_generated_first_frame_when_mode_has_no_hard_keyframe(mode):
    frames = torch.stack((torch.full((8, 8, 3), 0.25), torch.full((8, 8, 3), 0.5)))
    result, _ = MiniMaxH3ColorGuard().apply(frames, {"mode": mode}, enabled=True, strength=1.0)
    assert torch.allclose(result[0], frames[0])
    assert result[1].mean() < frames[1].mean()
