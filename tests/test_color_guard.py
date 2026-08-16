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
def test_color_guard_does_not_use_generated_first_frame_as_exposure_anchor(mode):
    frames = torch.stack((torch.full((8, 8, 3), 0.25), torch.full((8, 8, 3), 0.5)))
    result, _ = MiniMaxH3ColorGuard().apply(frames, {"mode": mode}, enabled=True, strength=1.0)
    assert torch.allclose(result, frames)


def test_fast4_quality_guard_lifts_dark_output_without_hard_keyframe():
    frames = torch.full((2, 16, 16, 3), 0.2)
    result, message = MiniMaxH3ColorGuard().apply(
        frames,
        {"mode": "T2VA", "performance_preset": "fast_4step"},
        enabled=True,
        strength=1.0,
    )
    assert result.mean() > frames.mean()
    assert "极速4步" in message


def test_color_guard_exposes_bounded_frame_chunks_for_long_videos():
    chunks = MiniMaxH3ColorGuard._chunk_ranges(19, chunk_size=8)
    assert chunks == [(0, 8), (8, 16), (16, 19)]


def test_color_guard_offloads_large_cuda_outputs_to_system_memory():
    large_video_bytes = 362 * 1024 * 1792 * 3 * 4
    assert MiniMaxH3ColorGuard._requires_cpu_output(large_video_bytes, is_cuda=True)
    assert not MiniMaxH3ColorGuard._requires_cpu_output(large_video_bytes, is_cuda=False)
    assert not MiniMaxH3ColorGuard._requires_cpu_output(64 * 1024**2, is_cuda=True)
