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


def test_color_guard_reduces_reference_excess_chroma_without_changing_luma():
    # The generated clip is deliberately more chromatic than its ordinary
    # REF2VA reference.  The guard must only pull RGB toward each pixel's
    # luminance axis; motion and spatial structure stay in the same tensor.
    reference = torch.full((1, 8, 8, 3), 0.5)
    reference[..., 0] = 0.52
    reference[..., 1] = 0.48
    reference[..., 2] = 0.50
    frames = reference.repeat(3, 1, 1, 1)
    frames[1, 2:6, 2:6, 0] = 0.90
    frames[1, 2:6, 2:6, 1] = 0.10
    frames[2, 1:7, 1:7, 0] = 0.78
    frames[2, 1:7, 1:7, 1] = 0.22

    result, message = MiniMaxH3ColorGuard().apply(
        frames,
        {"mode": "REF2VA", "ref_images": {"ref_image_1": reference}},
        enabled=True,
        strength=1.0,
    )

    luma = torch.tensor([0.2126, 0.7152, 0.0722])
    before_luma = (frames * luma).sum(-1)
    after_luma = (result * luma).sum(-1)
    before_chroma = (frames - before_luma[..., None]).abs().mean()
    after_chroma = (result - after_luma[..., None]).abs().mean()
    assert torch.allclose(after_luma, before_luma, atol=2e-5)
    assert after_chroma < before_chroma
    assert torch.equal(result[1, 2:6, 2:6, 0] > result[1, 2:6, 2:6, 1],
                       frames[1, 2:6, 2:6, 0] > frames[1, 2:6, 2:6, 1])
    assert "色度" in message


def test_color_guard_does_not_change_reference_like_chroma():
    reference = torch.full((1, 8, 8, 3), 0.5)
    reference[..., 0] = 0.56
    reference[..., 1] = 0.44
    frames = reference.repeat(2, 1, 1, 1)
    result, message = MiniMaxH3ColorGuard().apply(
        frames,
        {"mode": "REF2VA", "ref_images": {"ref_image_1": reference}},
        enabled=True,
        strength=1.0,
    )
    assert torch.allclose(result, frames, atol=2e-5)
    assert "参考色度" in message or "原始" in message


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


@pytest.mark.parametrize("case", ["disabled", "no_anchor", "anchor", "fast4"])
def test_color_guard_preserves_execution_lineage(case):
    from nodes.execution_report import attach_report, new_report, report_from, stage_timer
    frames = torch.full((2, 8, 8, 3), 0.2)
    report = new_report({"mode": "T2VA"})
    with stage_timer(report, "decode"):
        pass
    attach_report(frames, report)
    guide = {"first_frame": torch.full((1, 8, 8, 3), 0.4)} if case == "anchor" else {}
    if case == "fast4":
        guide["performance_preset"] = "fast_4step"
    output, _ = MiniMaxH3ColorGuard().apply(frames, guide, enabled=case != "disabled")
    assert report_from(output) == report_from(frames)
    if case in {"anchor", "fast4"}:
        assert output is not frames
        assert report_from(output) is not report_from(frames)
