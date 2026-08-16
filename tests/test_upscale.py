import torch

from nodes.upscale import MiniMaxH3VideoUpscale


def test_video_upscale_passes_through_when_not_requested():
    images = torch.rand(3, 8, 8, 3)
    result, message = MiniMaxH3VideoUpscale().apply(images, {"upscale_required": False})
    assert result is images
    assert "未启用" in message


def test_video_upscale_returns_target_size_on_cpu_in_bounded_chunks():
    images = torch.rand(5, 8, 8, 3, dtype=torch.float16)
    result, message = MiniMaxH3VideoUpscale().apply(
        images,
        {
            "upscale_required": True,
            "target_width": 16,
            "target_height": 12,
        },
    )
    assert result.shape == (5, 12, 16, 3)
    assert result.dtype == images.dtype
    assert result.device.type == "cpu"
    assert "CPU" in message


def test_video_upscale_uses_four_frame_chunks():
    assert MiniMaxH3VideoUpscale._chunk_ranges(9, chunk_size=4) == [
        (0, 4), (4, 8), (8, 9)
    ]
