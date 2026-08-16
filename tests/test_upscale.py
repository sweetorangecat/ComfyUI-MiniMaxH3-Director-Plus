import torch

from nodes.upscale import MiniMaxH3VideoUpscale
from nodes.stream_output import _iter_resized_frame_chunks


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


def test_streaming_resize_yields_target_frames_without_full_target_batch():
    images = torch.rand(5, 8, 8, 3)

    chunks = list(_iter_resized_frame_chunks(images, 16, 12, max_chunk_bytes=16 * 12 * 3))

    assert [chunk.shape for chunk in chunks] == [(1, 12, 16, 3)] * 5
    assert all(chunk.device.type == "cpu" for chunk in chunks)


def test_streaming_resize_preserves_pingpong_order():
    images = torch.linspace(0.0, 1.0, 5 * 2 * 2 * 3, dtype=torch.float32).reshape(5, 2, 2, 3)

    chunks = list(_iter_resized_frame_chunks(images, 2, 2, max_chunk_bytes=2 * 2 * 3, pingpong=True))

    assert len(chunks) == 8
    assert chunks[0].equal(images[0:1])
    assert chunks[-1].equal(images[1:2])
