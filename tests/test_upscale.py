import torch

import pytest

from nodes.upscale import (
    MiniMaxH3VideoUpscale,
    resolve_upscale_model_name,
)
from nodes.stream_output import (
    _iter_ai_upscale_frame_chunks,
    _iter_lanczos_frame_chunks,
    _iter_resized_frame_chunks,
)


def test_auto_upscale_model_prefers_x2_for_two_times_target():
    selected = resolve_upscale_model_name(
        "auto",
        scale_factor=1.9,
        available=["RealESRGAN_x4plus.pth", "RealESRGAN_x2plus.pth", "OmniSR_X2_DIV2K.safetensors"],
    )

    assert selected == "RealESRGAN_x2plus.pth"


def test_auto_upscale_model_prefers_x4_for_four_k_target():
    selected = resolve_upscale_model_name(
        "auto",
        scale_factor=2.9,
        available=["RealESRGAN_x2plus.pth", "OmniSR_X4_DIV2K.safetensors", "RealESRGAN_x4plus.pth"],
    )

    assert selected == "OmniSR_X4_DIV2K.safetensors"


def test_manual_upscale_model_must_exist():
    with pytest.raises(ValueError, match="不存在"):
        resolve_upscale_model_name("missing.safetensors", 2.0, available=["RealESRGAN_x2plus.pth"])


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


def test_streaming_lanczos_resize_yields_exact_target_frames():
    images = torch.rand(3, 4, 4, 3)

    chunks = list(_iter_lanczos_frame_chunks(images, 8, 6, max_chunk_bytes=8 * 6 * 3))

    assert [chunk.shape for chunk in chunks] == [(1, 6, 8, 3)] * 3
    assert all(chunk.device.type == "cpu" for chunk in chunks)


def test_streaming_ai_upscale_uses_model_and_releases_it(monkeypatch):
    images = torch.rand(2, 4, 4, 3)
    calls = []

    class FakePatcher:
        pass

    model = type("FakeModel", (), {"patcher": FakePatcher()})()

    monkeypatch.setattr("nodes.stream_output.resolve_upscale_model_name", lambda *args, **kwargs: "fake.pth")
    monkeypatch.setattr("nodes.stream_output._load_upscale_model", lambda name: model)
    monkeypatch.setattr("nodes.stream_output._release_upscale_model", lambda value: calls.append(value))
    monkeypatch.setattr(
        "nodes.stream_output._upscale_image_with_model",
        lambda value, image: image.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2),
    )

    chunks = list(_iter_ai_upscale_frame_chunks(images, 7, 6, max_chunk_bytes=7 * 6 * 3))

    assert [chunk.shape for chunk in chunks] == [(1, 6, 7, 3)] * 2
    assert calls == [model]


def test_streaming_ai_upscale_batches_frames_before_model_load_check(monkeypatch):
    images = torch.rand(4, 4, 4, 3)
    calls = []
    model = type("FakeModel", (), {"patcher": object()})()

    monkeypatch.setattr("nodes.stream_output.resolve_upscale_model_name", lambda *args, **kwargs: "fake.pth")
    monkeypatch.setattr("nodes.stream_output._load_upscale_model", lambda name: model)
    monkeypatch.setattr("nodes.stream_output._release_upscale_model", lambda value: None)

    def upscale(value, batch):
        calls.append(batch.shape[0])
        return batch.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)

    monkeypatch.setattr("nodes.stream_output._upscale_image_with_model", upscale)

    chunks = list(_iter_ai_upscale_frame_chunks(images, 8, 8, max_chunk_bytes=4 * 8 * 8 * 3))

    assert calls == [4]
    assert [chunk.shape for chunk in chunks] == [(4, 8, 8, 3)]
