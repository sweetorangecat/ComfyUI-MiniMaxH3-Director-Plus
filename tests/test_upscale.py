import torch
from torch.utils._python_dispatch import TorchDispatchMode

import pytest

from nodes.upscale import (
    MiniMaxH3VideoUpscale,
    is_x2_upscale_model_name,
    is_x4_upscale_model_name,
    resolve_upscale_model_name,
    validate_frames_for_reconstruction,
)
from nodes.stream_output import (
    _iter_ai_upscale_frame_chunks,
    _iter_lanczos_frame_chunks,
    _iter_resized_frame_chunks,
)


def test_auto_upscale_model_prefers_omnisr_x2_for_two_times_target():
    selected = resolve_upscale_model_name(
        "auto",
        scale_factor=1.9,
        available=["RealESRGAN_x4plus.pth", "RealESRGAN_x2plus.pth", "OmniSR_X2_DIV2K.safetensors"],
    )

    assert selected == "OmniSR_X2_DIV2K.safetensors"


def test_auto_upscale_model_prefers_x2_over_ultrasharp_for_two_times_target():
    selected = resolve_upscale_model_name(
        "auto",
        scale_factor=1.5,
        available=["4x-UltraSharpV2.safetensors", "RealESRGAN_x2plus.pth"],
    )

    assert selected == "RealESRGAN_x2plus.pth"


def test_auto_upscale_model_falls_back_to_x4_when_no_x2_installed():
    selected = resolve_upscale_model_name(
        "auto",
        scale_factor=1.5,
        available=["4x-UltraSharpV2.safetensors"],
    )

    assert selected == "4x-UltraSharpV2.safetensors"


def test_x4_model_name_guard():
    assert is_x4_upscale_model_name("4x-UltraSharpV2.safetensors") is True
    assert is_x4_upscale_model_name("RealESRGAN_x4plus.pth") is True
    assert is_x4_upscale_model_name("OmniSR_X4_DIV2K.safetensors") is True
    assert is_x4_upscale_model_name("RealESRGAN_x2plus.pth") is False
    assert is_x4_upscale_model_name("auto") is False


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


def test_x2_model_name_guard_rejects_x4_models():
    assert is_x2_upscale_model_name("RealESRGAN_x2plus.pth") is True
    assert is_x2_upscale_model_name("OmniSR_X2_DIV2K.safetensors") is True
    assert is_x2_upscale_model_name("RealESRGAN_x4plus.pth") is False
    assert is_x2_upscale_model_name("4x-UltraSharp.pth") is False


@pytest.mark.parametrize("images", [
    torch.full((4, 16, 16, 3), float("nan")),
    torch.full((4, 16, 16, 3), float("inf")),
    torch.zeros((4, 16, 16, 3)),
    torch.full((4, 16, 16, 3), 0.5),
])
def test_invalid_decoded_video_is_rejected_before_reconstruction(images):
    with pytest.raises(ValueError, match="生成阶段输出无效，请重新生成"):
        validate_frames_for_reconstruction(images)


def test_low_saturation_but_structured_video_is_valid():
    ramp = torch.linspace(0.1, 0.9, 16).view(1, 1, 16, 1).repeat(4, 16, 1, 3)
    validate_frames_for_reconstruction(ramp)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("channel", [0, 3], ids=["rgb", "alpha"])
def test_reconstruction_rejects_nonfinite_values_in_late_frames(invalid, channel):
    images = torch.rand(13, 8, 8, 4, dtype=torch.float16)
    images[-1, -1, -1, channel] = invalid

    with pytest.raises(ValueError, match="NaN 或 Inf"):
        validate_frames_for_reconstruction(images)


def test_reconstruction_uses_global_range_across_individually_flat_chunks():
    images = torch.cat([
        torch.full((4, 8, 8, 3), 0.25, dtype=torch.float16),
        torch.full((4, 8, 8, 3), 0.75, dtype=torch.float16),
    ])

    validate_frames_for_reconstruction(images)


def test_reconstruction_dynamic_range_does_not_include_alpha():
    images = torch.full((9, 8, 8, 4), 0.5, dtype=torch.float16)
    images[..., 3] = torch.linspace(0.0, 1.0, 9).view(9, 1, 1)

    with pytest.raises(ValueError, match="动态范围"):
        validate_frames_for_reconstruction(images)


def test_reconstruction_bounds_mask_and_float_conversion_for_long_video():
    images = torch.linspace(0.0, 1.0, 257 * 8 * 8 * 4, dtype=torch.float16).reshape(257, 8, 8, 4)
    temporary_shapes = {torch.bool: [], torch.float32: []}

    class AllocationTrace(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            result = func(*args, **(kwargs or {}))
            results = result if isinstance(result, (tuple, list)) else (result,)
            for tensor in results:
                if isinstance(tensor, torch.Tensor) and tensor.ndim == 4 and tensor.dtype in temporary_shapes:
                    temporary_shapes[tensor.dtype].append(tuple(tensor.shape))
            return result

    with AllocationTrace():
        validate_frames_for_reconstruction(images)

    for shapes in temporary_shapes.values():
        assert shapes, "The real validation must exercise both finite masks and FP32 conversion"
        assert max(shape[0] for shape in shapes) <= 4, shapes


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


@pytest.mark.parametrize("architecture,supports_half,cuda_available,enabled", [
    ("OmniSR", True, True, True),
    ("OmniSR", False, True, False),
    ("OmniSR", True, False, False),
    ("ESRGAN", True, True, False),
    ("", True, True, False),
])
def test_generic_upscale_autocast_respects_model_support(
    monkeypatch, architecture, supports_half, cuda_available, enabled,
):
    import sys
    from types import ModuleType, SimpleNamespace
    from nodes.stream_output import _upscale_image_with_model

    contexts = []
    active = []

    class AutocastContext:
        def __init__(self, device_type, dtype, enabled):
            contexts.append((device_type, dtype, enabled))

        def __enter__(self):
            active.append(True)

        def __exit__(self, *exception):
            active.pop()

    class Upscale:
        def upscale(self, model, image):
            assert active
            return (image,)

    module = ModuleType("comfy_extras.nodes_upscale_model")
    module.ImageUpscaleWithModel = Upscale
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(torch, "autocast", AutocastContext)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    model = SimpleNamespace(
        architecture=SimpleNamespace(name=architecture), supports_half=supports_half,
    )
    image = torch.zeros(1, 4, 4, 3)

    assert _upscale_image_with_model(model, image) is image
    assert contexts == [("cuda", torch.float16, enabled)]
    assert active == []
