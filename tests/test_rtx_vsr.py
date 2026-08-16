import sys
from types import SimpleNamespace

import pytest
import torch

from nodes.rtx_vsr_stream import (
    VsrFrameProcessor,
    load_vsr_api,
    resolve_vsr_quality,
)


REQUIREMENTS_PATH = (
    "D:/ComfyUI_windows_portable-G313/ComfyUI/custom_nodes/"
    "ComfyUI-DaSiWa-Nodes/requirements.txt"
)
EMBEDDED_INSTALL_COMMAND = (
    "D:/ComfyUI_windows_portable-G313/python_embeded/python.exe -m pip install -r "
    + REQUIREMENTS_PATH
)


def test_load_vsr_api_reports_actionable_embedded_python_install_steps(monkeypatch):
    monkeypatch.setitem(sys.modules, "nvvfx", None)

    with pytest.raises(RuntimeError) as error:
        load_vsr_api()

    message = str(error.value)
    assert sys.executable in message
    assert REQUIREMENTS_PATH in message
    assert EMBEDDED_INSTALL_COMMAND in message
    assert "nvidia-vfx" in message
    assert "NVIDIA Broadcast SDK" in message
    assert "重启 ComfyUI" in message
    assert f'"{sys.executable}" -c "import nvvfx; print(nvvfx.VideoSuperRes)"' in message


def test_load_vsr_api_locates_video_super_res_and_effect_quality(monkeypatch):
    video_super_res = object()
    quality_level = object()
    fake_nvvfx = SimpleNamespace(
        VideoSuperRes=video_super_res,
        effects=SimpleNamespace(QualityLevel=quality_level),
    )
    monkeypatch.setitem(sys.modules, "nvvfx", fake_nvvfx)

    assert load_vsr_api() == (video_super_res, quality_level)


@pytest.mark.parametrize("name, expected", [("HIGH", "high"), ("ULTRA", "ultra")])
def test_resolve_vsr_quality_maps_only_supported_levels(name, expected):
    quality_level = SimpleNamespace(HIGH="high", ULTRA="ultra", MEDIUM="medium")

    assert resolve_vsr_quality(quality_level, name) == expected


@pytest.mark.parametrize("quality", ["MEDIUM", "high", "", None])
def test_resolve_vsr_quality_rejects_other_values(quality):
    quality_level = SimpleNamespace(HIGH="high", ULTRA="ultra", MEDIUM="medium")

    with pytest.raises(ValueError, match="HIGH.*ULTRA"):
        resolve_vsr_quality(quality_level, quality)


def test_processor_uses_compatible_constructor_and_loads_dimensions():
    attempts = []

    class FakeEffect:
        def __init__(self):
            self.loaded = False

        def load(self):
            self.loaded = True

        def close(self):
            pass

    effect = FakeEffect()

    def video_super_res(*args, **kwargs):
        attempts.append((args, kwargs))
        if args == ("high",) and kwargs == {"device": 2}:
            return effect
        raise TypeError("unsupported constructor")

    processor = VsrFrameProcessor(
        (video_super_res, SimpleNamespace(HIGH="high", ULTRA="ultra")),
        "HIGH",
        2,
        1920,
        1080,
    )

    assert attempts == [
        ((), {"quality": "high", "device": 2}),
        (("high",), {"device": 2}),
    ]
    assert effect.output_width == 1920
    assert effect.output_height == 1080
    assert effect.loaded is True
    processor.close()


def test_process_returns_cloned_hwc_float32_cpu_frame(monkeypatch):
    cuda_calls = []
    run_inputs = []
    sdk_output = torch.full((3, 6, 8), 0.75, dtype=torch.float32)

    class FakeStream:
        def synchronize(self):
            cuda_calls.append("before")

    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: FakeStream())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: cuda_calls.append("after"))
    original_to = torch.Tensor.to

    def fake_to(tensor, *args, **kwargs):
        device = kwargs.get("device", args[0] if args else None)
        if device is not None and torch.device(device).type == "cuda":
            dtype = kwargs.get("dtype")
            return tensor.to(dtype=dtype) if dtype is not None else tensor
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", fake_to)

    class FakeEffect:
        def run(self, frame):
            run_inputs.append(frame)
            return SimpleNamespace(image=sdk_output)

        def close(self):
            pass

    processor = VsrFrameProcessor(
        (lambda *args, **kwargs: FakeEffect(), SimpleNamespace(HIGH="high", ULTRA="ultra")),
        "HIGH",
        0,
        8,
        6,
    )

    result = processor.process(torch.zeros(3, 2, 4, dtype=torch.float16).transpose(1, 2))
    sdk_output.zero_()

    assert cuda_calls == ["before", "after"]
    assert len(run_inputs) == 1
    assert run_inputs[0].shape == (3, 4, 2)
    assert run_inputs[0].dtype == torch.float32
    assert run_inputs[0].is_contiguous()
    assert result.shape == (6, 8, 3)
    assert result.dtype == torch.float32
    assert result.device.type == "cpu"
    assert result.is_contiguous()
    assert torch.all(result == 0.75)
    processor.close()


def test_context_manager_closes_effect_once_when_processing_raises():
    calls = []

    class FakeEffect:
        def close(self):
            calls.append("close")

    with pytest.raises(RuntimeError, match="encode failed"):
        with VsrFrameProcessor(
            (lambda *args, **kwargs: FakeEffect(), SimpleNamespace(HIGH="high", ULTRA="ultra")),
            "HIGH",
            0,
            8,
            6,
        ) as processor:
            assert processor is not None
            raise RuntimeError("encode failed")

    processor.close()
    assert calls == ["close"]


def test_close_uses_destroy_when_close_is_unavailable():
    calls = []

    class FakeEffect:
        def destroy(self):
            calls.append("destroy")

    processor = VsrFrameProcessor(
        (lambda *args, **kwargs: FakeEffect(), SimpleNamespace(HIGH="high", ULTRA="ultra")),
        "HIGH",
        0,
        8,
        6,
    )

    processor.close()
    processor.close()

    assert calls == ["destroy"]
