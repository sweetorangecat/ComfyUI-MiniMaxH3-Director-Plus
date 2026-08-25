import logging
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from nodes import rtx_vsr_stream
from nodes.rtx_vsr_stream import (
    DeblurVsrFrameProcessor,
    VsrFrameProcessor,
    _dasiwa_requirements_path,
    load_vsr_api,
    probe_vsr_deblur_chain,
    probe_vsr_capability,
    resolve_vsr_quality,
)


REQUIREMENTS_PATH = str(_dasiwa_requirements_path())
EMBEDDED_INSTALL_COMMAND = (
    f'"{sys.executable}" -m pip install -r "{REQUIREMENTS_PATH}"'
)
REAL_CUDA_DEVICE_CONTEXT = torch.cuda.device


@pytest.fixture(autouse=True)
def mock_cuda_lifecycle(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: None)


def _mock_probe_cuda(monkeypatch):
    empty_cache_calls = []

    def fail_if_empty_cache_is_called():
        empty_cache_calls.append(True)
        raise AssertionError("probe must not call empty_cache")

    monkeypatch.setattr(rtx_vsr_stream.torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(rtx_vsr_stream.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        rtx_vsr_stream.torch.cuda,
        "empty_cache",
        fail_if_empty_cache_is_called,
    )
    return empty_cache_calls


def _mock_probe_input(monkeypatch):
    original_zeros = torch.zeros
    monkeypatch.setattr(
        rtx_vsr_stream.torch,
        "zeros",
        lambda *args, **kwargs: original_zeros((3, 360, 640), dtype=torch.float32),
    )


def test_load_vsr_api_reports_actionable_embedded_python_install_steps(monkeypatch):
    monkeypatch.setattr(rtx_vsr_stream.sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "nvvfx", None)

    with pytest.raises(RuntimeError) as error:
        load_vsr_api()

    message = str(error.value)
    assert sys.executable in message
    assert EMBEDDED_INSTALL_COMMAND in message
    assert REQUIREMENTS_PATH in message
    assert EMBEDDED_INSTALL_COMMAND in message
    assert "nvidia-vfx" in message
    assert "NVIDIA Broadcast SDK" in message
    assert "重启 ComfyUI" in message
    assert f'"{sys.executable}" -c "import nvvfx; print(nvvfx.VideoSuperRes)"' in message


def test_linux_dependency_error_prefers_driver_and_wheel_guidance(monkeypatch):
    monkeypatch.setattr(rtx_vsr_stream.sys, "platform", "linux")
    monkeypatch.setitem(sys.modules, "nvvfx", None)

    with pytest.raises(RuntimeError) as error:
        load_vsr_api()

    message = str(error.value)
    assert "570.190+" in message
    assert "580.82+" in message
    assert "590.44+" in message
    assert "nvidia-vfx" in message
    assert "Broadcast SDK" not in message


def test_windows_dependency_error_keeps_video_effects_runtime_guidance(monkeypatch):
    monkeypatch.setattr(rtx_vsr_stream.sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "nvvfx", None)

    with pytest.raises(RuntimeError) as error:
        load_vsr_api()

    assert "NVIDIA Broadcast SDK/Video Effects" in str(error.value)


def test_linux_probe_failure_includes_platform_runtime_guidance(monkeypatch):
    monkeypatch.setattr(rtx_vsr_stream.sys, "platform", "linux")
    monkeypatch.setattr(
        rtx_vsr_stream,
        "load_vsr_api",
        lambda: (_ for _ in ()).throw(RuntimeError("load failed")),
    )

    with pytest.raises(RuntimeError) as error:
        probe_vsr_capability("HIGH", 0)

    message = str(error.value)
    assert "570.190+" in message
    assert "580.82+" in message
    assert "590.44+" in message
    assert "nvidia-vfx" in message
    assert "Broadcast SDK" not in message


def test_probe_does_not_repeat_dependency_runtime_guidance(monkeypatch):
    monkeypatch.setattr(rtx_vsr_stream.sys, "platform", "linux")
    monkeypatch.setitem(sys.modules, "nvvfx", None)

    with pytest.raises(RuntimeError) as error:
        probe_vsr_capability("HIGH", 0)

    message = str(error.value)
    assert "未找到 nvvfx 模块" in message
    assert message.count("570.190+") == 1
    assert isinstance(error.value.__cause__, RuntimeError)


def test_load_vsr_api_locates_video_super_res_and_effect_quality(monkeypatch):
    video_super_res = object()
    quality_level = object()
    fake_nvvfx = SimpleNamespace(
        VideoSuperRes=video_super_res,
        effects=SimpleNamespace(QualityLevel=quality_level),
    )
    monkeypatch.setitem(sys.modules, "nvvfx", fake_nvvfx)

    assert load_vsr_api() == (video_super_res, quality_level)


@pytest.mark.parametrize("quality", ["HIGH", "ULTRA", "HIGHBITRATE_ULTRA"])
def test_probe_vsr_capability_requests_cuda_360p_frame_at_the_requested_quality(monkeypatch, quality, caplog):
    calls = []
    original_zeros = torch.zeros

    class FakeProcessor:
        def __init__(self, api, received_quality, device_id, output_width, output_height):
            calls.append(("create", api, received_quality, device_id, output_width, output_height))

        def process(self, frame):
            calls.append(("process", frame))
            return original_zeros((720, 1280, 3), dtype=torch.float32)

        def close(self):
            calls.append("close")

    def fake_zeros(*args, **kwargs):
        calls.append(("zeros", args, kwargs))
        cpu_kwargs = dict(kwargs)
        cpu_kwargs.pop("device", None)
        return original_zeros(*args, **cpu_kwargs)

    api = object()
    monkeypatch.setattr(rtx_vsr_stream, "load_vsr_api", lambda: api)
    monkeypatch.setattr(rtx_vsr_stream, "VsrFrameProcessor", FakeProcessor)
    monkeypatch.setattr(rtx_vsr_stream.torch, "zeros", fake_zeros)
    empty_cache_calls = _mock_probe_cuda(monkeypatch)

    with caplog.at_level(logging.INFO, logger=rtx_vsr_stream.LOGGER.name):
        assert probe_vsr_capability(quality, 0) is True

    assert next(call for call in calls if call[0] == "create") == (
        "create", api, quality, 0, 1280, 720
    )
    zeros_call = next(call for call in calls if call[0] == "zeros")
    assert zeros_call[1] == ((3, 360, 640),)
    assert zeros_call[2] == {"device": torch.device("cuda", 0), "dtype": torch.float32}
    processed_frame = next(call[1] for call in calls if call[0] == "process")
    assert processed_frame.shape == (3, 360, 640)
    assert calls[-1] == "close"
    assert empty_cache_calls == []
    assert "RTX VSR 前置检查成功" in caplog.text


def test_probe_vsr_capability_cleans_up_when_processing_raises(monkeypatch):
    calls = []

    class FakeProcessor:
        def __init__(self, *args):
            calls.append("create")

        def process(self, frame):
            calls.append("process")
            raise RuntimeError("kernel refused")

        def close(self):
            calls.append("close")

    monkeypatch.setattr(rtx_vsr_stream, "load_vsr_api", lambda: object())
    monkeypatch.setattr(rtx_vsr_stream, "VsrFrameProcessor", FakeProcessor)
    monkeypatch.setattr(rtx_vsr_stream.torch, "zeros", lambda *args, **kwargs: torch.ones((3, 360, 640)))
    empty_cache_calls = _mock_probe_cuda(monkeypatch)

    with pytest.raises(RuntimeError, match="kernel refused"):
        probe_vsr_capability("HIGH", 0)

    assert calls == ["create", "process", "close"]
    assert empty_cache_calls == []


def test_probe_vsr_capability_rejects_an_unexpected_output_shape(monkeypatch):
    original_zeros = torch.zeros

    class FakeProcessor:
        def __init__(self, *args):
            pass

        def process(self, frame):
            return original_zeros((3, 720, 1280), dtype=torch.float32)

        def close(self):
            pass

    monkeypatch.setattr(rtx_vsr_stream, "load_vsr_api", lambda: object())
    monkeypatch.setattr(rtx_vsr_stream, "VsrFrameProcessor", FakeProcessor)
    monkeypatch.setattr(rtx_vsr_stream.torch, "zeros", lambda *args, **kwargs: original_zeros((3, 360, 640)))
    empty_cache_calls = _mock_probe_cuda(monkeypatch)

    with pytest.raises(RuntimeError, match="输出尺寸异常"):
        probe_vsr_capability("HIGH", 0)

    assert empty_cache_calls == []


def test_probe_vsr_capability_preserves_process_error_when_close_fails(monkeypatch, caplog):
    class FakeProcessor:
        def __init__(self, *args):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self.close()
            return False

        def process(self, frame):
            raise RuntimeError("PRIMARY_PROCESS_ERROR")

        def close(self):
            raise RuntimeError("CLEANUP_ERROR")

    monkeypatch.setattr(rtx_vsr_stream, "load_vsr_api", lambda: object())
    monkeypatch.setattr(rtx_vsr_stream, "VsrFrameProcessor", FakeProcessor)
    _mock_probe_input(monkeypatch)
    _mock_probe_cuda(monkeypatch)

    with caplog.at_level(logging.WARNING, logger=rtx_vsr_stream.LOGGER.name):
        with pytest.raises(RuntimeError) as error:
            probe_vsr_capability("HIGH", 0)

    assert "PRIMARY_PROCESS_ERROR" in str(error.value)
    assert str(error.value.__cause__) == "PRIMARY_PROCESS_ERROR"
    assert "CLEANUP_ERROR" in caplog.text


def test_probe_vsr_capability_preserves_shape_error_when_close_fails(monkeypatch, caplog):
    original_zeros = torch.zeros

    class FakeProcessor:
        def __init__(self, *args):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self.close()
            return False

        def process(self, frame):
            return original_zeros((3, 720, 1280), dtype=torch.float32)

        def close(self):
            raise RuntimeError("CLEANUP_ERROR")

    monkeypatch.setattr(rtx_vsr_stream, "load_vsr_api", lambda: object())
    monkeypatch.setattr(rtx_vsr_stream, "VsrFrameProcessor", FakeProcessor)
    _mock_probe_input(monkeypatch)
    _mock_probe_cuda(monkeypatch)

    with caplog.at_level(logging.WARNING, logger=rtx_vsr_stream.LOGGER.name):
        with pytest.raises(RuntimeError, match="探测输出尺寸异常") as error:
            probe_vsr_capability("HIGH", 0)

    assert "探测输出尺寸异常" in str(error.value.__cause__)
    assert "CLEANUP_ERROR" in caplog.text


def test_probe_vsr_capability_fails_when_close_fails_after_success(monkeypatch, caplog):
    original_zeros = torch.zeros

    class FakeProcessor:
        def __init__(self, *args):
            pass

        def process(self, frame):
            return original_zeros((720, 1280, 3), dtype=torch.float32)

        def close(self):
            raise RuntimeError("CLEANUP_ERROR")

    monkeypatch.setattr(rtx_vsr_stream, "load_vsr_api", lambda: object())
    monkeypatch.setattr(rtx_vsr_stream, "VsrFrameProcessor", FakeProcessor)
    _mock_probe_input(monkeypatch)
    empty_cache_calls = _mock_probe_cuda(monkeypatch)

    with caplog.at_level(logging.INFO, logger=rtx_vsr_stream.LOGGER.name):
        with pytest.raises(RuntimeError) as error:
            probe_vsr_capability("HIGH", 0)

    message = str(error.value)
    assert "RTX VSR 前置检查失败，尚未开始 H3 视频生成" in message
    assert "质量：HIGH；GPU：0" in message
    assert "CLEANUP_ERROR" in message
    assert "原生尺寸直出" in message
    assert str(error.value.__cause__) == "CLEANUP_ERROR"
    assert empty_cache_calls == []
    assert "RTX VSR 前置检查成功" not in caplog.text


@pytest.mark.parametrize(
    ("device_id", "expected_cause"),
    [(None, TypeError), ("not-a-device", ValueError)],
)
def test_probe_vsr_capability_wraps_invalid_device_id_conversion(monkeypatch, device_id, expected_cause):
    monkeypatch.setattr(
        rtx_vsr_stream.torch,
        "device",
        lambda *args, **kwargs: pytest.fail("invalid device IDs must fail before CUDA device creation"),
    )

    with pytest.raises(RuntimeError, match="RTX VSR 前置检查失败") as error:
        probe_vsr_capability("HIGH", device_id)

    assert isinstance(error.value.__cause__, expected_cause)


def test_probe_vsr_capability_reports_unsupported_sdk_before_generation(monkeypatch):
    original_zeros = torch.zeros

    class FakeEffect:
        def load(self):
            raise RuntimeError("NvVFX_Load failed: The requested feature or capability was not found (code -14)")

    monkeypatch.setattr(
        "nodes.rtx_vsr_stream.load_vsr_api",
        lambda: (lambda *args, **kwargs: FakeEffect(), SimpleNamespace(HIGH="high", ULTRA="ultra")),
    )
    monkeypatch.setattr(rtx_vsr_stream.torch, "zeros", lambda *args, **kwargs: original_zeros((3, 360, 640)))
    monkeypatch.setattr(rtx_vsr_stream.torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="code -14"):
        probe_vsr_capability("ULTRA", 0)


def test_probe_vsr_deblur_chain_uses_real_360p_deblur_then_upscale_contract(monkeypatch, caplog):
    calls = []
    original_zeros = torch.zeros
    api = object()

    class FakeProcessor:
        def __init__(self, *args):
            calls.append(("create", args))

        def process(self, frame):
            calls.append(("process", frame))
            return original_zeros((720, 1280, 3), dtype=torch.float32)

        def close(self):
            calls.append("close")

    def fake_zeros(*args, **kwargs):
        calls.append(("zeros", args, kwargs))
        cpu_kwargs = dict(kwargs)
        cpu_kwargs.pop("device", None)
        return original_zeros(*args, **cpu_kwargs)

    load_calls = []
    monkeypatch.setattr(rtx_vsr_stream, "load_vsr_api", lambda: load_calls.append(True) or api)
    monkeypatch.setattr(rtx_vsr_stream, "DeblurVsrFrameProcessor", FakeProcessor)
    monkeypatch.setattr(rtx_vsr_stream.torch, "zeros", fake_zeros)
    empty_cache_calls = _mock_probe_cuda(monkeypatch)

    with caplog.at_level(logging.INFO, logger=rtx_vsr_stream.LOGGER.name):
        assert probe_vsr_deblur_chain("HIGHBITRATE_ULTRA", 2) is True

    assert load_calls == [True]
    assert calls[0] == (
        "zeros",
        ((3, 360, 640),),
        {"device": torch.device("cuda", 2), "dtype": torch.float32},
    )
    assert calls[1] == (
        "create",
        (api, "HIGHBITRATE_ULTRA", 2, 640, 360, 1280, 720),
    )
    assert calls[2][0] == "process"
    assert calls[2][1].shape == (3, 360, 640)
    assert calls[-1] == "close"
    assert empty_cache_calls == []
    assert (
        "RTX 轻度去模糊 + VSR 前置检查成功：deblur=DEBLUR_LOW "
        "quality=HIGHBITRATE_ULTRA gpu=2 input=640x360 output=1280x720"
    ) in caplog.text


def test_probe_vsr_deblur_chain_rejects_unexpected_cpu_hwc_output(monkeypatch):
    original_zeros = torch.zeros

    class FakeProcessor:
        def __init__(self, *args):
            pass

        def process(self, frame):
            return original_zeros((3, 720, 1280), dtype=torch.float32)

        def close(self):
            pass

    monkeypatch.setattr(rtx_vsr_stream, "load_vsr_api", lambda: object())
    monkeypatch.setattr(rtx_vsr_stream, "DeblurVsrFrameProcessor", FakeProcessor)
    _mock_probe_input(monkeypatch)
    empty_cache_calls = _mock_probe_cuda(monkeypatch)

    with pytest.raises(RuntimeError, match=r"RTX 轻度去模糊 \+ VSR 前置检查失败") as error:
        probe_vsr_deblur_chain("ULTRA", 0)

    assert "CPU HWC (720, 1280, 3)" in str(error.value.__cause__)
    assert empty_cache_calls == []


def test_probe_vsr_deblur_chain_wraps_process_error_and_closes(monkeypatch):
    calls = []

    class FakeProcessor:
        def __init__(self, *args):
            calls.append("create")

        def process(self, frame):
            calls.append("process")
            raise RuntimeError("CHAIN_PROCESS_ERROR")

        def close(self):
            calls.append("close")

    monkeypatch.setattr(rtx_vsr_stream, "load_vsr_api", lambda: object())
    monkeypatch.setattr(rtx_vsr_stream, "DeblurVsrFrameProcessor", FakeProcessor)
    _mock_probe_input(monkeypatch)
    _mock_probe_cuda(monkeypatch)

    with pytest.raises(RuntimeError, match="CHAIN_PROCESS_ERROR") as error:
        probe_vsr_deblur_chain("ULTRA", 0)

    assert calls == ["create", "process", "close"]
    assert str(error.value.__cause__) == "CHAIN_PROCESS_ERROR"


def test_probe_vsr_deblur_chain_close_failure_after_success_is_primary(monkeypatch, caplog):
    original_zeros = torch.zeros

    class FakeProcessor:
        def __init__(self, *args):
            pass

        def process(self, frame):
            return original_zeros((720, 1280, 3), dtype=torch.float32)

        def close(self):
            raise RuntimeError("CHAIN_CLEANUP_ERROR")

    monkeypatch.setattr(rtx_vsr_stream, "load_vsr_api", lambda: object())
    monkeypatch.setattr(rtx_vsr_stream, "DeblurVsrFrameProcessor", FakeProcessor)
    _mock_probe_input(monkeypatch)
    _mock_probe_cuda(monkeypatch)

    with caplog.at_level(logging.INFO, logger=rtx_vsr_stream.LOGGER.name):
        with pytest.raises(RuntimeError, match="CHAIN_CLEANUP_ERROR") as error:
            probe_vsr_deblur_chain("ULTRA", 0)

    assert str(error.value.__cause__) == "CHAIN_CLEANUP_ERROR"
    assert "RTX 轻度去模糊 + VSR 前置检查成功" not in caplog.text


def test_probe_vsr_deblur_chain_preserves_primary_error_when_cleanup_also_fails(monkeypatch, caplog):
    class FakeProcessor:
        def __init__(self, *args):
            pass

        def process(self, frame):
            raise RuntimeError("CHAIN_PRIMARY_ERROR")

        def close(self):
            raise RuntimeError("CHAIN_CLEANUP_ERROR")

    monkeypatch.setattr(rtx_vsr_stream, "load_vsr_api", lambda: object())
    monkeypatch.setattr(rtx_vsr_stream, "DeblurVsrFrameProcessor", FakeProcessor)
    _mock_probe_input(monkeypatch)
    _mock_probe_cuda(monkeypatch)

    with caplog.at_level(logging.WARNING, logger=rtx_vsr_stream.LOGGER.name):
        with pytest.raises(RuntimeError, match="CHAIN_PRIMARY_ERROR") as error:
            probe_vsr_deblur_chain("ULTRA", 0)

    assert str(error.value.__cause__) == "CHAIN_PRIMARY_ERROR"
    assert "CHAIN_CLEANUP_ERROR" in caplog.text


@pytest.mark.parametrize(("device_id", "expected_cause"), [(None, TypeError), ("bad", ValueError)])
def test_probe_vsr_deblur_chain_wraps_invalid_device_id(monkeypatch, device_id, expected_cause):
    monkeypatch.setattr(
        rtx_vsr_stream.torch,
        "device",
        lambda *args, **kwargs: pytest.fail("invalid device IDs must fail before CUDA device creation"),
    )

    with pytest.raises(RuntimeError, match=r"RTX 轻度去模糊 \+ VSR 前置检查失败") as error:
        probe_vsr_deblur_chain("ULTRA", device_id)

    assert isinstance(error.value.__cause__, expected_cause)


def test_probe_vsr_deblur_chain_reports_unsupported_deblur_low_before_generation(monkeypatch):
    monkeypatch.setattr(
        rtx_vsr_stream,
        "load_vsr_api",
        lambda: (lambda *args, **kwargs: object(), SimpleNamespace(ULTRA="ultra")),
    )
    _mock_probe_input(monkeypatch)
    _mock_probe_cuda(monkeypatch)

    with pytest.raises(RuntimeError, match="DEBLUR_LOW") as error:
        probe_vsr_deblur_chain("ULTRA", 0)

    assert "RTX 轻度去模糊 + VSR 前置检查失败" in str(error.value)
    assert "当前 nvvfx SDK 不支持 RTX VSR 质量 DEBLUR_LOW" in str(error.value.__cause__)


def test_probe_vsr_deblur_chain_rejects_unavailable_cuda_before_generation(monkeypatch):
    monkeypatch.setattr(rtx_vsr_stream.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        rtx_vsr_stream,
        "load_vsr_api",
        lambda: pytest.fail("CUDA availability must be checked before loading the SDK"),
    )

    with pytest.raises(RuntimeError, match="CUDA 不可用") as error:
        probe_vsr_deblur_chain("ULTRA", 0)

    assert "RTX 轻度去模糊 + VSR 前置检查失败" in str(error.value)


@pytest.mark.parametrize(
    "name, expected",
    [
        ("HIGH", "high"),
        ("ULTRA", "ultra"),
        ("HIGHBITRATE_ULTRA", "highbitrate_ultra"),
        ("DEBLUR_LOW", "deblur_low"),
    ],
)
def test_resolve_vsr_quality_maps_only_supported_levels(name, expected):
    quality_level = SimpleNamespace(
        HIGH="high",
        ULTRA="ultra",
        HIGHBITRATE_ULTRA="highbitrate_ultra",
        DEBLUR_LOW="deblur_low",
        MEDIUM="medium",
    )

    assert resolve_vsr_quality(quality_level, name) == expected


def test_resolve_vsr_quality_reports_complete_capabilities_when_deblur_is_missing_from_sdk():
    quality_level = SimpleNamespace(
        HIGH="high",
        ULTRA="ultra",
        HIGHBITRATE_ULTRA="highbitrate_ultra",
    )

    with pytest.raises(ValueError) as error:
        resolve_vsr_quality(quality_level, "DEBLUR_LOW")

    assert str(error.value) == (
        "当前 nvvfx SDK 不支持 RTX VSR 质量 DEBLUR_LOW"
        "（需要 HIGH/ULTRA/HIGHBITRATE_ULTRA/DEBLUR_LOW）"
    )


@pytest.mark.parametrize("quality", ["MEDIUM", "high", "", None])
def test_resolve_vsr_quality_rejects_other_values(quality):
    quality_level = SimpleNamespace(HIGH="high", ULTRA="ultra", MEDIUM="medium")

    with pytest.raises(ValueError, match="HIGH.*ULTRA.*HIGHBITRATE_ULTRA.*DEBLUR_LOW"):
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


def test_non_default_device_context_covers_create_run_and_cleanup(monkeypatch):
    active_devices = []
    operations = []
    sdk_output = torch.ones((3, 6, 8), dtype=torch.float32)

    class FakeDeviceContext:
        def __init__(self, device):
            self.device = str(device)

        def __enter__(self):
            active_devices.append(self.device)

        def __exit__(self, exc_type, exc_value, traceback):
            assert active_devices.pop() == self.device

    def record(operation):
        operations.append((operation, active_devices[-1] if active_devices else None))

    class FakeStream:
        def synchronize(self):
            record("stream_sync")

    class FakeEffect:
        def load(self):
            record("load")

        def run(self, frame):
            record("run")
            return SimpleNamespace(image=sdk_output)

        def close(self):
            record("close")

    def video_super_res(*args, **kwargs):
        record("create")
        return FakeEffect()

    monkeypatch.setattr(torch.cuda, "device", FakeDeviceContext)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: FakeStream())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: record("device_sync"))
    original_to = torch.Tensor.to

    def fake_to(tensor, *args, **kwargs):
        device = kwargs.get("device", args[0] if args else None)
        if device is not None and torch.device(device).type == "cuda":
            return tensor.to(dtype=kwargs.get("dtype"))
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", fake_to)

    processor = VsrFrameProcessor(
        (video_super_res, SimpleNamespace(HIGH="high", ULTRA="ultra")),
        "HIGH",
        2,
        8,
        6,
    )
    processor.process(torch.zeros(3, 2, 4))
    processor.close()

    assert operations == [
        ("create", "cuda:2"),
        ("load", "cuda:2"),
        ("stream_sync", "cuda:2"),
        ("run", "cuda:2"),
        ("device_sync", "cuda:2"),
        ("device_sync", "cuda:2"),
        ("close", "cuda:2"),
    ]
    assert active_devices == []


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


def test_sdk_context_manager_uses_entered_effect_and_exits_once():
    calls = []

    class EnteredEffect:
        def load(self):
            calls.append("load")

    entered_effect = EnteredEffect()

    class FakeEffectContext:
        def __enter__(self):
            calls.append("enter")
            return entered_effect

        def __exit__(self, exc_type, exc_value, traceback):
            calls.append(("exit", exc_type, exc_value, traceback))

    processor = VsrFrameProcessor(
        (lambda *args, **kwargs: FakeEffectContext(), SimpleNamespace(HIGH="high", ULTRA="ultra")),
        "HIGH",
        0,
        8,
        6,
    )

    assert entered_effect.output_width == 8
    assert entered_effect.output_height == 6
    processor.close()
    processor.close()

    assert calls == ["enter", "load", ("exit", None, None, None)]


def test_sdk_context_manager_exits_when_load_fails():
    calls = []

    class EnteredEffect:
        def load(self):
            calls.append("load")
            raise RuntimeError("load failed")

    class FakeEffectContext:
        def __enter__(self):
            calls.append("enter")
            return EnteredEffect()

        def __exit__(self, exc_type, exc_value, traceback):
            calls.append("exit")

    with pytest.raises(RuntimeError, match="load failed"):
        VsrFrameProcessor(
            (lambda *args, **kwargs: FakeEffectContext(), SimpleNamespace(HIGH="high", ULTRA="ultra")),
            "HIGH",
            0,
            8,
            6,
        )

    assert calls == ["enter", "load", "exit"]


def test_sdk_context_enter_failure_uses_raw_effect_cleanup_without_exit():
    calls = []

    class RawEffect:
        def __enter__(self):
            calls.append("enter")
            raise RuntimeError("enter failed")

        def __exit__(self, exc_type, exc_value, traceback):
            calls.append("exit")

        def destroy(self):
            calls.append("destroy")

    with pytest.raises(RuntimeError, match="enter failed"):
        VsrFrameProcessor(
            (lambda *args, **kwargs: RawEffect(), SimpleNamespace(HIGH="high", ULTRA="ultra")),
            "HIGH",
            0,
            8,
            6,
        )

    assert calls == ["enter", "destroy"]


def test_process_cuda_returns_cloned_contiguous_chw_float32(monkeypatch):
    sdk_output = torch.full((3, 6, 8), 0.75, dtype=torch.float32).transpose(1, 2)

    class FakeStream:
        def synchronize(self):
            pass

    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: FakeStream())
    original_to = torch.Tensor.to

    def fake_to(tensor, *args, **kwargs):
        device = kwargs.get("device", args[0] if args else None)
        if device is not None and torch.device(device).type == "cuda":
            return tensor.to(dtype=kwargs.get("dtype"))
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", fake_to)

    class FakeEffect:
        def run(self, frame):
            return SimpleNamespace(image=sdk_output)

        def close(self):
            pass

    processor = VsrFrameProcessor(
        (lambda *args, **kwargs: FakeEffect(), SimpleNamespace(HIGH="high", ULTRA="ultra")),
        "HIGH",
        0,
        6,
        8,
    )

    result = processor.process_cuda(torch.zeros(3, 2, 4, dtype=torch.float16))
    sdk_output.zero_()

    assert result.shape == (3, 8, 6)
    assert result.dtype == torch.float32
    assert result.is_contiguous()
    assert torch.all(result == 0.75)
    processor.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for this tensor contract")
def test_process_cuda_returns_independent_cuda_chw_float32_tensor(monkeypatch):
    sdk_output = torch.full((3, 6, 8), 0.75, device="cuda", dtype=torch.float32)

    monkeypatch.setattr(torch.cuda, "device", REAL_CUDA_DEVICE_CONTEXT)

    class FakeEffect:
        def run(self, frame):
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

    result = processor.process_cuda(torch.zeros((3, 2, 4), dtype=torch.float32))
    sdk_output.zero_()

    assert result.device.type == "cuda"
    assert result.dtype == torch.float32
    assert result.shape == (3, 6, 8)
    assert result.data_ptr() != sdk_output.data_ptr()
    assert torch.all(result == 0.75)
    processor.close()


def test_deblur_processor_chains_deblur_cuda_to_upscale_with_requested_shapes(monkeypatch):
    calls = []
    intermediate = torch.ones((3, 4, 6), dtype=torch.float32)
    output = torch.ones((12, 18, 3), dtype=torch.float32)

    class FakeProcessor:
        def __init__(self, api, quality, device_id, width, height):
            self.quality = quality
            self.dimensions = (width, height)
            calls.append(("create", quality, device_id, width, height))

        def process_cuda(self, frame):
            calls.append(("deblur.process_cuda", frame))
            return intermediate

        def process(self, frame):
            calls.append(("upscale.process", frame))
            return output

        def close(self):
            calls.append(("close", self.quality))

    monkeypatch.setattr(rtx_vsr_stream, "VsrFrameProcessor", FakeProcessor)

    processor = DeblurVsrFrameProcessor(object(), "ULTRA", 2, 6, 4, 18, 12)
    result = processor.process(torch.zeros((3, 4, 6), dtype=torch.float32))

    assert calls[:2] == [
        ("create", "DEBLUR_LOW", 2, 6, 4),
        ("create", "ULTRA", 2, 18, 12),
    ]
    assert calls[2][0] == "deblur.process_cuda"
    assert calls[2][1].shape == (3, 4, 6)
    assert calls[3] == ("upscale.process", intermediate)
    assert result is output


def test_deblur_processor_logs_cleanup_failure_when_upscale_creation_fails(monkeypatch, caplog):
    calls = []
    original_error = RuntimeError("upscale setup failed")
    cleanup_error = RuntimeError("deblur cleanup failed")

    class FakeProcessor:
        def __init__(self, api, quality, device_id, width, height):
            calls.append(("create", quality))
            if quality == "ULTRA":
                raise original_error
            self.quality = quality

        def close(self):
            calls.append(("close", self.quality))
            raise cleanup_error

    monkeypatch.setattr(rtx_vsr_stream, "VsrFrameProcessor", FakeProcessor)

    with caplog.at_level(logging.WARNING, logger=rtx_vsr_stream.LOGGER.name):
        with pytest.raises(RuntimeError) as error:
            DeblurVsrFrameProcessor(object(), "ULTRA", 0, 6, 4, 18, 12)

    assert error.value is original_error
    assert calls == [("create", "DEBLUR_LOW"), ("create", "ULTRA"), ("close", "DEBLUR_LOW")]
    assert "deblur cleanup failed" in caplog.text


def test_deblur_processor_can_be_closed_after_processing_failure(monkeypatch):
    calls = []

    class FakeProcessor:
        def __init__(self, api, quality, device_id, width, height):
            self.quality = quality

        def process_cuda(self, frame):
            raise RuntimeError("deblur failed")

        def process(self, frame):
            pytest.fail("upscale must not run after deblur failure")

        def close(self):
            calls.append(self.quality)

    monkeypatch.setattr(rtx_vsr_stream, "VsrFrameProcessor", FakeProcessor)
    processor = DeblurVsrFrameProcessor(object(), "ULTRA", 0, 6, 4, 18, 12)

    with pytest.raises(RuntimeError, match="deblur failed"):
        processor.process(torch.zeros((3, 4, 6)))
    processor.close()

    assert calls == ["ULTRA", "DEBLUR_LOW"]


def test_deblur_processor_close_attempts_both_logs_secondary_and_prioritizes_upscale_error(
    monkeypatch, caplog
):
    calls = []
    upscale_error = RuntimeError("upscale close failed")

    class FakeProcessor:
        def __init__(self, api, quality, device_id, width, height):
            self.quality = quality

        def close(self):
            calls.append(self.quality)
            if self.quality == "ULTRA":
                raise upscale_error
            raise RuntimeError("deblur close failed")

    monkeypatch.setattr(rtx_vsr_stream, "VsrFrameProcessor", FakeProcessor)
    processor = DeblurVsrFrameProcessor(object(), "ULTRA", 0, 6, 4, 18, 12)

    with caplog.at_level(logging.WARNING, logger=rtx_vsr_stream.LOGGER.name):
        with pytest.raises(RuntimeError) as error:
            processor.close()

    assert error.value is upscale_error
    assert calls == ["ULTRA", "DEBLUR_LOW"]
    assert "deblur close failed" in caplog.text
    processor.close()
    assert calls == ["ULTRA", "DEBLUR_LOW"]


def test_deblur_processor_context_manager_closes_once(monkeypatch):
    calls = []

    class FakeProcessor:
        def __init__(self, api, quality, device_id, width, height):
            self.quality = quality

        def close(self):
            calls.append(self.quality)

    monkeypatch.setattr(rtx_vsr_stream, "VsrFrameProcessor", FakeProcessor)

    with DeblurVsrFrameProcessor(object(), "ULTRA", 0, 6, 4, 18, 12) as processor:
        assert processor is not None

    processor.close()
    assert calls == ["ULTRA", "DEBLUR_LOW"]


def test_deblur_context_manager_preserves_business_error_when_both_closes_fail(monkeypatch, caplog):
    calls = []
    business_error = RuntimeError("encode failed")

    class FakeProcessor:
        def __init__(self, api, quality, device_id, width, height):
            self.quality = quality

        def close(self):
            calls.append(self.quality)
            raise RuntimeError(f"{self.quality} close failed")

    monkeypatch.setattr(rtx_vsr_stream, "VsrFrameProcessor", FakeProcessor)

    with caplog.at_level(logging.WARNING, logger=rtx_vsr_stream.LOGGER.name):
        with pytest.raises(RuntimeError) as error:
            with DeblurVsrFrameProcessor(object(), "ULTRA", 0, 6, 4, 18, 12):
                raise business_error

    assert error.value is business_error
    assert calls == ["ULTRA", "DEBLUR_LOW"]
    assert "ULTRA close failed" in caplog.text
    assert "DEBLUR_LOW close failed" in caplog.text
