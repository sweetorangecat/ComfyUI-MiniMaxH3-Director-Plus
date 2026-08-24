"""Strict, frame-at-a-time NVIDIA RTX VSR integration."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch


LOGGER = logging.getLogger(__name__)
PROBE_INPUT_WIDTH = 640
PROBE_INPUT_HEIGHT = 360
PROBE_OUTPUT_WIDTH = 1280
PROBE_OUTPUT_HEIGHT = 720


def _dasiwa_requirements_path() -> Path:
    """Resolve the sibling DaSiWa dependency file for any ComfyUI install."""
    custom_nodes = Path(__file__).resolve().parents[2]
    return custom_nodes / "ComfyUI-DaSiWa-Nodes" / "requirements.txt"


def _dependency_error(reason: str) -> RuntimeError:
    requirements = _dasiwa_requirements_path()
    python_executable = Path(sys.executable)
    install_command = f'"{python_executable}" -m pip install -r "{requirements}"'
    verify_command = f'"{python_executable}" -c "import nvvfx; print(nvvfx.VideoSuperRes)"'
    return RuntimeError(
        "RTX VSR 依赖不可用："
        f"{reason}\n"
        f"当前 ComfyUI Python：{sys.executable}\n"
        f"DaSiWa 依赖文件（绝对路径）：{requirements}\n"
        f"请安装 nvidia-vfx：{install_command}\n"
        "部分 nvidia-vfx 版本还需要先安装 NVIDIA Broadcast SDK。\n"
        f"安装完成后重启 ComfyUI，并运行以下命令验证：{verify_command}"
    )


def load_vsr_api() -> tuple[object, object]:
    """Load only nvvfx's VideoSuperRes API, with actionable setup guidance."""
    try:
        import nvvfx
    except (ImportError, ModuleNotFoundError, OSError) as exc:
        raise _dependency_error("未找到 nvvfx 模块。") from exc

    video_super_res = getattr(nvvfx, "VideoSuperRes", None)
    if video_super_res is None:
        try:
            from nvvfx import VideoSuperRes as video_super_res
        except (ImportError, ModuleNotFoundError, OSError) as exc:
            raise _dependency_error("nvvfx 已安装，但缺少 VideoSuperRes。") from exc

    effects = getattr(nvvfx, "effects", None)
    quality_level = getattr(effects, "QualityLevel", None)
    if quality_level is None:
        quality_level = getattr(video_super_res, "QualityLevel", None)
    if quality_level is None:
        raise _dependency_error("VideoSuperRes 缺少 QualityLevel。")

    return video_super_res, quality_level


def resolve_vsr_quality(quality_level, quality: str):
    """Resolve the two quality levels supported by the public U11 contract."""
    if quality not in ("HIGH", "ULTRA"):
        raise ValueError("RTX VSR 质量仅支持 HIGH 或 ULTRA")
    if not hasattr(quality_level, quality):
        raise ValueError(f"当前 nvvfx SDK 不支持 RTX VSR 质量 {quality}（需要 HIGH/ULTRA）")
    return getattr(quality_level, quality)


def _create_effect(video_super_res, quality, device_id: int):
    attempts = (
        ((), {"quality": quality, "device": device_id}),
        ((quality,), {"device": device_id}),
        ((), {"quality": quality}),
        ((quality,), {}),
    )
    last_type_error = None
    for args, kwargs in attempts:
        try:
            return video_super_res(*args, **kwargs)
        except TypeError as exc:
            last_type_error = exc
    raise last_type_error


def _close_effect(effect) -> None:
    for method_name in ("close", "destroy", "unload"):
        method = getattr(effect, method_name, None)
        if callable(method):
            method()
            return


class VsrFrameProcessor:
    """Own one VideoSuperRes effect and process independent CHW RGB frames."""

    def __init__(
        self,
        api,
        quality: str,
        device_id: int,
        output_width: int,
        output_height: int,
    ):
        if int(output_width) <= 0 or int(output_height) <= 0:
            raise ValueError("RTX VSR 输出宽高必须为正整数")

        video_super_res, quality_level = api
        resolved_quality = resolve_vsr_quality(quality_level, quality)
        self.device_id = int(device_id)
        self.output_width = int(output_width)
        self.output_height = int(output_height)
        self.cuda_device = torch.device("cuda", self.device_id)
        self._effect = None
        self._effect_context = None
        self._effect_entered = False

        try:
            with torch.cuda.device(self.cuda_device):
                effect = _create_effect(video_super_res, resolved_quality, self.device_id)
                self._effect = effect
                if hasattr(effect, "__enter__") and hasattr(effect, "__exit__"):
                    entered_effect = effect.__enter__()
                    self._effect_context = effect
                    self._effect_entered = True
                    self._effect = entered_effect
                effect = self._effect
                effect.output_width = self.output_width
                effect.output_height = self.output_height
                load = getattr(effect, "load", None)
                if callable(load):
                    load()
        except Exception as exc:
            try:
                self.close()
            except Exception:
                pass
            raise RuntimeError(f"创建 NVIDIA RTX VSR 效果失败（{quality}）：{exc}") from exc

    def process(self, chw_frame: torch.Tensor) -> torch.Tensor:
        """Run one CHW RGB frame and return a cloned CPU HWC float32 frame."""
        if self._effect is None:
            raise RuntimeError("RTX VSR 处理器已关闭")
        if not torch.is_tensor(chw_frame) or chw_frame.ndim != 3 or chw_frame.shape[0] != 3:
            raise ValueError("RTX VSR 输入必须是 [3, H, W] 的 CHW RGB 张量")

        with torch.cuda.device(self.cuda_device):
            cuda_frame = (
                chw_frame.detach()
                .to(
                    device=self.cuda_device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
                .clamp(0.0, 1.0)
                .contiguous()
            )

            torch.cuda.current_stream(self.cuda_device).synchronize()
            result = self._effect.run(cuda_frame)
            torch.cuda.synchronize(self.cuda_device)

            image = getattr(result, "image", None)
            if image is None:
                raise RuntimeError("NVIDIA RTX VSR 未返回 DLPack 图像")
            output_chw = torch.from_dlpack(image).clone().contiguous()
            expected_shape = (3, self.output_height, self.output_width)
            if tuple(output_chw.shape) != expected_shape:
                raise RuntimeError(
                    "NVIDIA RTX VSR 输出尺寸异常："
                    f"期望 {expected_shape}，实际 {tuple(output_chw.shape)}"
                )

            return (
                output_chw.movedim(0, -1)
                .clamp(0.0, 1.0)
                .to(device="cpu", dtype=torch.float32, non_blocking=False)
                .contiguous()
            )

    def close(self) -> None:
        effect = self._effect
        effect_context = self._effect_context
        effect_entered = self._effect_entered
        self._effect = None
        self._effect_context = None
        self._effect_entered = False
        if effect is None and effect_context is None:
            return
        with torch.cuda.device(self.cuda_device):
            torch.cuda.synchronize(self.cuda_device)
            if effect_context is not None and effect_entered:
                effect_context.__exit__(None, None, None)
            elif effect is not None:
                _close_effect(effect)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


def probe_vsr_capability(quality: str = "HIGH", device_id: int = 0) -> bool:
    """Validate the real RTX VSR effect before H3 denoising starts.

    Importing ``nvvfx`` only proves that the Python package is present.  The
    NVIDIA SDK can still reject ``NvVFX_Load`` because the installed driver,
    Broadcast SDK, GPU, or requested quality is unsupported.  Reuse the same
    constructor/load path as frame processing so this failure is reported by
    the director node before any expensive video generation begins.
    """
    normalized_device_id = None
    cuda_device = None
    processor = None
    input_frame = None
    output_frame = None
    try:
        normalized_device_id = int(device_id)
        cuda_device = torch.device("cuda", normalized_device_id)
        api = load_vsr_api()
        input_frame = torch.zeros(
            (3, PROBE_INPUT_HEIGHT, PROBE_INPUT_WIDTH),
            device=cuda_device,
            dtype=torch.float32,
        )
        processor = VsrFrameProcessor(
            api,
            quality,
            normalized_device_id,
            PROBE_OUTPUT_WIDTH,
            PROBE_OUTPUT_HEIGHT,
        )
        output_frame = processor.process(input_frame)

        expected_shape = (PROBE_OUTPUT_HEIGHT, PROBE_OUTPUT_WIDTH, 3)
        if output_frame.device.type != "cpu" or tuple(output_frame.shape) != expected_shape:
            raise RuntimeError(
                "NVIDIA RTX VSR 探测输出尺寸异常："
                f"期望 CPU HWC {expected_shape}，实际 "
                f"{output_frame.device.type} {tuple(output_frame.shape)}"
            )

        LOGGER.info(
            "RTX VSR 前置检查成功：quality=%s gpu=%s input=%sx%s output=%sx%s",
            quality,
            normalized_device_id,
            PROBE_INPUT_WIDTH,
            PROBE_INPUT_HEIGHT,
            PROBE_OUTPUT_WIDTH,
            PROBE_OUTPUT_HEIGHT,
        )
        return True
    except Exception as exc:
        raise RuntimeError(
            "RTX VSR 前置检查失败，尚未开始 H3 视频生成。"
            f"\n质量：{quality}；GPU：{device_id}。"
            f"\n详细错误：{exc}"
            "\n请确认 NVIDIA 驱动、NVIDIA Broadcast SDK 与 nvidia-vfx 版本匹配。"
            "\n如果当前设备不支持该能力，请将‘最终输出’切换为‘原生尺寸直出’，不会影响 H3 生成。"
        ) from exc
    finally:
        if processor is not None:
            try:
                processor.close()
            except Exception as cleanup_exc:
                LOGGER.warning("RTX VSR 前置检查效果清理失败：%s", cleanup_exc)
        input_frame = None
        output_frame = None
