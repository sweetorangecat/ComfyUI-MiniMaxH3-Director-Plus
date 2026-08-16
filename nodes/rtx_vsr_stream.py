"""Strict, frame-at-a-time NVIDIA RTX VSR integration."""

from __future__ import annotations

import sys

import torch


_DASIWA_REQUIREMENTS = (
    "D:/ComfyUI_windows_portable-G313/ComfyUI/custom_nodes/"
    "ComfyUI-DaSiWa-Nodes/requirements.txt"
)
_EMBEDDED_PYTHON = "D:/ComfyUI_windows_portable-G313/python_embeded/python.exe"


def _dependency_error(reason: str) -> RuntimeError:
    install_command = (
        f"{_EMBEDDED_PYTHON} -m pip install -r {_DASIWA_REQUIREMENTS}"
    )
    verify_command = f'"{sys.executable}" -c "import nvvfx; print(nvvfx.VideoSuperRes)"'
    return RuntimeError(
        "RTX VSR 依赖不可用："
        f"{reason}\n"
        f"当前 ComfyUI Python：{sys.executable}\n"
        f"DaSiWa 依赖文件（绝对路径）：{_DASIWA_REQUIREMENTS}\n"
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

        try:
            effect = _create_effect(video_super_res, resolved_quality, self.device_id)
            self._effect = effect
            effect.output_width = self.output_width
            effect.output_height = self.output_height
            load = getattr(effect, "load", None)
            if callable(load):
                load()
        except Exception as exc:
            effect = self._effect
            self._effect = None
            if effect is not None:
                try:
                    _close_effect(effect)
                except Exception:
                    pass
            raise RuntimeError(f"创建 NVIDIA RTX VSR 效果失败（{quality}）：{exc}") from exc

    def process(self, chw_frame: torch.Tensor) -> torch.Tensor:
        """Run one CHW RGB frame and return a cloned CPU HWC float32 frame."""
        if self._effect is None:
            raise RuntimeError("RTX VSR 处理器已关闭")
        if not torch.is_tensor(chw_frame) or chw_frame.ndim != 3 or chw_frame.shape[0] != 3:
            raise ValueError("RTX VSR 输入必须是 [3, H, W] 的 CHW RGB 张量")

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
        self._effect = None
        if effect is not None:
            _close_effect(effect)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False
