"""CPU-backed, frame-chunked video resize for low-VRAM H3 runs."""

from __future__ import annotations

import re

import torch
import torch.nn.functional as F


def _available_upscale_models():
    try:
        import folder_paths

        return list(folder_paths.get_filename_list("upscale_models"))
    except (ImportError, OSError, AttributeError):
        return []


def is_x2_upscale_model_name(model_name):
    """Return whether a model basename explicitly declares an X2 scale."""
    value = str(model_name or "").replace("\\", "/").rsplit("/", 1)[-1]
    return re.search(r"(^|[^0-9])(x2|2x)(?=[^0-9]|$)", value, re.IGNORECASE) is not None


def resolve_upscale_model_name(model_name="auto", scale_factor=2.0, available=None):
    """Resolve an installed general-purpose model without hiding missing files."""
    available = list(_available_upscale_models() if available is None else available)
    requested = str(model_name or "auto").strip() or "auto"
    by_lower = {str(name).lower(): name for name in available}
    if requested.lower() != "auto":
        if requested.lower() not in by_lower:
            raise ValueError(
                f"通用 AI 超分模型不存在：{requested}。请将模型放入 ComfyUI/models/upscale_models。"
            )
        return by_lower[requested.lower()]

    factor = float(scale_factor)
    preferred = (
        ["realesrgan_x2plus.pth", "omnisr_x2_div2k.safetensors"]
        if factor <= 2.0
        else ["omnisr_x4_div2k.safetensors", "realesrgan_x4plus.pth"]
    )
    for name in preferred:
        if name in by_lower:
            return by_lower[name]

    marker = "x2" if factor <= 2.0 else "x4"
    matches = sorted(
        name for name in available
        if marker in str(name).lower() or f"{int(factor)}x" in str(name).lower()
    )
    if matches:
        return matches[0]
    raise ValueError(
        f"未找到适合约 {factor:.1f} 倍放大的通用 AI 超分模型。"
        "请安装 X2/X4 模型到 ComfyUI/models/upscale_models，或切换为 Lanczos。"
    )


class MiniMaxH3VideoUpscale:
    """Resize decoded H3 frames without allocating the target video on CUDA."""

    CHUNK_SIZE = 4

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "guide": ("MINIMAX_H3_DIRECTOR_PLUS_GUIDE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("目标尺寸视频帧", "放大说明")
    FUNCTION = "apply"
    CATEGORY = "MiniMax H3 导演台 Plus"

    @staticmethod
    def _chunk_ranges(frame_count, chunk_size=4):
        chunk_size = max(1, int(chunk_size))
        return [
            (start, min(start + chunk_size, int(frame_count)))
            for start in range(0, int(frame_count), chunk_size)
        ]

    def apply(self, images, guide):
        if not guide.get("upscale_required"):
            return images, "分块放大未启用，保持 H3 原生尺寸"

        target_width = int(guide.get("target_width", 0))
        target_height = int(guide.get("target_height", 0))
        if target_width <= 0 or target_height <= 0:
            raise ValueError("放大目标尺寸无效，请检查导演台分辨率设置")

        # Always place the result on CPU. The decoded low-resolution video may
        # still be resident on CUDA, and a target-sized GPU copy defeats the
        # low-VRAM route before the encoder can consume it.
        output = torch.empty(
            (images.shape[0], target_height, target_width, images.shape[-1]),
            dtype=images.dtype,
            device="cpu",
        )
        for start, end in self._chunk_ranges(images.shape[0], self.CHUNK_SIZE):
            chunk = images[start:end].to(device="cpu", dtype=torch.float32)
            nchw = chunk.movedim(-1, 1)
            resized = F.interpolate(
                nchw,
                size=(target_height, target_width),
                mode="bicubic",
                align_corners=False,
            )
            output[start:end] = resized.movedim(1, -1).clamp(0.0, 1.0).to(dtype=images.dtype)

        return output, f"已用 CPU 分块放大到 {target_width}×{target_height}，每块 {self.CHUNK_SIZE} 帧"
