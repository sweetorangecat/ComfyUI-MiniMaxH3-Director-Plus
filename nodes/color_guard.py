"""Temporal exposure and colour continuity guard for MiniMax H3 outputs."""

from __future__ import annotations

import torch
import torch.nn.functional as F


class MiniMaxH3ColorGuard:
    """Keep generated frames close to the scene exposure without changing motion."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "guide": ("MINIMAX_H3_DIRECTOR_PLUS_GUIDE",),
                "enabled": ("BOOLEAN", {"default": True}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("色彩稳定帧", "色彩保护说明")
    FUNCTION = "apply"
    CATEGORY = "MiniMax H3 导演台 Plus"

    @staticmethod
    def _luma(image):
        return image[..., :3].mul(image.new_tensor([0.2126, 0.7152, 0.0722])).sum(-1)

    @staticmethod
    def _anchor(images, guide):
        # I2VA/FL2VA/L2VA expose a real first frame. REF2VA deliberately does
        # not, because its first reference is not a keyframe. T2VA also has no
        # scene-exposure anchor. Never use the first generated frame as one:
        # fast distilled runs often begin with a deliberately dark transition.
        source = guide.get("first_frame") if isinstance(guide, dict) else None
        if torch.is_tensor(source) and source.ndim == 4 and source.shape[0]:
            source = source[..., :3].to(device=images.device, dtype=images.dtype)
            source = source.movedim(-1, 1)
            source = F.interpolate(source[:1], size=images.shape[1:3], mode="bilinear", align_corners=False)
            return source.movedim(1, -1)
        return None

    @staticmethod
    def _fast4_finish(frames, strength):
        """Apply a conservative lift and detail recovery for four-step output."""
        if strength <= 0:
            return frames

        # Turbo output is commonly under-exposed in the shadows. A sub-unity
        # gamma lifts dark values while retaining highlight headroom.
        gamma = max(0.88, 1.0 - 0.08 * float(strength))
        corrected = frames.clamp(0.0, 1.0).pow(gamma)

        # A small unsharp mask restores edge definition lost by the distilled
        # trajectory. Process a few frames at a time: a 15-second 1.75 MP
        # decode can contain hundreds of frames and should not need several
        # extra full-resolution working tensors simultaneously.
        detail_amount = min(0.14, 0.10 * float(strength))
        chunk_size = 8
        for start in range(0, corrected.shape[0], chunk_size):
            end = min(start + chunk_size, corrected.shape[0])
            nchw = corrected[start:end].movedim(-1, 1)
            padded = F.pad(nchw, (1, 1, 1, 1), mode="reflect")
            blur = F.avg_pool2d(padded, kernel_size=3, stride=1)
            corrected[start:end] = (
                nchw + (nchw - blur) * detail_amount
            ).movedim(1, -1).clamp(0.0, 1.0)
        return corrected

    def apply(self, images, guide, enabled=True, strength=1.0):
        if not enabled or images.ndim != 4 or images.shape[0] < 2:
            return images, "色彩保护未启用或帧数不足"

        frames = images[..., :3].to(dtype=torch.float32)
        anchor = self._anchor(frames, guide)
        if anchor is None:
            # There is no reliable absolute exposure reference for text-only
            # or reference-only generation. Scaling to frame 1 would turn a
            # dark opening transition into a dark full video.
            corrected = frames
        else:
            target_luma = self._luma(anchor).mean(dim=(1, 2), keepdim=True).clamp_min(1e-4)
            frame_luma = self._luma(frames).mean(dim=(1, 2), keepdim=True).clamp_min(1e-4)

            # Correct only the frame-to-anchor exposure drift. The exponent
            # keeps the default conservative and gains are bounded.
            ratio = (target_luma / frame_luma).pow(float(strength)).clamp(0.45, 1.38)
            corrected = frames * ratio.unsqueeze(-1)

        fast4 = isinstance(guide, dict) and guide.get("performance_preset") in {"fast_4step", "极速4步"}
        if fast4:
            corrected = self._fast4_finish(corrected, strength)

        # Keep extra channels (if present) untouched and restore the input dtype.
        if images.shape[-1] > 3:
            corrected = torch.cat((corrected, images[..., 3:].to(corrected)), dim=-1)
        corrected = corrected.clamp(0.0, 1.0).to(dtype=images.dtype)
        if fast4:
            return corrected, "极速4步已启用轻度提亮与细节恢复；未将生成第1帧当作曝光基准"
        if anchor is None:
            return corrected, "未发现硬首帧，保持原始曝光；未将生成第1帧当作曝光基准"
        return corrected, "已锁定场景曝光与色彩连续性；REF2VA 未将第一张参考图视为首帧"
