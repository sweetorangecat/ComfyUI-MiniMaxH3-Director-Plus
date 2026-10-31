"""Temporal exposure and colour continuity guard for MiniMax H3 outputs."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .execution_report import attach_report, report_from


class MiniMaxH3ColorGuard:
    """Keep generated frames close to the scene exposure without changing motion."""

    GPU_OUTPUT_LIMIT_BYTES = 512 * 1024**2
    COLOR_SAMPLE_FRAMES = 24
    COLOR_SAMPLE_EDGE = 128
    CHROMA_EXCESS_THRESHOLD = 1.08
    CHROMA_SCALE_EXPONENT = 0.65
    MIN_CHROMA_SCALE = 0.85

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
    def _chunk_ranges(frame_count, chunk_size=8):
        chunk_size = max(1, int(chunk_size))
        return [
            (start, min(start + chunk_size, frame_count))
            for start in range(0, int(frame_count), chunk_size)
        ]

    @staticmethod
    def _requires_cpu_output(total_bytes, is_cuda):
        """Avoid a second multi-GiB IMAGE allocation beside the decoded video."""
        return bool(is_cuda and int(total_bytes) > MiniMaxH3ColorGuard.GPU_OUTPUT_LIMIT_BYTES)

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
    def _sample_indices(frame_count, device):
        count = min(int(frame_count), MiniMaxH3ColorGuard.COLOR_SAMPLE_FRAMES)
        if count <= 0:
            return None
        return torch.linspace(
            0,
            max(0, int(frame_count) - 1),
            steps=count,
            device=device,
        ).round().long().unique()

    @staticmethod
    def _chroma_metric(images):
        """Return a small global RGB chroma statistic for a clip or reference.

        The metric is deliberately scene-level rather than a per-frame grade.
        It ignores near-black/near-white pixels and downsamples before doing
        any arithmetic, so a long 2K clip never receives a second full-sized
        working tensor.  The returned value is used only to choose one global
        chroma scale; it cannot alter geometry or frame timing.
        """
        if not torch.is_tensor(images):
            return None
        if images.ndim == 3:
            images = images.unsqueeze(0)
        if images.ndim != 4 or images.shape[-1] < 3 or not images.shape[0]:
            return None
        sample_indices = MiniMaxH3ColorGuard._sample_indices(images.shape[0], images.device)
        if sample_indices is None:
            return None
        sample = images.index_select(0, sample_indices)[..., :3].to(dtype=torch.float32)
        sample = sample.movedim(-1, 1)
        height, width = int(sample.shape[-2]), int(sample.shape[-1])
        edge = max(height, width)
        if edge > MiniMaxH3ColorGuard.COLOR_SAMPLE_EDGE:
            scale = MiniMaxH3ColorGuard.COLOR_SAMPLE_EDGE / float(edge)
            sample = F.interpolate(
                sample,
                size=(max(8, int(round(height * scale))), max(8, int(round(width * scale)))),
                mode="area",
            )
        sample = sample.movedim(1, -1)
        luma = MiniMaxH3ColorGuard._luma(sample)
        chroma = (sample - luma.unsqueeze(-1)).abs().mean(-1)
        valid = (luma > 0.05) & (luma < 0.95)
        if int(valid.sum()) >= 32:
            chroma = chroma[valid]
        if not chroma.numel():
            return None
        return float(chroma.mean().detach().cpu())

    @classmethod
    def _reference_chroma_metric(cls, images, guide):
        """Find a conservative scene-color target from ordinary references."""
        if not isinstance(guide, dict):
            return None
        references = guide.get("ref_images")
        if not isinstance(references, dict):
            return None
        metrics = []
        for reference in references.values():
            metric = cls._chroma_metric(reference)
            if metric is not None and metric > 1e-5:
                metrics.append(metric)
        if not metrics:
            return None
        # Median avoids letting one white-background identity sheet force the
        # entire generated clip toward gray.
        return float(torch.tensor(metrics, dtype=torch.float32).median())

    @classmethod
    def _resolve_chroma_scale(cls, images, guide, strength, anchor=None):
        if float(strength) <= 0:
            return None, None
        target = cls._chroma_metric(anchor) if anchor is not None else cls._reference_chroma_metric(images, guide)
        if target is None or target <= 1e-5:
            return None, None
        current = cls._chroma_metric(images)
        if current is None or current <= target * cls.CHROMA_EXCESS_THRESHOLD:
            return None, (current, target)
        ratio = max(0.0, min(1.0, target / current))
        # Use a fractional exponent and a hard floor: this is a gentle
        # correction for the plastic/saturated look, never a desaturation pass.
        base_scale = max(cls.MIN_CHROMA_SCALE, ratio ** cls.CHROMA_SCALE_EXPONENT)
        scale = 1.0 - (1.0 - base_scale) * min(1.0, float(strength))
        return float(scale), (current, target)

    @staticmethod
    def _apply_chroma_scale(frames, scale):
        if scale is None or abs(float(scale) - 1.0) < 1e-6:
            return frames
        luma = MiniMaxH3ColorGuard._luma(frames)
        return luma.unsqueeze(-1) + (frames - luma.unsqueeze(-1)) * float(scale)

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
        for start, end in MiniMaxH3ColorGuard._chunk_ranges(corrected.shape[0]):
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

        fast4 = isinstance(guide, dict) and guide.get("performance_preset") in {"fast_4step", "极速4步"}
        output_bytes = images.numel() * images.element_size()
        output_device = torch.device(
            "cpu" if self._requires_cpu_output(output_bytes, images.is_cuda) else images.device
        )
        anchor = self._anchor(images, guide)
        chroma_scale, chroma_stats = self._resolve_chroma_scale(
            images, guide, strength, anchor=anchor
        )
        if anchor is None and not fast4 and chroma_scale is None:
            # There is no reliable absolute exposure reference for text-only
            # or reference-only generation. Return before any dtype conversion
            # so long videos keep their original memory footprint.
            return images, "未发现硬首帧，保持原始曝光；未将生成第1帧当作曝光基准"

        target_luma = None
        if anchor is not None:
            target_luma = self._luma(
                anchor.to(device=output_device, dtype=torch.float32)
            ).mean(dim=(1, 2), keepdim=True).clamp_min(1e-4)

        # Keep the original IMAGE tensor resident and process only a few RGB
        # frames as float32. This prevents a 15-second high-resolution decode
        # from allocating another 8-12 GiB just for color protection.
        corrected = torch.empty(images.shape, dtype=images.dtype, device=output_device)
        for start, end in self._chunk_ranges(images.shape[0]):
            frames = images[start:end, ..., :3].to(device=output_device, dtype=torch.float32)
            if target_luma is not None:
                frame_luma = self._luma(frames).mean(dim=(1, 2), keepdim=True).clamp_min(1e-4)
                ratio = (target_luma / frame_luma).pow(float(strength)).clamp(0.45, 1.38)
                frames = frames * ratio.unsqueeze(-1)
            if chroma_scale is not None:
                frames = self._apply_chroma_scale(frames, chroma_scale)
            if fast4:
                frames = self._fast4_finish(frames, strength)
            corrected[start:end, ..., :3] = frames.clamp(0.0, 1.0).to(dtype=images.dtype)
            if images.shape[-1] > 3:
                corrected[start:end, ..., 3:] = images[start:end, ..., 3:].to(output_device)

        report = report_from(images)
        if report is not None:
            attach_report(corrected, report)
        chroma_note = ""
        if chroma_scale is not None and chroma_stats is not None:
            current, target = chroma_stats
            chroma_note = (
                f"；已按参考色度轻度保护（{current:.4f}->{target:.4f}，系数 {chroma_scale:.3f}）"
            )
        if fast4:
            return corrected, "极速4步已启用轻度提亮与细节恢复；未将生成第1帧当作曝光基准" + chroma_note
        if anchor is None:
            return corrected, "未发现硬首帧，保持原始曝光；未将生成第1帧当作曝光基准" + chroma_note
        return corrected, "已锁定场景曝光与色彩连续性；REF2VA 未将第一张参考图视为首帧" + chroma_note
