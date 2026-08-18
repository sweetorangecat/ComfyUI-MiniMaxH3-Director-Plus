"""Self-contained MiniMax H3 latent two-stage refinement sampler."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def split_refinement_sigmas(sigmas, refinement_steps):
    """Split a descending sigma schedule while preserving one boundary value."""
    if not isinstance(sigmas, torch.Tensor) or sigmas.ndim != 1:
        raise ValueError("二阶段采样需要一维 SIGMAS 轨迹")
    refinement_steps = int(refinement_steps)
    if refinement_steps < 1:
        raise ValueError("二阶段采样步数必须大于 0")
    if sigmas.numel() < refinement_steps + 2:
        raise ValueError("二阶段采样步数超过第一阶段 sigma 轨迹")
    split_index = sigmas.numel() - refinement_steps
    return sigmas[:split_index], sigmas[split_index - 1:]


def upscale_video_latent(video_latent, scale):
    """Scale only H3 video latent spatial dimensions; keep time and metadata."""
    scale = float(scale)
    if scale <= 1.0:
        return video_latent
    samples = video_latent.get("samples")
    if not isinstance(samples, torch.Tensor) or samples.ndim != 5:
        raise ValueError("H3 视频 latent 必须是 [B,C,T,H,W] 五维张量")
    batch, channels, frames, height, width = samples.shape
    target_height = max(1, round(height * scale))
    target_width = max(1, round(width * scale))
    # Interpolate each temporal slice as an image so time remains untouched.
    image_batch = samples.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
    upscaled = F.interpolate(image_batch, size=(target_height, target_width), mode="nearest-exact")
    upscaled = upscaled.reshape(batch, frames, channels, target_height, target_width).permute(0, 2, 1, 3, 4)
    result = dict(video_latent)
    result["samples"] = upscaled
    if "noise_mask" in result and isinstance(result["noise_mask"], torch.Tensor):
        mask = result["noise_mask"]
        if mask.ndim == 5 and tuple(mask.shape[-2:]) == (height, width):
            mask_batch = mask.permute(0, 2, 1, 3, 4).reshape(batch * frames, mask.shape[1], height, width)
            mask = F.interpolate(mask_batch, size=(target_height, target_width), mode="nearest")
            result["noise_mask"] = mask.reshape(batch, frames, -1, target_height, target_width).permute(0, 2, 1, 3, 4)
    return result


def _node_output(result, index=0):
    result = getattr(result, "result", result)
    return result[index] if isinstance(result, (tuple, list)) else result[index]


class MiniMaxH3TwoStageSampler:
    """Run U15-style continuous sigma refinement without extra user wiring."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noise": ("NOISE",),
                "guider": ("GUIDER",),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "latent_image": ("LATENT",),
            },
            "optional": {"guide": ("MINIMAX_H3_DIRECTOR_PLUS_GUIDE",)},
        }

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("输出Latent", "去噪Latent")
    FUNCTION = "execute"
    CATEGORY = "MiniMax H3 导演台 Plus"

    @staticmethod
    def _enabled(guide):
        return bool((guide or {}).get("two_stage_enabled"))

    def execute(self, noise, guider, sampler, sigmas, latent_image, guide=None):
        from comfy_extras.nodes_custom_sampler import Noise_RandomNoise, SamplerCustomAdvanced
        from comfy_extras.nodes_lt import LTXVConcatAVLatent, LTXVSeparateAVLatent
        from .performance import memory_policy

        guide = guide or {}
        enabled = self._enabled(guide)
        if not enabled:
            with memory_policy(guide):
                result = SamplerCustomAdvanced.execute(noise, guider, sampler, sigmas, latent_image)
            guide["two_stage_status"] = "旁路"
            return _node_output(result, 0), _node_output(result, 1)

        refinement_steps = int(guide.get("two_stage_steps", 6))
        scale = float(guide.get("two_stage_scale", 1.5))
        try:
            first_sigmas, second_sigmas = split_refinement_sigmas(sigmas, refinement_steps)
        except ValueError as exc:
            raise ValueError(f"U15 二阶段 sigma 轨迹无效：{exc}") from exc
        if scale <= 1.0:
            raise ValueError("U15 二阶段视频 latent 放大倍率必须大于 1.0")
        guide["two_stage_status"] = "U15 二阶段 latent 细化"
        guide["two_stage_enabled"] = True

        with memory_policy(guide):
            first = SamplerCustomAdvanced.execute(noise, guider, sampler, first_sigmas, latent_image)
            # U15 feeds the sampled latent (slot 0) into the latent upscale.
            # Slot 1 is an x0 preview and is not on the sampler's sigma path.
            sampled = _node_output(first, 0)
            video_latent, audio_latent = LTXVSeparateAVLatent.execute(sampled)
            # The audio stream is intentionally never resized; H3's AV concat
            # node aligns it back to the refined video stream.
            video_latent = upscale_video_latent(video_latent, scale)
            merged = _node_output(LTXVConcatAVLatent.execute(video_latent, audio_latent), 0)
            seed = int(getattr(noise, "seed", 0))
            second_noise = Noise_RandomNoise(seed)
            second = SamplerCustomAdvanced.execute(second_noise, guider, sampler, second_sigmas, merged)
        return _node_output(second, 0), _node_output(second, 1)
