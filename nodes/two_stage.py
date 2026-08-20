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


def _image_space_refine(video_latent, video_vae, scale):
    """Decode, resize, and re-encode one H3 video stream for U09-style redraw."""
    samples = video_latent["samples"]
    decoded = video_vae.decode(samples)
    if not isinstance(decoded, torch.Tensor):
        raise ValueError("H3 VAE 解码未返回 Tensor")
    if decoded.ndim == 5:
        # The VAE wrapper returns [B,T,H,W,C]; direct H3 VAE implementations
        # may return [B,C,T,H,W]. Accept both forms.
        if decoded.shape[1] in (1, 3, 4) and decoded.shape[-1] not in (1, 3, 4):
            decoded = decoded.movedim(1, -1)
        decoded = decoded.reshape(-1, decoded.shape[-3], decoded.shape[-2], decoded.shape[-1])
    if decoded.ndim != 4 or decoded.shape[-1] < 3:
        raise ValueError(f"H3 二采重绘需要 [帧,高,宽,通道] 图像，实际为 {tuple(decoded.shape)}")
    height, width = int(decoded.shape[1]), int(decoded.shape[2])
    target_height = max(16, int(round(height * float(scale) / 16.0)) * 16)
    target_width = max(16, int(round(width * float(scale) / 16.0)) * 16)
    if (target_height, target_width) != (height, width):
        pixels = F.interpolate(
            decoded[..., :3].movedim(-1, 1).to(dtype=torch.float32),
            size=(target_height, target_width),
            mode="bicubic",
            align_corners=False,
        ).movedim(1, -1).clamp(0.0, 1.0)
    else:
        pixels = decoded[..., :3].to(dtype=torch.float32).clamp(0.0, 1.0)
    encoded = video_vae.encode(pixels)
    refined = dict(video_latent)
    refined["samples"] = encoded
    refined.pop("noise_mask", None)
    return refined


class MiniMaxH3TwoStageSampler:
    """Run the U09 image-space redraw without extra user wiring."""

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
            "optional": {
                "guide": ("MINIMAX_H3_DIRECTOR_PLUS_GUIDE",),
                "video_vae": ("VAE",),
            },
        }

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("输出Latent", "去噪Latent")
    FUNCTION = "execute"
    CATEGORY = "MiniMax H3 导演台 Plus"

    @staticmethod
    def _enabled(guide):
        return bool((guide or {}).get("two_stage_enabled"))

    def execute(self, noise, guider, sampler, sigmas, latent_image, guide=None, video_vae=None):
        from comfy_extras.nodes_custom_sampler import Noise_EmptyNoise, Noise_RandomNoise, SamplerCustomAdvanced
        from comfy_extras.nodes_lt import LTXVConcatAVLatent, LTXVSeparateAVLatent
        from .performance import memory_policy

        guide = guide or {}
        enabled = self._enabled(guide)
        if not enabled:
            with memory_policy(guide):
                result = SamplerCustomAdvanced.execute(noise, guider, sampler, sigmas, latent_image)
            guide["two_stage_status"] = "旁路"
            return _node_output(result, 0), _node_output(result, 1)

        refinement_steps = int(guide.get("two_stage_steps", 5))
        scale = float(guide.get("two_stage_scale", 1.5))
        try:
            first_sigmas, second_sigmas = split_refinement_sigmas(sigmas, refinement_steps)
        except ValueError as exc:
            raise ValueError(f"U09 二采 sigma 轨迹无效：{exc}") from exc
        if scale <= 1.0:
            raise ValueError("U09 二采视频放大倍率必须大于 1.0")
        guide["two_stage_status"] = "U09 图像空间二采重绘"
        guide["two_stage_enabled"] = True

        with memory_policy(guide):
            first = SamplerCustomAdvanced.execute(noise, guider, sampler, first_sigmas, latent_image)
            sampled = _node_output(first, 0)
            denoised = _node_output(first, 1)
            # Match the H3 AV split: refine the clean x0 video stream,
            # but preserve the sampler-path audio stream for the final AV
            # continuation. Upscaling slot 0's noisy video produces the flat
            # grey output seen after VAE decode.
            _, audio_latent = LTXVSeparateAVLatent.execute(sampled)
            video_latent, _ = LTXVSeparateAVLatent.execute(denoised)
            # The audio stream is intentionally never resized; H3's AV concat
            # node aligns it back to the refined video stream.
            if video_vae is not None:
                # U09's quality path is an image-space redraw, not a second
                # generic upscaler.  Decode the clean first-pass x0, enlarge
                # it on 16px H3 boundaries, and re-encode before low-denoise
                # refinement.  The legacy latent path remains available for
                # legacy workflows that have no VAE socket.
                video_latent = _image_space_refine(video_latent, video_vae, scale)
                second_sigmas = _node_output(
                    __import__("comfy_extras.nodes_custom_sampler", fromlist=["BasicScheduler"])
                    .BasicScheduler.execute(
                        guider.model_patcher,
                        "simple",
                        2,
                        0.2,
                    ),
                    0,
                )
                merged = _node_output(LTXVConcatAVLatent.execute(video_latent, audio_latent), 0)
                second_noise = Noise_RandomNoise(int(getattr(noise, "seed", 0)))
                second = SamplerCustomAdvanced.execute(second_noise, guider, sampler, second_sigmas, merged)
                guide["two_stage_status"] = "U09 图像空间二采重绘"
                final_denoised = _node_output(second, 1)
                return final_denoised, final_denoised
            video_latent = upscale_video_latent(video_latent, scale)
            # Legacy latent fallback retains U15's boundary sampler. It has one sigma only, so it adds no
            # effective user-visible step. It is nevertheless required to
            # convert the upscaled denoised latent back to the sigma boundary
            # expected by the low-sigma continuation. Skipping this conversion
            # produces the grey/noise decode reported on long clips.
            boundary_noise = Noise_RandomNoise(int(getattr(noise, "seed", 0)))
            boundary = SamplerCustomAdvanced.execute(
                boundary_noise,
                guider,
                sampler,
                second_sigmas[:1],
                video_latent,
            )
            video_latent = _node_output(boundary, 0)
            merged = _node_output(LTXVConcatAVLatent.execute(video_latent, audio_latent), 0)
            # The final continuation uses empty noise, matching the legacy boundary path after its
            # boundary conversion rather than injecting a second random field.
            second_noise = Noise_EmptyNoise()
            second = SamplerCustomAdvanced.execute(second_noise, guider, sampler, second_sigmas, merged)
        # The legacy latent fallback decodes the final denoised AV latent (slot 1) for both video and
        # audio. Slot 0 is still the sampler path output and can contain noise,
        # which produces a grey/noisy video when sent directly to VAEDecode.
        final_denoised = _node_output(second, 1)
        return final_denoised, final_denoised
