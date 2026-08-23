"""Self-contained MiniMax H3 latent two-stage refinement sampler."""

from __future__ import annotations

import copy
import gc
import logging

import torch
import torch.nn.functional as F


LOGGER = logging.getLogger("MiniMaxH3.DirectorPlus.TwoStage")


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
    # MiniMax H3 spatial patch size is 2x2 in latent space. An odd latent
    # height/width can fail later while the DiT builds its token grid.
    target_height = max(2, ((round(height * scale) + 1) // 2) * 2)
    target_width = max(2, ((round(width * scale) + 1) // 2) * 2)
    # Interpolate each temporal slice as an image so time remains untouched.
    image_batch = samples.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
    # Interpolate in fp32 with bilinear blending. Nearest-neighbor interpolation
    # duplicates latent cells; the low-sigma redraw then turns those cell
    # boundaries into coarse grain and RTX VSR makes them more visible.
    original_dtype = image_batch.dtype
    upscaled = F.interpolate(
        image_batch.float(),
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
    ).to(dtype=original_dtype)
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


def _nested_members(samples):
    if getattr(samples, "is_nested", False):
        members = list(samples.unbind())
        if not members:
            raise ValueError("H3 AV latent 为空")
        return members, True
    if isinstance(samples, torch.Tensor):
        return [samples], False
    raise TypeError(f"H3 latent 必须是 Tensor 或 NestedTensor，实际为 {type(samples)!r}")


def _wrap_members(members, nested):
    if nested:
        from comfy.nested_tensor import NestedTensor

        return NestedTensor(members)
    if len(members) != 1:
        raise ValueError("普通 latent 只能包含一个张量")
    return members[0]


def _upscale_visual_tensor(latent, scale):
    if not isinstance(latent, torch.Tensor) or latent.ndim not in (4, 5):
        raise ValueError("H3 参考图/关键帧 latent 必须是四维或五维张量")
    is_image = latent.ndim == 4
    source = latent.unsqueeze(2) if is_image else latent
    upscaled = upscale_video_latent({"samples": source}, scale)["samples"]
    return upscaled.squeeze(2) if is_image else upscaled


def _upscale_reference_block(block, scale):
    result = dict(block)
    if result.get("kind") == "audio" or result.get("latent") is None:
        return result
    latent = _upscale_visual_tensor(result["latent"], scale)
    result["latent"] = latent
    result["latent_h"] = int(latent.shape[-2])
    result["latent_w"] = int(latent.shape[-1])
    if latent.ndim == 5 and "latent_t" in result:
        result["latent_t"] = int(latent.shape[2])
    return result


def _upscale_keyframe(keyframe, scale):
    result = dict(keyframe)
    if result.get("latent") is not None:
        result["latent"] = _upscale_visual_tensor(result["latent"], scale)
    return result


def upscale_h3_guider(guider, scale):
    """Clone a Guider and keep H3 reference/keyframe metadata on the new grid."""
    original = getattr(guider, "original_conds", None)
    if not isinstance(original, dict):
        return guider
    result = copy.copy(guider)
    result.original_conds = {}
    for name, entries in original.items():
        scaled_entries = []
        for entry in entries:
            if not isinstance(entry, dict):
                scaled_entries.append(entry)
                continue
            scaled = dict(entry)
            if entry.get("minimax_refs") is not None:
                scaled["minimax_refs"] = [
                    _upscale_reference_block(block, scale) for block in entry["minimax_refs"]
                ]
            if entry.get("minimax_keyframes") is not None:
                scaled["minimax_keyframes"] = [
                    _upscale_keyframe(keyframe, scale) for keyframe in entry["minimax_keyframes"]
                ]
            scaled_entries.append(scaled)
        result.original_conds[name] = scaled_entries
    return result


def _nonzero(samples):
    members, _ = _nested_members(samples)
    return any(torch.count_nonzero(member) > 0 for member in members)


def prepare_h3_two_stage_latent(latent, model, noise, sigmas, scale):
    """Upscale and CONST re-noise an H3 AV latent for a DisableNoise pass.

    This is intentionally latent-only. It follows the H3-specific two-pass
    requirements documented by these GPL-3.0 reference implementations:
    https://github.com/rockerBOO/h3-latent-upscaler and
    https://github.com/Tr1dae/ComfyUI-MiniMaxH3_LatentUpscaler
    resize only the video stream, pre-cancel CONST's second noise scaling,
    and lock the already-generated audio with a zero denoise mask.
    """
    if not isinstance(sigmas, torch.Tensor) or sigmas.numel() == 0:
        return latent
    samples = latent.get("samples")
    members, nested = _nested_members(samples)
    if not nested or len(members) < 2:
        raise ValueError("H3 二采需要包含视频和音频的 AV NestedTensor latent")

    source_video_shape = tuple(members[0].shape)
    video = upscale_video_latent({"samples": members[0]}, scale)["samples"]
    LOGGER.info(
        "H3 latent 二采准备：视频 %s -> %s；全程不调用 VideoVAE",
        source_video_shape,
        tuple(video.shape),
    )
    upscaled_members = [video, *members[1:]]
    upscaled = dict(latent)
    upscaled["samples"] = _wrap_members(upscaled_members, True)
    upscaled.pop("noise_mask", None)

    generated_noise = noise.generate_noise(upscaled)
    noise_members, noise_nested = _nested_members(generated_noise)
    if not noise_nested or len(noise_members) != len(upscaled_members):
        raise ValueError("H3 二采随机噪声与 AV latent 结构不一致")

    model_sampling = model.get_model_object("model_sampling")
    process_latent_in = model.get_model_object("process_latent_in")
    process_latent_out = model.get_model_object("process_latent_out")
    latent_for_mix = process_latent_in(upscaled["samples"]) if _nonzero(upscaled["samples"]) else upscaled["samples"]
    latent_members, _ = _nested_members(latent_for_mix)
    sigma = sigmas[0]
    mixed_members = []
    for index, (source, random_field) in enumerate(zip(latent_members, noise_members)):
        # Preserve pass-one audio; only the enlarged video stream receives new noise.
        if index > 0:
            random_field = torch.zeros_like(random_field)
        mixed = model_sampling.noise_scaling(sigma, random_field, source)
        if hasattr(model_sampling, "inverse_noise_scaling"):
            mixed = model_sampling.inverse_noise_scaling(sigma, mixed)
        mixed_members.append(mixed)
    mixed = process_latent_out(_wrap_members(mixed_members, True))
    mixed_members, _ = _nested_members(mixed)

    result = dict(upscaled)
    result["samples"] = _wrap_members(
        [torch.nan_to_num(member, nan=0.0, posinf=0.0, neginf=0.0).detach().cpu() for member in mixed_members],
        True,
    )
    masks = [
        torch.ones((video.shape[0], 1, *video.shape[2:]), dtype=torch.float32),
        *[
            torch.zeros((member.shape[0], 1, *member.shape[2:]), dtype=torch.float32)
            for member in upscaled_members[1:]
        ],
    ]
    result["noise_mask"] = _wrap_members(masks, True)

    # Do not unload the quantized H3 model between passes. Parking the small
    # latent on CPU and clearing allocator fragments is sufficient and avoids
    # the process-killing full VAE round trip used by the previous route.
    gc.collect()
    try:
        import comfy.model_management as model_management

        model_management.soft_empty_cache()
    except Exception:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return result


class MiniMaxH3TwoStageSampler:
    """Run the full-clip H3 latent two-pass route without extra user wiring."""

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
        from .performance import memory_policy

        guide = guide or {}
        enabled = self._enabled(guide)
        if not enabled:
            with memory_policy(guide):
                result = SamplerCustomAdvanced.execute(noise, guider, sampler, sigmas, latent_image)
            guide["two_stage_status"] = "旁路"
            return _node_output(result, 0), _node_output(result, 1)

        refinement_steps = int(guide.get("two_stage_steps", 2))
        scale = float(guide.get("two_stage_scale", 1.5))
        try:
            first_sigmas, second_sigmas = split_refinement_sigmas(sigmas, refinement_steps)
        except ValueError as exc:
            raise ValueError(f"H3 latent 二采 sigma 轨迹无效：{exc}") from exc
        if scale <= 1.0:
            raise ValueError("H3 latent 二采视频放大倍率必须大于 1.0")
        guide["two_stage_status"] = "H3 专用 latent 二采"
        guide["two_stage_enabled"] = True

        with memory_policy(guide):
            first = SamplerCustomAdvanced.execute(noise, guider, sampler, first_sigmas, latent_image)
            denoised = _node_output(first, 1)
            second_random_noise = Noise_RandomNoise(int(getattr(noise, "seed", 0)))
            prepared = prepare_h3_two_stage_latent(
                denoised,
                guider.model_patcher,
                second_random_noise,
                second_sigmas,
                scale,
            )
            second_guider = upscale_h3_guider(guider, scale)
            # prepare_h3_two_stage_latent already injected the correctly
            # inverse-scaled CONST noise, so the continuation must use empty
            # noise. Adding random noise here a second time creates grey/noise.
            second_noise = Noise_EmptyNoise()
            second = SamplerCustomAdvanced.execute(
                second_noise,
                second_guider,
                sampler,
                second_sigmas,
                prepared,
            )
        guide["two_stage_status"] = "H3 专用 latent 二采完成"
        final_denoised = _node_output(second, 1)
        return final_denoised, final_denoised
