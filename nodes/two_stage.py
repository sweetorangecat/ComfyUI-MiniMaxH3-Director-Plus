"""Matched MiniMax H3 trained-latent two-stage AV sampler."""

from __future__ import annotations

import copy
import logging

import torch
import torch.nn.functional as F

from .two_stage_assets import (
    resolve_split_upscale_callables,
    run_trained_latent_upscaler,
)


LOGGER = logging.getLogger("MiniMaxH3.DirectorPlus.TwoStage")


def split_sigmas_at_step(sigmas, split_step):
    """Match ComfyUI SplitSigmas while preserving the shared boundary sigma."""
    if not isinstance(sigmas, torch.Tensor) or sigmas.ndim != 1:
        raise ValueError("二阶段采样需要一维 SIGMAS 轨迹")
    split_step = int(split_step)
    if split_step < 1:
        raise ValueError("二阶段切分步必须大于 0")
    if sigmas.numel() < split_step + 2:
        raise ValueError("二阶段切分步超过 sigma 轨迹")
    return sigmas[: split_step + 1], sigmas[split_step:]


def _node_output(result, index=0):
    result = getattr(result, "result", result)
    if isinstance(result, (tuple, list)):
        return result[index]
    if index == 0:
        return result
    raise IndexError(f"节点结果没有第 {index} 个输出")


def _latent_shape(latent):
    samples = latent.get("samples") if isinstance(latent, dict) else None
    if getattr(samples, "is_nested", False):
        return [tuple(member.shape) for member in samples.unbind()]
    if isinstance(samples, torch.Tensor):
        return tuple(samples.shape)
    return None


def clone_guider_with_model(guider, second_model):
    """Keep conditioning intact while switching to the matched tail model."""
    result = copy.copy(guider)
    result.model_patcher = second_model
    result.model_options = second_model.model_options
    return result


def _resize_condition_latent(latent, target_h, target_w):
    """Resize one FL keyframe latent to the second-stage spatial grid."""
    container = None
    if isinstance(latent, dict):
        container = latent
        samples = latent.get("samples")
    else:
        samples = latent
    if not isinstance(samples, torch.Tensor) or samples.ndim not in (4, 5):
        raise ValueError("H3 首尾帧条件格式无效：latent 必须是 4D/5D 张量")
    target_h, target_w = int(target_h), int(target_w)
    if target_h < 1 or target_w < 1:
        raise ValueError("H3 二采目标 latent 尺寸无效")
    if samples.shape[-2:] == (target_h, target_w):
        return latent

    source_dtype = samples.dtype
    source_device = samples.device
    if samples.ndim == 5:
        resized = F.interpolate(
            samples.to(dtype=torch.float32),
            size=(samples.shape[-3], target_h, target_w),
            mode="trilinear",
            align_corners=False,
        )
    else:
        resized = F.interpolate(
            samples.to(dtype=torch.float32),
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )
    resized = resized.to(device=source_device, dtype=source_dtype)
    return {**container, "samples": resized} if container is not None else resized


def prepare_second_stage_guider(guider, second_model, target_video_shape):
    """Clone guider and align FL keyframe conditions with the enlarged target grid.

    REF2VA references intentionally retain their own per-reference grids. Only
    FL2VA keyframes are target-grid conditions and must be resized before the
    second sampler rebuilds the MiniMax packed layout.
    """
    if not isinstance(target_video_shape, (tuple, list)) or len(target_video_shape) < 5:
        raise ValueError("H3 二采目标 video latent 尺寸无效")
    target_h, target_w = int(target_video_shape[-2]), int(target_video_shape[-1])
    result = clone_guider_with_model(guider, second_model)
    original = getattr(guider, "original_conds", None)
    if not isinstance(original, dict):
        raise ValueError("H3 二采 guider 缺少原始 conditioning，无法校验视觉条件尺寸")

    copied = {}
    for cond_name, conditions in original.items():
        if not isinstance(conditions, list):
            copied[cond_name] = conditions
            continue
        copied_conditions = []
        for condition in conditions:
            if not isinstance(condition, dict) or "minimax_keyframes" not in condition:
                copied_conditions.append(condition)
                continue
            updated = condition.copy()
            keyframes = condition.get("minimax_keyframes") or []
            resized_keyframes = []
            for keyframe in keyframes:
                if not isinstance(keyframe, dict) or "latent" not in keyframe:
                    raise ValueError("H3 首尾帧条件格式无效：缺少 latent")
                updated_keyframe = keyframe.copy()
                updated_keyframe["latent"] = _resize_condition_latent(
                    keyframe["latent"], target_h, target_w
                )
                resized_keyframes.append(updated_keyframe)
            updated["minimax_keyframes"] = resized_keyframes
            copied_conditions.append(updated)
        copied[cond_name] = copied_conditions
    result.original_conds = copied
    LOGGER.info(
        "[H3 two-stage] FL 首尾帧条件已对齐到第二阶段 latent 网格 %sx%s；REF2VA 参考图保持原网格",
        target_w,
        target_h,
    )
    return result


def validate_second_stage_condition_shapes(guider, target_video_shape):
    """Fail before sampling if any FL keyframe cannot match the target grid."""
    if not isinstance(target_video_shape, (tuple, list)) or len(target_video_shape) < 5:
        raise ValueError("H3 二采目标 video latent 尺寸无效")
    target_h, target_w = int(target_video_shape[-2]), int(target_video_shape[-1])
    for cond_name, conditions in (getattr(guider, "original_conds", {}) or {}).items():
        for condition in conditions or []:
            for keyframe in (condition.get("minimax_keyframes") or []) if isinstance(condition, dict) else []:
                latent = keyframe.get("latent") if isinstance(keyframe, dict) else None
                samples = latent.get("samples") if isinstance(latent, dict) else latent
                if isinstance(samples, torch.Tensor) and samples.ndim in (4, 5):
                    if samples.shape[-2:] != (target_h, target_w):
                        raise ValueError(
                            "H3 二采首尾帧条件尺寸未对齐："
                            f"{tuple(samples.shape[-2:])} != {(target_h, target_w)}"
                        )


def _release_between_stages():
    """Offload H3 before the learned GPU upscaler, then reload for pass two."""
    try:
        import comfy.model_management as model_management

        model_management.unload_all_models()
        model_management.soft_empty_cache()
        LOGGER.info("[H3 two-stage] 第一阶段 H3 已卸载，训练型放大器将在 GPU 执行")
    except (AttributeError, ImportError, RuntimeError) as exc:
        LOGGER.warning("[H3 two-stage] 阶段间显存释放未完整执行: %s", exc)


def _positive_conditioning(guider):
    """Return raw positive CONDITIONING for nodes that build their own guider."""
    original = getattr(guider, "original_conds", None)
    positive = original.get("positive") if isinstance(original, dict) else None
    if not isinstance(positive, list):
        raise ValueError("H3 分块二采缺少 positive conditioning，无法构建分块引导")
    return positive


def run_tiled_second_stage(
    split_callables,
    *,
    model,
    guider,
    noise,
    sampler,
    sigmas,
    latent,
    guide,
):
    """Run the MMH3 tiled second stage through the installed SplitUpscale node.

    MMH3SplitUpscale re-denoises the enlarged AV latent tile by tile, keeping
    the peak VRAM of the second pass proportional to one tile instead of the
    full 2K frame.  It rebuilds its own guider from raw conditioning and
    re-anchors/crops ``minimax_keyframes`` internally, so FL keyframes do not
    need the full-frame grid alignment done by ``prepare_second_stage_guider``.
    """
    upscale_fn, temporal_fn, spatial_fn = split_callables
    positive = _positive_conditioning(guider)

    temporal_param = None
    if temporal_fn is not None:
        temporal_param = _node_output(
            temporal_fn(
                chunk_frames=int(guide.get("split_chunk_frames", 141)),
                temporal_overlap_frames=int(guide.get("split_temporal_overlap_frames", 39)),
                anchor_strength=float(guide.get("split_anchor_strength", 0.999)),
                motion_anchor_frames=str(guide.get("split_motion_anchor_frames", "39")),
                identity_anchor_frames=int(guide.get("split_identity_anchor_frames", 24)),
            ),
            0,
        )
    spatial_param = None
    if spatial_fn is not None:
        spatial_param = _node_output(
            spatial_fn(
                tile_width=int(guide.get("split_tile_width", 512)),
                tile_height=int(guide.get("split_tile_height", 512)),
                overlap_ratio=float(guide.get("split_overlap_ratio", 0.25)),
                fade_ratio=float(guide.get("split_fade_ratio", 0.5)),
                min_tile_size=int(guide.get("split_min_tile_size", 256)),
                seam_denoise=float(guide.get("split_seam_denoise", 0.65)),
            ),
            0,
        )

    result = upscale_fn(
        latent=latent,
        conditioning=positive,
        negative=None,
        model=model,
        noise=noise,
        sampler=sampler,
        sigmas=sigmas,
        cfg=1.0,
        temporal_split_param=temporal_param,
        spatial_split_param=spatial_param,
        seam_polish=str(guide.get("split_seam_polish", "auto")),
        color_match=bool(guide.get("split_color_match", True)),
    )
    return _node_output(result, 0)


class MiniMaxH3TwoStageSampler:
    """Run FL 4+4 or Reference 4+5 while preserving pass-one audio."""

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
                "second_model": ("MODEL",),
            },
        }

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("输出Latent", "去噪Latent")
    FUNCTION = "execute"
    CATEGORY = "MiniMax H3 导演台 Plus"

    @staticmethod
    def _enabled(guide):
        return bool((guide or {}).get("two_stage_enabled"))

    def execute(
        self,
        noise,
        guider,
        sampler,
        sigmas,
        latent_image,
        guide=None,
        second_model=None,
    ):
        from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced
        from .performance import memory_policy

        guide = guide or {}
        if not self._enabled(guide):
            with memory_policy(guide):
                result = SamplerCustomAdvanced.execute(noise, guider, sampler, sigmas, latent_image)
            guide["two_stage_status"] = "旁路"
            return _node_output(result, 0), _node_output(result, 1)

        if second_model is None:
            raise ValueError("训练型二采缺少第二阶段模型，请更新并重新载入 U11 工作流")
        from comfy_extras.nodes_custom_sampler import Noise_EmptyNoise
        split_step = int(guide.get("two_stage_split_step", 4))
        scale = float(guide.get("two_stage_scale", 1.5))
        if scale <= 1.0:
            raise ValueError("训练型 H3 latent 放大倍率必须大于 1.0")
        try:
            first_sigmas, second_sigmas = split_sigmas_at_step(sigmas, split_step)
        except ValueError as exc:
            raise ValueError(f"训练型 H3 二采 sigma 轨迹无效：{exc}") from exc

        route = guide.get("resolved_two_stage_route", "trained_latent_fl")
        guide["two_stage_status"] = "训练型 3D latent 二采执行中"
        LOGGER.info(
            "[H3 two-stage] route=%s split=%s sigmas=%s+%s first_lora=%s second_lora=%s input=%s",
            route,
            split_step,
            len(first_sigmas),
            len(second_sigmas),
            guide.get("first_lora_name"),
            guide.get("second_lora_name"),
            _latent_shape(latent_image),
        )

        from comfy_extras.nodes_lt import LTXVConcatAVLatent, LTXVSeparateAVLatent

        with memory_policy(guide):
            first = SamplerCustomAdvanced.execute(
                noise,
                guider,
                sampler,
                first_sigmas,
                latent_image,
            )
            first_denoised = _node_output(first, 1)
            separated = LTXVSeparateAVLatent.execute(first_denoised)
            video_latent = _node_output(separated, 0)
            audio_latent = _node_output(separated, 1)
            source_video_shape = _latent_shape(video_latent)
            source_audio_shape = _latent_shape(audio_latent)
            _release_between_stages()
            upscaled_video = run_trained_latent_upscaler(video_latent, scale)
            LOGGER.info(
                "[H3 two-stage] trained_upscaler video=%s->%s audio_preserved=%s scale=%.2f",
                source_video_shape,
                _latent_shape(upscaled_video),
                source_audio_shape,
                scale,
            )
            merged = _node_output(LTXVConcatAVLatent.execute(upscaled_video, audio_latent))
            merged_video_shape = _latent_shape(upscaled_video)
            # Continue from the clean first-pass x0 latent.  Injecting a new
            # random field here repaints the enlarged latent and was the
            # source of the soft, grainy 1080p output reported in production.
            second_noise = Noise_EmptyNoise()
            split_callables = None
            if guide.get("two_stage_tiled", True):
                try:
                    split_callables = resolve_split_upscale_callables()
                except RuntimeError as exc:
                    LOGGER.warning(
                        "[H3 two-stage] MMH3SplitUpscale 不可用，回退整帧二采: %s", exc
                    )
            if split_callables is not None:
                LOGGER.info(
                    "[H3 two-stage] 第二阶段使用 MMH3SplitUpscale 时空分块采样，显存峰值按瓦片计"
                )
                final_denoised = run_tiled_second_stage(
                    split_callables,
                    model=second_model,
                    guider=guider,
                    noise=second_noise,
                    sampler=sampler,
                    sigmas=second_sigmas,
                    latent=merged,
                    guide=guide,
                )
                guide["two_stage_second_stage_path"] = "mmh3_split_upscale"
                guide["two_stage_status"] = "训练型 3D latent 二采完成（分块）"
            else:
                second_guider = prepare_second_stage_guider(
                    guider, second_model, merged_video_shape
                )
                validate_second_stage_condition_shapes(second_guider, merged_video_shape)
                second = SamplerCustomAdvanced.execute(
                    second_noise,
                    second_guider,
                    sampler,
                    second_sigmas,
                    merged,
                )
                final_denoised = _node_output(second, 1)
                guide["two_stage_second_stage_path"] = "full_frame"
                guide["two_stage_status"] = "训练型 3D latent 二采完成"

        guide["two_stage_first_sigma_count"] = len(first_sigmas)
        guide["two_stage_second_sigma_count"] = len(second_sigmas)
        LOGGER.info("[H3 two-stage] completed output=%s", _latent_shape(final_denoised))
        return final_denoised, final_denoised
