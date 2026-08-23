"""Matched MiniMax H3 trained-latent two-stage AV sampler."""

from __future__ import annotations

import copy
import logging

import torch

from .two_stage_assets import run_trained_latent_upscaler


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


def _release_between_stages():
    """Offload H3 before the learned GPU upscaler, then reload for pass two."""
    try:
        import comfy.model_management as model_management

        model_management.unload_all_models()
        model_management.soft_empty_cache()
        LOGGER.info("[H3 two-stage] 第一阶段 H3 已卸载，训练型放大器将在 GPU 执行")
    except (AttributeError, ImportError, RuntimeError) as exc:
        LOGGER.warning("[H3 two-stage] 阶段间显存释放未完整执行: %s", exc)


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
        from comfy_extras.nodes_custom_sampler import Noise_RandomNoise
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
            second_guider = clone_guider_with_model(guider, second_model)
            second_noise = Noise_RandomNoise(int(getattr(noise, "seed", guide.get("seed", 0))))
            second = SamplerCustomAdvanced.execute(
                second_noise,
                second_guider,
                sampler,
                second_sigmas,
                merged,
            )

        final_denoised = _node_output(second, 1)
        guide["two_stage_status"] = "训练型 3D latent 二采完成"
        guide["two_stage_first_sigma_count"] = len(first_sigmas)
        guide["two_stage_second_sigma_count"] = len(second_sigmas)
        LOGGER.info("[H3 two-stage] completed output=%s", _latent_shape(final_denoised))
        return final_denoised, final_denoised
