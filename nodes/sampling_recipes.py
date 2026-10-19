"""Versioned, mode-specific sampling contracts; old presets stay independent."""

from copy import copy
import importlib
import logging
import math

LOGGER = logging.getLogger("MiniMaxH3.DirectorPlus")
BALANCED_PRESET = "balanced_8step"
BALANCED_LABEL = "均衡高质（768p 8步）"
BALANCED_LORAS = {
    "fl2va_model": "minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors",
    "ref2va_model": "minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors",
}


def balanced_recipe(guide):
    if (guide or {}).get("performance_preset") not in {BALANCED_PRESET, BALANCED_LABEL}:
        return None
    backend = guide.get("resolved_backend", "fl2va_model")
    if backend not in BALANCED_LORAS:
        raise ValueError(f"8 步配方不支持后端 {backend}")
    return {
        "id": f"lightx2v-768p-8step-v1/{backend}", "backend": backend,
        "lora_name": BALANCED_LORAS[backend], "lora_strength": 1.0,
        "steps": 8, "sampler": "euler", "scheduler": "simple",
        "video_shift": 6.0, "audio_shift": 3.0,
        "use_cache": False, "use_sage": True, "two_stage": False,
    }


def apply_balanced_shift(model, guide, *, installing=False):
    """Normalize a clone, including saved graphs with a downstream manual shift.

    A marker on the actual model, not just the mutable guide, proves that the
    matching adapter loader succeeded. Other presets never enter this path.
    """
    recipe = balanced_recipe(guide)
    if recipe is None:
        return model
    if not installing and getattr(model, "model_options", {}).get("mmh3_recipe_id") != recipe["id"]:
        raise ValueError("8 步配方的匹配 LoRA 未加载；MODEL 必须连接到加速路由输出")
    sampling = model.get_model_object("model_sampling")
    matches = all(math.isclose(float(getattr(sampling, attr, float('nan'))), recipe[key])
                  for attr, key in (("shift", "video_shift"), ("audio_shift", "audio_shift")))
    if matches and not installing:
        return model
    shift_node = importlib.import_module("comfy_extras.nodes_minimax_h3").MiniMaxH3SigmaShift
    result = shift_node.execute(model, recipe["video_shift"], recipe["audio_shift"])
    shifted = getattr(result, "result", result)[0]
    shifted.model_options["mmh3_recipe_id"] = recipe["id"]
    LOGGER.info("[H3 recipe] %s: fixed video/audio shift=6/3 (input=%s/%s)",
                recipe["id"], getattr(sampling, "shift", None), getattr(sampling, "audio_shift", None))
    return shifted


def prepare_balanced_guider(guider, guide):
    if balanced_recipe(guide) is None:
        return guider
    model = apply_balanced_shift(getattr(guider, "model_patcher", None), guide)
    if model is guider.model_patcher:
        return guider
    result = copy(guider)
    # Native CFGGuider/Guider_Basic exposes these fields, no set_model API.
    result.model_patcher = model
    result.model_options = model.model_options
    return result


def validate_balanced_sampling(model, sampler, sigmas, guide):
    recipe = balanced_recipe(guide)
    if recipe is None:
        return
    import torch
    native = importlib.import_module("comfy.k_diffusion.sampling")
    if getattr(sampler, "sampler_function", None) is not native.sample_euler or getattr(sampler, "extra_options", {}).get("s_churn", 0) != 0:
        raise ValueError("均衡高质 8 步必须使用标准 Euler 采样器")
    if getattr(model, "model_options", {}).get("mmh3_recipe_id") != recipe["id"]:
        raise ValueError("均衡高质 8 步缺少匹配 LoRA 模型")
    sampling = model.get_model_object("model_sampling")
    for attr, key in (("shift", "video_shift"), ("audio_shift", "audio_shift")):
        if not math.isclose(float(getattr(sampling, attr, float('nan'))), recipe[key]):
            raise ValueError("均衡高质 8 步需要 video/audio shift=6/3")
    expected = importlib.import_module("comfy.samplers").simple_scheduler(sampling, 8)
    if not isinstance(sigmas, torch.Tensor) or sigmas.ndim != 1 or len(sigmas) != 9 or not torch.allclose(sigmas.to(expected), expected, atol=1e-6, rtol=1e-5):
        raise ValueError("均衡高质 8 步 sigma 必须来自匹配的完整 simple 调度；不支持截断或额外二采")
