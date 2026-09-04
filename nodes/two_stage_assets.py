"""Exact assets and external-node contracts for trained H3 two-stage sampling."""

from __future__ import annotations

import inspect
import logging
from pathlib import Path


LOGGER = logging.getLogger("MiniMaxH3.DirectorPlus")


FL_STAGE1_LORA = "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"
FL_STAGE2_LORA = "minimax_h3_fl2v_turbo_4step_v1.1_768p_comfyui_bf16.safetensors"
REF_STAGE_LORA = "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"
# Community-proven U22 recipe: the 12-step-capable turbo v4 adapter drives both
# passes of a REF2VA trained two-stage run (8 high-sigma steps + 4-step
# low-sigma redraw after the learned 3D latent upscale).  The legacy
# ref2v_turbo_4step adapter stays listed for old workflows only.
V4_TURBO_LORA = "minimax_h3_turbo_v4_step600_ema.safetensors"
# drbaph's pruned ComfyUI conversion of the same v4 adapter (the exact file
# the U22 workflow ships with).  Either file satisfies the REF two-stage
# dependency and both load through the native LoRA path.
V4_TURBO_LORA_PRUNED = "minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors"
V4_TURBO_LORA_CANDIDATES = (V4_TURBO_LORA, V4_TURBO_LORA_PRUNED)
# U22-validated base checkpoint: the INT8 hybrid FL2VA+REF2VA model.  The
# pruned convrot/nvfp4 re-quantized checkpoints produce a woven cross-hatch
# texture under the INT8/FP8 SageAttention kernels (measured on RTX 4080),
# while this hybrid model stays clean with SageAttention on both stages.
HYBRID_BASE_MODEL = "minimax/minimax_h3_hybrid_fl2va_ref2va_b25-49-int8.safetensors"
# Community detail LoRAs from the U22 recipe.  They are stacked onto the REF
# two-stage chain (after the turbo adapter) whenever the files are registered,
# and skipped silently otherwise.  Order matches U22: physics first, detail
# enhancer second.
REF_DETAIL_LORA_CHAIN = (
    ("wushu_spatial_physics_v2_1000_pruned.safetensors", 0.3),
    ("MysticXXX_MMH3-V1.safetensors", 0.5),
)
# Community add-ons beyond the U22 recipe (user-requested): a general
# motion-continuity fix (四只兔子) and a cinematic-look adapter that removes
# the "plastic" AI look.  Each entry lists acceptable file names (the first
# registered basename wins) plus a conservative starting strength.
# MMH3_EXTRA_LORAS=0 disables this chain; strengths are starting points and
# can be tuned after A/B runs.
REF_EXTRA_LORA_CHAIN = (
    (("动作i连续性修复LORA.safetensors", "动作连续性修复LORA.safetensors"), 0.4),
    (("MinimaxH3真实电影质感V1.0.safetensors", "MinimaxH3真实电影质感V0.1.safetensors"), 0.5),
)
LATENT_UPSCALER_MODEL_FP16 = "minimax_h3_latent_upscaler_3d_fp16.safetensors"
LATENT_UPSCALER_MODEL = "minimax_h3_latent_upscaler_3d_bf16.safetensors"
# U22 runs the fp16 build of the trained 3D upscaler; its finer mantissa keeps
# flat regions clean where the bf16 build leaves a fine latent-grid weave that
# survives the low-sigma redraw.  Prefer fp16 whenever it is registered and
# keep bf16 as the fallback.
LATENT_UPSCALER_MODEL_CANDIDATES = (LATENT_UPSCALER_MODEL_FP16, LATENT_UPSCALER_MODEL)
UPSCALE_NODE_IDS = ("MinimaxH3LatentUpscaler3D", "MinimaxH3LatentUpscalerNode3D")
SIGMA_REFINER_NODE_ID = "H3SigmaRefiner"
SPLIT_UPSCALE_NODE_ID = "MMH3SplitUpscale"
SPLIT_TEMPORAL_NODE_ID = "MMH3TemporalSplitParamsV10"
SPLIT_SPATIAL_NODE_ID = "MMH3SpatialSplitParamsV10"


def resolve_two_stage_route(guide):
    """Resolve the one trained route compatible with the active H3 backend."""
    guide = guide or {}
    if str(guide.get("performance_preset", "")) == "smart_free_1080p":
        raise ValueError("免费智能 1080p 必须先由导演台解析为具体性能预设")
    if str(guide.get("voice_mode", "none")) == "fish_lock":
        return "bypass"
    if str(guide.get("resolved_backend", "fl2va_model")) == "ref2va_model":
        # U22-validated: turbo v4 (12-step) + trained 3D latent upscale works
        # on the REF2VA backend and preserves the audio latent via the LTXV
        # AV split/concat contract.
        return "trained_latent_ref"
    return "trained_latent_fl"


def _required_assets(route):
    if route == "trained_latent_ref":
        return (V4_TURBO_LORA,)
    if route == "trained_latent_fl":
        return (FL_STAGE1_LORA, FL_STAGE2_LORA)
    return ()


def resolve_registered_model_name(category, name, comfy_root=None):
    """Resolve an exact basename to ComfyUI's registered relative model name."""
    try:
        import folder_paths

        if comfy_root is not None:
            runtime_root = getattr(folder_paths, "base_path", None)
            if runtime_root is None:
                return None
            if Path(runtime_root).resolve() != Path(comfy_root).resolve():
                return None

        if folder_paths.get_full_path(category, name):
            return name

        basename = Path(name).name
        for candidate in folder_paths.get_filename_list(category):
            candidate = str(candidate)
            if Path(candidate).name != basename:
                continue
            if folder_paths.get_full_path(category, candidate):
                return candidate
    except (AttributeError, ImportError, KeyError, OSError, TypeError, ValueError):
        return None
    return None


def resolve_latent_upscaler_model(comfy_root=None):
    """Return ``(registered_name, precision)`` for the best installed upscaler."""
    for name in LATENT_UPSCALER_MODEL_CANDIDATES:
        resolved = resolve_registered_model_name("latent_upscale_models", name, comfy_root)
        if resolved is not None:
            return resolved, ("fp16" if "fp16" in name else "bf16")
    return None


def _latent_upscaler_variant():
    resolved = resolve_latent_upscaler_model()
    if resolved is not None:
        return resolved
    return LATENT_UPSCALER_MODEL, "bf16"


def _registered_asset_exists(comfy_root, category, name):
    """Check the same model paths registered with the active ComfyUI runtime."""
    return resolve_registered_model_name(category, name, comfy_root) is not None


def _asset_exists(comfy_root, category, directory, name):
    return _registered_asset_exists(comfy_root, category, name) or (
        Path(comfy_root) / "models" / directory / name
    ).is_file()


def resolve_v4_turbo_lora_name(comfy_root=None):
    """Return the installed turbo-v4 adapter name, preferring the pruned conversion."""
    for name in (V4_TURBO_LORA_PRUNED, V4_TURBO_LORA):
        if comfy_root is None:
            if resolve_registered_model_name("loras", name) is not None:
                return name
        elif _asset_exists(comfy_root, "loras", "loras", name):
            return name
    return None


def dependency_report(comfy_root, route, node_mappings=None):
    """Return route-specific missing assets without loading any model."""
    root = Path(comfy_root)
    mappings = dict(node_mappings or {})
    if route == "bypass":
        return {
            "route": route,
            "ready": True,
            "missing": [],
            "upscaler_node_id": None,
            "required_assets": [],
        }

    upscaler_node_id = next(
        (node_id for node_id in UPSCALE_NODE_IDS if node_id in mappings),
        None,
    )
    missing = []
    if upscaler_node_id is None:
        missing.append(UPSCALE_NODE_IDS[0])
    # The revived U22-style REF route runs plain beta/Euler sigmas; the legacy
    # H3SigmaRefiner belonged to the banned 4-step ref recipe and is no longer
    # a dependency.

    if not any(
        _asset_exists(root, "latent_upscale_models", "latent_upscale_models", candidate)
        for candidate in LATENT_UPSCALER_MODEL_CANDIDATES
    ):
        missing.append(" 或 ".join(LATENT_UPSCALER_MODEL_CANDIDATES))

    loras = _required_assets(route)
    if route == "trained_latent_ref":
        # The v4 adapter ships under two file names (original + pruned
        # ComfyUI conversion); either one satisfies the dependency.
        if resolve_v4_turbo_lora_name(root) is None:
            missing.append(" 或 ".join(V4_TURBO_LORA_CANDIDATES))
        required_loras = list(V4_TURBO_LORA_CANDIDATES)
    else:
        for name in loras:
            if not _asset_exists(root, "loras", "loras", name):
                missing.append(name)
        required_loras = list(loras)

    return {
        "route": route,
        "ready": not missing,
        "missing": missing,
        "upscaler_node_id": upscaler_node_id,
        "required_assets": [*LATENT_UPSCALER_MODEL_CANDIDATES, *required_loras],
    }


def _comfy_node_mappings():
    import nodes as comfy_nodes

    return getattr(comfy_nodes, "NODE_CLASS_MAPPINGS", {})


def _unwrap_node_output(result):
    result = getattr(result, "result", result)
    if isinstance(result, (tuple, list)):
        return result[0]
    return result


def _is_unbound_self_method(function):
    if not inspect.isfunction(function):
        return False
    parameters = tuple(inspect.signature(function).parameters.values())
    return bool(parameters) and parameters[0].name == "self"


def _resolve_upscaler_callable(node_class):
    """Resolve the callable while bypassing ComfyUI v3's normalized wrapper."""
    if not callable(node_class):
        raise RuntimeError("训练型 H3 latent 放大节点没有可调用实现")
    function_name = getattr(node_class, "FUNCTION", None)
    node = None
    if function_name is None:
        node = node_class()
        function_name = getattr(node, "FUNCTION", "execute")
    if str(function_name).startswith("EXECUTE_NORMALIZED"):
        class_execute = getattr(node_class, "execute", None)
        if callable(class_execute):
            if _is_unbound_self_method(class_execute):
                node = node_class()
                return node.execute
            return class_execute

    if node is None:
        node = node_class()
    function = getattr(node, function_name, None)
    if not callable(function):
        raise RuntimeError(
            f"训练型 H3 latent 放大节点没有可调用实现：{function_name}"
        )
    return function


def _upscaler_kwargs(function, video_latent, scale, model_name=None, precision=None):
    model_name = model_name or LATENT_UPSCALER_MODEL
    precision = precision or "bf16"
    parameters = inspect.signature(function).parameters
    positional_only_required = [
        name
        for name, parameter in parameters.items()
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY
        and parameter.default is inspect.Parameter.empty
    ]
    if positional_only_required:
        raise RuntimeError(
            "训练型 H3 latent 放大节点缺少必需参数："
            + ", ".join(positional_only_required)
        )
    declared = {
        name: parameter
        for name, parameter in parameters.items()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    if "mode" not in declared and "scale" not in declared:
        raise RuntimeError(
            "训练型 H3 latent 放大节点调用契约必须声明 mode 或 scale"
        )

    kwargs = {
        "latent": video_latent,
        "model_name": model_name,
        "device": "cuda",
        "precision": precision,
    }
    if "mode" in declared:
        kwargs["mode"] = {"mode": "scale by multiplier", "scale": float(scale)}
    else:
        kwargs["scale"] = float(scale)
    kwargs.update(
        {
            "align": 32,
            "enable_temporal_chunking": True,
            "force_unload": True,
            "enable_chunking": True,
            "keep_proportion": False,
        }
    )
    kwargs = {name: value for name, value in kwargs.items() if name in declared}
    missing = [
        name
        for name, parameter in declared.items()
        if parameter.default is inspect.Parameter.empty and name not in kwargs
    ]
    if missing:
        raise RuntimeError(
            "训练型 H3 latent 放大节点缺少必需参数：" + ", ".join(missing)
        )
    return kwargs


def resolve_split_upscale_callables(node_mappings=None):
    """Resolve the MMH3 tiled second-stage chain when the plugin is installed.

    Returns ``(upscale, temporal, spatial)`` callables, with the two
    parameter-node callables set to ``None`` when only the main node is
    registered. Returns ``None`` when ``MMH3SplitUpscale`` is absent so the
    caller can fall back to the single-pass full-frame second sampler.
    """
    mappings = dict(node_mappings) if node_mappings is not None else _comfy_node_mappings()
    node_class = mappings.get(SPLIT_UPSCALE_NODE_ID)
    if node_class is None:
        return None
    upscale = _resolve_upscaler_callable(node_class)
    temporal_class = mappings.get(SPLIT_TEMPORAL_NODE_ID)
    spatial_class = mappings.get(SPLIT_SPATIAL_NODE_ID)
    temporal = (
        _resolve_upscaler_callable(temporal_class)
        if temporal_class is not None
        else None
    )
    spatial = (
        _resolve_upscaler_callable(spatial_class)
        if spatial_class is not None
        else None
    )
    return upscale, temporal, spatial


def run_trained_latent_upscaler(video_latent, scale):
    """Run the installed learned H3 3D upscaler on video latent only."""
    import torch

    source_samples = (
        video_latent.get("samples") if isinstance(video_latent, dict) else None
    )
    if not isinstance(source_samples, torch.Tensor) or source_samples.ndim not in (4, 5):
        raise RuntimeError("训练型 H3 latent 放大输入无效")

    mappings = _comfy_node_mappings()
    node_id = next((name for name in UPSCALE_NODE_IDS if name in mappings), None)
    if node_id is None:
        raise RuntimeError("缺少 MinimaxH3LatentUpscaler3D，请先安装训练型 H3 latent 放大节点")
    node_class = mappings[node_id]
    function = _resolve_upscaler_callable(node_class)
    upscaler_model, upscaler_precision = _latent_upscaler_variant()
    LOGGER.info(
        "[H3 two-stage] 训练型 latent 放大器：%s（精度 %s）",
        upscaler_model,
        upscaler_precision,
    )
    kwargs = _upscaler_kwargs(
        function,
        video_latent,
        scale,
        model_name=upscaler_model,
        precision=upscaler_precision,
    )
    result = _unwrap_node_output(function(**kwargs))
    samples = result.get("samples") if isinstance(result, dict) else None
    if not isinstance(samples, torch.Tensor) or samples.ndim not in (4, 5):
        raise RuntimeError("训练型 H3 latent 放大节点返回了无效结果")
    if samples.ndim != source_samples.ndim:
        raise RuntimeError(
            "训练型 H3 latent 放大结果维度与输入不一致："
            f"{source_samples.ndim}D -> {samples.ndim}D"
        )
    if int(samples.shape[1]) != 24:
        raise RuntimeError(f"训练型 H3 latent 放大结果通道数错误：{samples.shape[1]}，应为 24")
    if tuple(samples.shape[:-2]) != tuple(source_samples.shape[:-2]):
        raise RuntimeError(
            "训练型 H3 latent 放大结果非空间尺寸与输入不一致："
            f"{tuple(source_samples.shape[:-2])} -> {tuple(samples.shape[:-2])}"
        )
    if float(scale) > 1.0 and (
        int(samples.shape[-2]) <= int(source_samples.shape[-2])
        or int(samples.shape[-1]) <= int(source_samples.shape[-1])
    ):
        raise RuntimeError(
            "训练型 H3 latent 放大结果空间尺寸没有增长："
            f"{tuple(source_samples.shape[-2:])} -> {tuple(samples.shape[-2:])}"
        )
    return result
