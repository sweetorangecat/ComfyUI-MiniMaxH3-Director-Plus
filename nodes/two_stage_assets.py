"""Exact assets and external-node contracts for trained H3 two-stage sampling."""

from __future__ import annotations

import inspect
from pathlib import Path


FL_STAGE1_LORA = "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"
FL_STAGE2_LORA = "minimax_h3_fl2v_turbo_4step_v1.1_768p_comfyui_bf16.safetensors"
REF_STAGE_LORA = "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"
LATENT_UPSCALER_MODEL = "minimax_h3_latent_upscaler_3d_bf16.safetensors"
UPSCALE_NODE_IDS = ("MinimaxH3LatentUpscaler3D", "MinimaxH3LatentUpscalerNode3D")
SIGMA_REFINER_NODE_ID = "H3SigmaRefiner"


def resolve_two_stage_route(guide):
    """Resolve the one trained route compatible with the active H3 backend."""
    guide = guide or {}
    if str(guide.get("performance_preset", "")) == "smart_free_1080p":
        raise ValueError("免费智能 1080p 必须先由导演台解析为具体性能预设")
    if str(guide.get("voice_mode", "none")) == "fish_lock":
        return "bypass"
    if str(guide.get("resolved_backend", "fl2va_model")) == "ref2va_model":
        raise ValueError(
            "REF2VA 不支持训练型二采，请使用 quality_sage、low_vram 或 ref_fast_4step"
        )
    return "trained_latent_fl"


def _required_assets(route):
    if route == "trained_latent_ref":
        return (REF_STAGE_LORA,)
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


def _registered_asset_exists(comfy_root, category, name):
    """Check the same model paths registered with the active ComfyUI runtime."""
    return resolve_registered_model_name(category, name, comfy_root) is not None


def _asset_exists(comfy_root, category, directory, name):
    return _registered_asset_exists(comfy_root, category, name) or (
        Path(comfy_root) / "models" / directory / name
    ).is_file()


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
    if route == "trained_latent_ref" and SIGMA_REFINER_NODE_ID not in mappings:
        missing.append(SIGMA_REFINER_NODE_ID)

    if not _asset_exists(
        root,
        "latent_upscale_models",
        "latent_upscale_models",
        LATENT_UPSCALER_MODEL,
    ):
        missing.append(LATENT_UPSCALER_MODEL)

    loras = _required_assets(route)
    for name in loras:
        if not _asset_exists(root, "loras", "loras", name):
            missing.append(name)

    return {
        "route": route,
        "ready": not missing,
        "missing": missing,
        "upscaler_node_id": upscaler_node_id,
        "required_assets": [LATENT_UPSCALER_MODEL, *loras],
    }


def _comfy_node_mappings():
    import nodes as comfy_nodes

    return getattr(comfy_nodes, "NODE_CLASS_MAPPINGS", {})


def _unwrap_node_output(result):
    result = getattr(result, "result", result)
    if isinstance(result, (tuple, list)):
        return result[0]
    return result


def run_trained_latent_upscaler(video_latent, scale):
    """Run the installed learned H3 3D upscaler on video latent only."""
    mappings = _comfy_node_mappings()
    node_id = next((name for name in UPSCALE_NODE_IDS if name in mappings), None)
    if node_id is None:
        raise RuntimeError("缺少 MinimaxH3LatentUpscaler3D，请先安装训练型 H3 latent 放大节点")
    node_class = mappings[node_id]
    node = node_class()
    function_name = getattr(node, "FUNCTION", getattr(node_class, "FUNCTION", "execute"))
    function = getattr(node, function_name)
    signature_target = function
    if str(function_name).startswith("EXECUTE_NORMALIZED"):
        signature_target = getattr(node_class, "execute", function)
    parameters = inspect.signature(signature_target).parameters
    kwargs = {
        "latent": video_latent,
        "model_name": LATENT_UPSCALER_MODEL,
        "device": "cuda",
        "precision": "bf16",
    }
    if "mode" in parameters:
        kwargs.update(
            mode={"mode": "scale by multiplier", "scale": float(scale)},
            align=32,
        )
        if "enable_chunking" in parameters:
            kwargs["enable_chunking"] = True
        if "keep_proportion" in parameters:
            kwargs["keep_proportion"] = False
    else:
        kwargs["scale"] = float(scale)
        if "align" in parameters:
            kwargs["align"] = 32
        if "enable_chunking" in parameters:
            kwargs["enable_chunking"] = True
    kwargs = {name: value for name, value in kwargs.items() if name in parameters}
    result = _unwrap_node_output(function(**kwargs))
    samples = result.get("samples") if isinstance(result, dict) else None
    if samples is None or getattr(samples, "ndim", 0) not in (4, 5):
        raise RuntimeError("训练型 H3 latent 放大节点返回了无效结果")
    if int(samples.shape[1]) != 24:
        raise RuntimeError(f"训练型 H3 latent 放大结果通道数错误：{samples.shape[1]}，应为 24")
    return result
