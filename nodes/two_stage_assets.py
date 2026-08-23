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
    if str(guide.get("voice_mode", "none")) == "fish_lock":
        return "bypass"
    if str(guide.get("resolved_backend", "fl2va_model")) == "ref2va_model":
        return "trained_latent_ref"
    return "trained_latent_fl"


def _required_assets(route):
    if route == "trained_latent_ref":
        return (REF_STAGE_LORA,)
    if route == "trained_latent_fl":
        return (FL_STAGE1_LORA, FL_STAGE2_LORA)
    return ()


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

    latent_path = root / "models" / "latent_upscale_models" / LATENT_UPSCALER_MODEL
    if not latent_path.is_file():
        missing.append(LATENT_UPSCALER_MODEL)

    lora_dir = root / "models" / "loras"
    loras = _required_assets(route)
    for name in loras:
        if not (lora_dir / name).is_file():
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
    parameters = inspect.signature(function).parameters
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
            enable_chunking=True,
        )
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
