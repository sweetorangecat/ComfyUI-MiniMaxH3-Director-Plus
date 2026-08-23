"""Exact assets and external-node contracts for trained H3 two-stage sampling."""

from __future__ import annotations

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
