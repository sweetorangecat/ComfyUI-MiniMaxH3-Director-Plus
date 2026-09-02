"""SeedVR2 diffusion video super-resolution route for U11 postprocess.

SeedVR2 (ByteDance, one-step diffusion video SR) keeps temporal consistency
that per-frame GAN upscalers cannot, and its ComfyUI integration supports
BlockSwap / VAE tiling / GGUF so even 8GB cards can reach 1080p.  This module
owns dependency detection, hardware-tiered plans, and callable resolution for
the installed ``ComfyUI-SeedVR2_VideoUpscaler`` nodes.  Everything degrades
gracefully: when nodes or weights are missing, callers fall back to the
per-frame ``ai_upscale`` route.
"""

from __future__ import annotations

from .two_stage_assets import (
    _asset_exists,
    _comfy_node_mappings,
    _resolve_upscaler_callable,
)


SEEDVR2_UPSCALER_NODE_ID = "SeedVR2VideoUpscaler"
SEEDVR2_DIT_LOADER_NODE_ID = "SeedVR2LoadDiTModel"
SEEDVR2_VAE_LOADER_NODE_ID = "SeedVR2LoadVAEModel"
SEEDVR2_NODE_IDS = (
    SEEDVR2_UPSCALER_NODE_ID,
    SEEDVR2_DIT_LOADER_NODE_ID,
    SEEDVR2_VAE_LOADER_NODE_ID,
)

SEEDVR2_MODEL_CATEGORY = "seedvr2"
SEEDVR2_MODEL_DIRECTORY = "SEEDVR2"
SEEDVR2_VAE_MODEL = "ema_vae_fp16.safetensors"

# Ordered by quality preference within each hardware tier.
SEEDVR2_DIT_LOW_VRAM = (
    "seedvr2_ema_3b-Q4_K_M.gguf",
    "seedvr2_ema_3b-Q8_0.gguf",
    "seedvr2_ema_3b_fp8_e4m3fn.safetensors",
)
SEEDVR2_DIT_FULL = (
    "seedvr2_ema_3b_fp8_e4m3fn.safetensors",
    "seedvr2_ema_3b_fp16.safetensors",
    "seedvr2_ema_3b-Q8_0.gguf",
    "seedvr2_ema_3b-Q4_K_M.gguf",
)


def _pick_dit_model(candidates, available_dit=None):
    if available_dit is None:
        return candidates[0]
    available = {str(name) for name in available_dit}
    for name in candidates:
        if name in available:
            return name
    return None


def resolve_seedvr2_plan(total_vram_gb, available_dit=None):
    """Return a hardware-tiered SeedVR2 configuration for the final SR pass.

    Tiers follow the official ComfyUI integration guidance: GGUF + aggressive
    BlockSwap + tiled VAE at 8GB, lighter swap at 12-20GB, and unswapped FP8
    with larger 4n+1 batches at 24GB+.  VAE decode stays tiled on every tier:
    an untiled 1080p+ portrait decode spikes past 2.5GB per chunk and collides
    with the still-resident DiT, which is exactly what OOMs 32GB cards.
    """
    total = float(total_vram_gb)
    if total <= 12.0:
        dit = _pick_dit_model(SEEDVR2_DIT_LOW_VRAM, available_dit)
        blocks = 32 if total <= 9.0 else 12
        return {
            "dit_model": dit,
            "vae_model": SEEDVR2_VAE_MODEL,
            "dit_offload_device": "cpu",
            "blocks_to_swap": blocks,
            "swap_io_components": True,
            "encode_tiled": True,
            "decode_tiled": True,
            "batch_size": 5,
            "temporal_overlap": 0,
            "color_correction": "lab",
        }
    if total <= 20.0:
        return {
            "dit_model": _pick_dit_model(SEEDVR2_DIT_FULL, available_dit),
            "vae_model": SEEDVR2_VAE_MODEL,
            "dit_offload_device": "cpu",
            "blocks_to_swap": 6,
            "swap_io_components": True,
            "encode_tiled": True,
            "decode_tiled": True,
            "batch_size": 9,
            "temporal_overlap": 0,
            "color_correction": "lab",
        }
    return {
        "dit_model": _pick_dit_model(SEEDVR2_DIT_FULL, available_dit),
        "vae_model": SEEDVR2_VAE_MODEL,
        "dit_offload_device": "none",
        "blocks_to_swap": 0,
        "swap_io_components": False,
        "encode_tiled": False,
        "decode_tiled": True,
        "batch_size": 13,
        "temporal_overlap": 0,
        "color_correction": "lab",
    }


def _available_seedvr2_models(comfy_root, name):
    return _asset_exists(
        comfy_root, SEEDVR2_MODEL_CATEGORY, SEEDVR2_MODEL_DIRECTORY, name
    )


def seedvr2_dependency_report(comfy_root=None, node_mappings=None):
    """Report SeedVR2 node/weight readiness without loading any model."""
    mappings = dict(node_mappings) if node_mappings is not None else _comfy_node_mappings()
    missing = [node_id for node_id in SEEDVR2_NODE_IDS if node_id not in mappings]

    dit_candidates = list(dict.fromkeys([*SEEDVR2_DIT_FULL, *SEEDVR2_DIT_LOW_VRAM]))
    dit_model = None
    available = []
    if comfy_root is not None:
        available = [
            name
            for name in dit_candidates
            if _available_seedvr2_models(comfy_root, name)
        ]
        dit_model = _pick_dit_model(SEEDVR2_DIT_FULL, available) or _pick_dit_model(
            SEEDVR2_DIT_LOW_VRAM, available
        )
        if dit_model is None:
            missing.append("seedvr2_ema_3b (FP8/GGUF)")
        if not _available_seedvr2_models(comfy_root, SEEDVR2_VAE_MODEL):
            missing.append(SEEDVR2_VAE_MODEL)
    else:
        dit_model = dit_candidates[0]
        available = dit_candidates

    return {
        "ready": not missing,
        "missing": missing,
        "dit_model": dit_model,
        "available_dit": available,
        "vae_model": SEEDVR2_VAE_MODEL,
        "upscaler_node_id": SEEDVR2_UPSCALER_NODE_ID,
    }


def resolve_seedvr2_callables(node_mappings=None):
    """Resolve (upscaler, dit_loader, vae_loader) callables, or None.

    Returns None when any of the three nodes is absent so callers can fall
    back to the per-frame AI upscale route.
    """
    mappings = dict(node_mappings) if node_mappings is not None else _comfy_node_mappings()
    if any(node_id not in mappings for node_id in SEEDVR2_NODE_IDS):
        return None
    return tuple(
        _resolve_upscaler_callable(mappings[node_id]) for node_id in SEEDVR2_NODE_IDS
    )
