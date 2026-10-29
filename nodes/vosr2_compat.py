"""Compatibility bridge for the current ComfyUI-VOSR2 extension.

The director's VOSR2 path was written against the older TE-Speed node
contract.  Recent ComfyUI-VOSR2 exposes a v3 extension instead, so the
director keeps its stable three-node contract and adapts the installed v3
loader/inference functions lazily.  This module has no model imports at
module load time; a machine without ComfyUI-VOSR2 simply remains unavailable
and receives the existing dependency error.
"""

from __future__ import annotations

import gc
import importlib
import logging
from typing import Any


LOGGER = logging.getLogger("MiniMaxH3.DirectorPlus")


def _load_current_vosr2():
    """Return the installed v3 VOSR2 module, or ``None`` when unavailable."""
    try:
        module = importlib.import_module("custom_nodes.ComfyUI-VOSR2.nodes_vosr")
    except (ImportError, ModuleNotFoundError, AttributeError, OSError):
        return None
    required = ("VOSR2ModelLoader", "run_vosr2")
    return module if all(hasattr(module, name) for name in required) else None


def _unwrap_node_output(value: Any):
    value = getattr(value, "result", value)
    while isinstance(value, (tuple, list)) and len(value) == 1:
        value = value[0]
    return value


def _dtype_name(precision: str) -> str:
    # The legacy adapter used ``auto``; the v3 node calls its matching option
    # ``default`` and resolves the actual ComfyUI dtype at runtime.
    if precision in (None, "", "auto", "default"):
        return "default"
    if precision in ("fp16", "bf16"):
        return precision
    raise ValueError(f"不支持的 VOSR2 精度：{precision!r}")


class TESpeedVOSR2Loader:
    FUNCTION = "load"

    def load(
        self,
        model_bundle="VOSR2",
        precision="auto",
        memory_policy="auto",
        torch_compile=False,
        vae_encode_amp=False,
    ):
        del memory_policy, torch_compile, vae_encode_amp
        module = _load_current_vosr2()
        if module is None:
            raise RuntimeError("当前 ComfyUI-VOSR2 节点不可用")
        result = module.VOSR2ModelLoader.execute(
            model=model_bundle,
            dtype=_dtype_name(precision),
        )
        model = _unwrap_node_output(result)
        if model is None:
            raise RuntimeError("ComfyUI-VOSR2 加载器没有返回模型")
        LOGGER.info(
            "[H3 VOSR2 compat] loaded current ComfyUI-VOSR2 bundle=%s precision=%s",
            model_bundle,
            precision,
        )
        return (model,)


class TESpeedVOSR2Settings:
    FUNCTION = "settings"

    def settings(self, **kwargs):
        # Keep the old settings object as a plain mapping.  The current
        # implementation accepts the concrete tile/color values below; the
        # remaining values are retained for execution reports and compatibility.
        settings = dict(kwargs)
        settings.setdefault("tile_size", 512)
        settings.setdefault("tile_overlap", 32)
        settings.setdefault("vae_tile_size", 1024)
        settings.setdefault("vae_tile_overlap", 32)
        settings.setdefault("color_alignment", "wavelet")
        return (settings,)


class TESpeedVOSR2Video:
    FUNCTION = "upscale"
    # The bridge batches independent video frames, but the v3 implementation
    # still performs its latent tiles one at a time.  Advertising False keeps
    # the existing 32GB experimental profile from assuming tile parallelism.
    H3_TILE_BATCH_SUPPORTED = False

    @staticmethod
    def H3_RELEASE_ENGINES():
        """Release v3 VOSR2 patchers and CUDA scratch after a complete clip."""
        try:
            import comfy.model_management as model_management

            cleanup = getattr(model_management, "soft_empty_cache", None)
            if callable(cleanup):
                cleanup()
        except (ImportError, AttributeError, RuntimeError):
            pass
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except (ImportError, AttributeError, RuntimeError):
            pass
        gc.collect()
        return "released"

    def upscale(
        self,
        model,
        images,
        scale,
        seed,
        settings,
        frame_batch=2,
        temporal_cache=False,
        cache_threshold=0.003,
        cache_refresh=4,
    ):
        del temporal_cache, cache_threshold, cache_refresh
        module = _load_current_vosr2()
        if module is None:
            raise RuntimeError("当前 ComfyUI-VOSR2 节点不可用")
        if not hasattr(images, "shape") or len(images.shape) != 4 or images.shape[-1] != 3:
            raise RuntimeError("VOSR2 输入必须是 [帧数, 高, 宽, 3] IMAGE")
        scale = int(scale)
        if scale < 1:
            raise ValueError(f"VOSR2 scale 必须是正整数：{scale}")
        frame_batch = max(1, int(frame_batch))
        settings = _unwrap_node_output(settings)
        if not isinstance(settings, dict):
            settings = {}
        color_alignment = settings.get("color_alignment", "wavelet")
        tile_size = int(settings.get("tile_size", 512))
        tile_overlap = int(settings.get("tile_overlap", 32))
        vae_tile_size = int(settings.get("vae_tile_size", 1024))
        vae_tile_overlap = int(settings.get("vae_tile_overlap", 32))

        outputs = []
        total = int(images.shape[0])
        for start in range(0, total, frame_batch):
            end = min(total, start + frame_batch)
            chunk = module.run_vosr2(
                model,
                images[start:end],
                scale,
                int(seed) + start,
                color_alignment,
                tile_size,
                tile_overlap,
                vae_tile_size,
                vae_tile_overlap,
            )
            chunk = _unwrap_node_output(chunk)
            if not hasattr(chunk, "shape") or len(chunk.shape) != 4:
                raise RuntimeError("ComfyUI-VOSR2 返回了无效 IMAGE")
            if int(chunk.shape[0]) != end - start:
                raise RuntimeError("VOSR2 处理改变了帧数")
            if tuple(chunk.shape[1:3]) != (
                int(images.shape[1]) * scale,
                int(images.shape[2]) * scale,
            ):
                raise RuntimeError("VOSR2 返回尺寸与 scale 不一致")
            outputs.append(chunk.detach().cpu() if hasattr(chunk, "detach") else chunk)

        if not outputs:
            raise RuntimeError("VOSR2 没有输出视频帧")
        import torch

        result = torch.cat(outputs, dim=0)
        LOGGER.info(
            "[H3 VOSR2 compat] frames=%s scale=%s frame_batch=%s tile=%sx%s",
            total,
            scale,
            frame_batch,
            tile_size,
            tile_overlap,
        )
        return (result,)


def compatibility_mappings() -> dict[str, type]:
    """Return aliases only when the installed v3 extension is usable."""
    if _load_current_vosr2() is None:
        return {}
    return {
        "TESpeedVOSR2Loader": TESpeedVOSR2Loader,
        "TESpeedVOSR2Settings": TESpeedVOSR2Settings,
        "TESpeedVOSR2Video": TESpeedVOSR2Video,
    }

