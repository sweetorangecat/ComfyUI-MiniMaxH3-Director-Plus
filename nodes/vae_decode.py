"""Memory-bounded MiniMax H3 video VAE decoding."""

from __future__ import annotations

import logging
import types

import torch


LOGGER = logging.getLogger("MiniMaxH3.DirectorPlus")


GPU_OUTPUT_LIMIT_BYTES = 12 * 1024**3


def should_use_gpu_output(output_device, frame_shape, free_memory=None):
    """Choose a fast GPU output buffer only when its peak is bounded."""
    if getattr(output_device, "type", None) != "cuda":
        return False
    if not frame_shape:
        return False
    elements = 1
    for value in frame_shape:
        elements *= int(value)
    # The output buffer is FP16. When ComfyUI can report free VRAM, reserve
    # half of it for decoder workspaces and the loaded VAE. On older builds
    # without that API, keep the conservative 3 GiB fallback.
    output_bytes = elements * 2
    if free_memory is None:
        return output_bytes <= 3 * 1024**3
    if output_bytes > GPU_OUTPUT_LIMIT_BYTES:
        return False
    if output_bytes * 2 > int(free_memory):
        return False
    return True


class MiniMaxH3SafeVAEDecode:
    """Decode H3 video frames with an adaptive GPU/CPU FP16 output buffer.

    H3's VAE already chunks its decoder temporally, but the standard VAE
    wrapper allocates the complete output using FP32 and may place it on the
    GPU when ComfyUI runs with ``--gpu-only``.  A long 2K clip can therefore
    be killed after sampling has completed.  This node keeps decoder math and
    latent inputs unchanged; only the inter-node frame buffer is bounded to
    CPU FP16.
    Small clips stay entirely on the GPU for speed; only large outputs use a
    CPU frame buffer to avoid a process-killing allocation peak. In both cases
    the VAE forward pass itself runs on the configured GPU device.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "samples": ("LATENT", {"tooltip": "H3 二采样后的最终视频 latent。"}),
            "vae": ("VAE", {"tooltip": "MiniMax H3 VideoVAE。"}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("视频帧（自适应 GPU/CPU FP16）",)
    FUNCTION = "decode"
    CATEGORY = "MiniMax H3 导演台 Plus"

    def decode(self, vae, samples):
        latent = samples["samples"]
        if getattr(latent, "is_nested", False):
            latent = latent.unbind()[0]

        old_output_device = getattr(vae, "output_device", None)
        old_dtype_method = getattr(vae, "vae_output_dtype", None)
        if old_dtype_method is None:
            raise RuntimeError("H3 VAE 缺少 vae_output_dtype 接口，无法启用安全解码")

        # Estimate only the final RGB buffer. Decoder workspaces continue to
        # use the VAE's configured dtype and ComfyUI's own temporal tiling.
        frames = None
        try:
            shape_fn = getattr(getattr(vae, "first_stage_model", None), "decode_output_shape", None)
            if callable(shape_fn):
                frames = tuple(int(value) for value in shape_fn(latent.shape))
        except (AttributeError, RuntimeError, TypeError, ValueError):
            frames = None
        estimate = "未知"
        if frames and len(frames) == 5:
            estimate = f"{frames[0]}x{frames[2]}x{frames[3]}x{frames[4]} FP16 CPU"
        original_output_device = getattr(vae, "output_device", torch.device("cpu"))
        compute_device = getattr(vae, "device", original_output_device)
        output_device = compute_device if getattr(compute_device, "type", None) == "cuda" else original_output_device
        free_memory = None
        if getattr(output_device, "type", None) == "cuda":
            try:
                import comfy.model_management as model_management
                free_memory = model_management.get_free_memory(output_device)
            except (ImportError, AttributeError, RuntimeError, TypeError):
                free_memory = None
        keep_gpu = should_use_gpu_output(output_device, frames, free_memory)
        decode_device = output_device if keep_gpu else torch.device("cpu")
        estimate = estimate.replace("CPU", "GPU" if keep_gpu else "CPU")
        LOGGER.info(
            "[H3 safe VAE] 开始视频解码 latent=%s，输出缓冲=%s，VAE计算设备=%s，帧缓存=%s",
            tuple(latent.shape), estimate, compute_device, decode_device,
        )

        # The VAE wrapper reads both attributes while allocating/copying its
        # output. Restore them even when decode raises or the process is
        # interrupted by a ComfyUI execution error.
        vae.output_device = decode_device
        vae.vae_output_dtype = types.MethodType(lambda _self: torch.float16, vae)
        try:
            with torch.inference_mode():
                images = vae.decode(latent)
        finally:
            if old_output_device is not None:
                vae.output_device = old_output_device
            if old_dtype_method is not None:
                vae.vae_output_dtype = old_dtype_method

        if len(images.shape) == 5:
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        if keep_gpu:
            images = images.to(device=output_device, dtype=torch.float16, copy=False).contiguous()
        else:
            images = images.to(device="cpu", dtype=torch.float16, copy=False).contiguous()
        LOGGER.info("[H3 safe VAE] 视频解码完成 frames=%s dtype=%s device=%s", tuple(images.shape), images.dtype, images.device)
        return (images,)
