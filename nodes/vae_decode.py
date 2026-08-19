"""Memory-bounded MiniMax H3 video VAE decoding."""

from __future__ import annotations

import logging
import types

import torch


LOGGER = logging.getLogger("MiniMaxH3.DirectorPlus")


def should_use_gpu_output(output_device, frame_shape, free_memory=None):
    """Keep the complete H3 video frame buffer on CPU.

    ``free_memory`` is not a safe budget for a full video decode: it excludes
    VAE workspaces, model reloads, latent tensors, and downstream nodes that
    may overlap the decode allocation.  The previous adaptive GPU path could
    therefore take the process down after sampling had already succeeded.
    The VAE *compute* still runs on its configured GPU; only this inter-node
    output buffer is kept on host memory.
    """
    return False


class MiniMaxH3SafeVAEDecode:
    """Decode H3 video frames with a bounded CPU FP16 output buffer.

    H3's VAE already chunks its decoder temporally, but the standard VAE
    wrapper allocates the complete output using FP32 and may place it on the
    GPU when ComfyUI runs with ``--gpu-only``.  A long 2K clip can therefore
    be killed after sampling has completed.  This node keeps decoder math and
    latent inputs unchanged; only the inter-node frame buffer is bounded to
    CPU FP16.
    The VAE forward pass itself still runs on the configured GPU device. Only
    the complete IMAGE batch returned to downstream nodes is placed on CPU,
    preventing a second multi-GiB GPU allocation for long clips.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "samples": ("LATENT", {"tooltip": "H3 二采样后的最终视频 latent。"}),
            "vae": ("VAE", {"tooltip": "MiniMax H3 VideoVAE。"}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("视频帧（GPU计算 / CPU帧缓存 FP16）",)
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
        keep_gpu = False
        decode_device = torch.device("cpu")
        estimate = estimate.replace("CPU", "CPU")
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
        images = images.to(device="cpu", dtype=torch.float16, copy=False).contiguous()
        LOGGER.info("[H3 safe VAE] 视频解码完成 frames=%s dtype=%s device=%s", tuple(images.shape), images.dtype, images.device)
        return (images,)
