"""Stream final-size H3 frames to FFmpeg without keeping a full 4K batch."""

from __future__ import annotations

import importlib
import logging
import os
import subprocess
import sys
import uuid
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from .upscale import resolve_upscale_model_name, validate_frames_for_reconstruction
from .video_sr import resolve_seedvr2_callables
from .rtx_vsr_stream import DeblurVsrFrameProcessor, VsrFrameProcessor, load_vsr_api
from .rife_stream import (
    DEFAULT_RIFE_MODEL,
    RifeProcessingError,
    iter_rife_frames,
    probe_rife_capability,
    smoothed_frame_count,
)


LOGGER = logging.getLogger("MiniMaxH3.DirectorPlus.StreamOutput")


class _VsrProcessingError(RuntimeError):
    """Distinguish VSR dependency/runtime failures from FFmpeg retry errors."""


_AUDIO_TARGET_PEAK = 10 ** (-1.5 / 20)
_AUDIO_MAX_GAIN = 10 ** (30 / 20)
_AUDIO_GATE_REDUCTION = 0.12
_AUDIO_GATE_RATIO = 1.8
_AUDIO_GATE_MIN_DYNAMIC_RANGE = 4.0
_SMART_UPSCALE_PROFILE = "smart_conservative_blend_v1"
_SMART_UPSCALE_AI_WEIGHT = 0.8
_TRAINED_TWO_STAGE_PRESETS = {
    "quality_two_stage",
    "质量优先二采样",
    "low_vram_two_stage",
    "低显存二采",
}


def _center_crop_to_target_aspect(frame, target_width, target_height):
    """Return an even-sized centered HWC view matching the target aspect ratio."""
    target_width = int(target_width)
    target_height = int(target_height)
    if target_width <= 0 or target_height <= 0:
        raise ValueError("目标宽高必须为正数")

    source_height = int(frame.shape[0])
    source_width = int(frame.shape[1])
    if (
        source_height < 2
        or source_width < 2
        or source_height % 2
        or source_width % 2
    ):
        raise ValueError("H3 双效果输入宽高必须为不小于 2 的偶数")
    if source_width * target_height == source_height * target_width:
        return frame
    if source_width * target_height > source_height * target_width:
        crop_width = min(source_width, (source_height * target_width) // target_height)
        crop_width -= crop_width % 2
        if crop_width <= 0:
            raise ValueError("源帧宽度不足以进行偶数比例裁切")
        left = (source_width - crop_width) // 2
        return frame[:, left:left + crop_width, :]

    crop_height = min(source_height, (source_width * target_height) // target_width)
    crop_height -= crop_height % 2
    if crop_height <= 0:
        raise ValueError("源帧高度不足以进行偶数比例裁切")
    top = (source_height - crop_height) // 2
    return frame[top:top + crop_height, :, :]


def _center_crop_batch_to_target_aspect(images, target_width, target_height):
    """Return a centered NHWC view matching the target aspect ratio."""
    if images.ndim != 4:
        raise ValueError("视频帧批次必须为 NHWC 四维张量")
    if len(images) == 0:
        return images
    sample = _center_crop_to_target_aspect(
        images[0], target_width, target_height
    )
    crop_height = int(sample.shape[0])
    crop_width = int(sample.shape[1])
    source_height = int(images.shape[1])
    source_width = int(images.shape[2])
    top = (source_height - crop_height) // 2
    left = (source_width - crop_width) // 2
    return images[:, top:top + crop_height, left:left + crop_width, :]


def _should_use_deblur_before_upscale(guide, postprocess_path):
    """Keep the retired deblur route disabled for all workflows.

    Older U11 files may still carry ``rtx_deblur_mode=DEBLUR_LOW``.  Returning
    false here is deliberate: the mode previously generated corrupted RGB
    frames on real NVIDIA servers, so stale metadata must not reactivate it.
    """
    return False


def _clean_output_audio(waveform):
    """Reduce stable low-energy noise and apply bounded peak normalization."""
    clean = torch.nan_to_num(
        waveform.detach().to(device="cpu", dtype=torch.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if not clean.numel():
        return clean
    peak = float(clean.abs().max().item())
    if peak <= 1e-8:
        return clean

    sample_count = int(clean.shape[-1]) if clean.ndim else 1
    if clean.ndim and sample_count >= 3:
        window = min(257, sample_count if sample_count % 2 else sample_count - 1)
        channels = clean.reshape(-1, 1, sample_count)
        padding = window // 2
        padded = F.pad(channels.square(), (padding, padding), mode="replicate")
        envelope = F.avg_pool1d(padded, kernel_size=window, stride=1).sqrt()
        positive_envelope = envelope[envelope > 1e-8]
        if positive_envelope.numel():
            noise_floor = float(torch.quantile(positive_envelope, 0.2).item())
            max_envelope = float(envelope.max().item())
            if (
                noise_floor > 1e-8
                and max_envelope / noise_floor >= _AUDIO_GATE_MIN_DYNAMIC_RANGE
            ):
                gate_ceiling = noise_floor * _AUDIO_GATE_RATIO
                blend = ((envelope - noise_floor) / max(1e-8, gate_ceiling - noise_floor))
                blend = blend.clamp(0.0, 1.0)
                gate = _AUDIO_GATE_REDUCTION + (1.0 - _AUDIO_GATE_REDUCTION) * blend
                clean = (channels * gate).reshape_as(clean)

    cleaned_peak = float(clean.abs().max().item())
    if cleaned_peak <= 1e-8:
        return clean
    gain = min(_AUDIO_TARGET_PEAK / cleaned_peak, _AUDIO_MAX_GAIN)
    return (clean * gain).clamp(-_AUDIO_TARGET_PEAK, _AUDIO_TARGET_PEAK)


def _normalize_output_audio(audio, mode):
    """Clean and normalize auto audio without modifying the source AUDIO."""
    mode = str(mode or "original")
    if audio is None or mode == "original":
        return audio
    if mode != "auto":
        raise ValueError(f"不支持的最终音频响度模式：{mode}")
    if not isinstance(audio, dict) or not isinstance(audio.get("waveform"), torch.Tensor):
        raise ValueError("自动音频响度需要标准 ComfyUI AUDIO 波形")

    waveform = audio["waveform"]
    if not torch.is_floating_point(waveform):
        raise ValueError("自动音频响度只支持浮点 AUDIO 波形")
    finite = bool(torch.isfinite(waveform).all().item())
    peak = float(waveform.detach().abs().max().item()) if waveform.numel() and finite else None
    if finite and peak <= 1e-8:
        return audio

    normalized = dict(audio)
    normalized["waveform"] = _clean_output_audio(waveform)
    return normalized


def _dasiwa_video_module():
    """Return DaSiWa's already-loaded encoder helper module."""
    for name, module in list(sys.modules.items()):
        if name.endswith("nodes_enhanced_video_combine"):
            return module
    for name in (
        "custom_nodes.ComfyUI-DaSiWa-Nodes.nodes.nodes_enhanced_video_combine",
        "ComfyUI-DaSiWa-Nodes.nodes.nodes_enhanced_video_combine",
    ):
        try:
            return importlib.import_module(name)
        except (ImportError, ModuleNotFoundError, ValueError):
            continue
    raise RuntimeError("缺少 DaSiWa Enhanced Video Combine 编码器")


def _resize_cpu_chunk(images, target_width, target_height, method="bicubic"):
    chunk = images.detach().to(device="cpu", dtype=torch.float32)
    if chunk.shape[1] == target_height and chunk.shape[2] == target_width:
        return chunk.clamp(0.0, 1.0)
    if method == "lanczos":
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        resized = []
        for frame in chunk:
            pixels = torch.round(frame[..., :3].clamp(0.0, 1.0) * 255).to(torch.uint8).numpy()
            image = Image.fromarray(pixels).resize(
                (int(target_width), int(target_height)), resampling
            )
            resized.append(torch.from_numpy(np.asarray(image).copy()).to(torch.float32) / 255.0)
        return torch.stack(resized, dim=0).clamp(0.0, 1.0)
    resized = F.interpolate(
        chunk.movedim(-1, 1),
        size=(target_height, target_width),
        mode="bicubic",
        align_corners=False,
    )
    return resized.movedim(1, -1).clamp(0.0, 1.0)


def _iter_resized_frame_chunks(
    images,
    target_width,
    target_height,
    max_chunk_bytes=64 * 1024 * 1024,
    bytes_per_channel=1,
    pingpong=False,
    method="bicubic",
):
    """Yield small target-size frame tensors; never allocate the full target batch."""
    target_frame_bytes = max(1, int(target_width) * int(target_height) * 3 * int(bytes_per_channel))
    frames_per_chunk = max(1, min(4, int(max_chunk_bytes) // target_frame_bytes))

    def resized(frame_chunk):
        return _resize_cpu_chunk(frame_chunk, int(target_width), int(target_height), method=method)

    for start in range(0, len(images), frames_per_chunk):
        yield resized(images[start:start + frames_per_chunk])
    if pingpong:
        for stop in range(len(images) - 1, 1, -frames_per_chunk):
            start = max(1, stop - frames_per_chunk)
            yield resized(images[start:stop].flip(0))


def _iter_lanczos_frame_chunks(
    images,
    target_width,
    target_height,
    max_chunk_bytes=64 * 1024 * 1024,
    bytes_per_channel=1,
    pingpong=False,
):
    yield from _iter_resized_frame_chunks(
        images,
        target_width,
        target_height,
        max_chunk_bytes=max_chunk_bytes,
        bytes_per_channel=bytes_per_channel,
        pingpong=pingpong,
        method="lanczos",
    )


def _iter_balanced_fhd_downscale_frame_chunks(
    images,
    target_width,
    target_height,
    max_chunk_bytes=64 * 1024 * 1024,
    bytes_per_channel=1,
    pingpong=False,
):
    """Aspect-crop and Lanczos-downscale FHD supersampled frames in small batches."""
    target_frame_bytes = max(
        1,
        int(target_width) * int(target_height) * 3 * int(bytes_per_channel),
    )
    frames_per_chunk = max(
        1, min(4, int(max_chunk_bytes) // target_frame_bytes)
    )

    def resized(frame_chunk):
        cropped = _center_crop_batch_to_target_aspect(
            frame_chunk, target_width, target_height
        )
        return _resize_cpu_chunk(
            cropped,
            int(target_width),
            int(target_height),
            method="lanczos",
        )

    for start in range(0, len(images), frames_per_chunk):
        yield resized(images[start:start + frames_per_chunk])
    if pingpong:
        for stop in range(len(images) - 1, 1, -frames_per_chunk):
            start = max(1, stop - frames_per_chunk)
            yield resized(images[start:stop].flip(0))


def _load_upscale_model(model_name):
    try:
        import comfy.model_management as model_management
        from comfy_extras.nodes_upscale_model import UpscaleModelLoader

        model_management.unload_all_models()
        loaded = UpscaleModelLoader().load_model(model_name)
        return loaded[0] if hasattr(loaded, "__getitem__") else loaded
    except Exception as exc:
        raise RuntimeError(f"通用 AI 超分模型加载失败：{model_name}：{exc}") from exc


def _upscale_image_with_model(model, image):
    try:
        from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel

        result = ImageUpscaleWithModel().upscale(model, image)
        return result[0] if hasattr(result, "__getitem__") else result
    except Exception as exc:
        raise RuntimeError(f"通用 AI 超分处理失败：{exc}") from exc


def _release_upscale_model(model):
    patcher = getattr(model, "patcher", None)
    if patcher is None:
        return
    try:
        import comfy.model_management as model_management

        model_management.unload_model_and_clones(patcher)
        model_management.soft_empty_cache()
    except (ImportError, AttributeError, RuntimeError):
        return


def _iter_ai_upscale_frame_chunks(
    images,
    target_width,
    target_height,
    model_name="auto",
    profile="standard",
    max_chunk_bytes=64 * 1024 * 1024,
    bytes_per_channel=1,
    pingpong=False,
):
    """Run ComfyUI's generic image upscaler one frame at a time."""
    scale_factor = max(
        float(target_width) / max(1, int(images.shape[2])),
        float(target_height) / max(1, int(images.shape[1])),
    )
    selected_model = resolve_upscale_model_name(model_name, scale_factor)
    target_frame_bytes = max(1, int(target_width) * int(target_height) * 3 * int(bytes_per_channel))
    frames_per_chunk = max(1, min(4, int(max_chunk_bytes) // target_frame_bytes))
    model = None

    def frame_indices():
        yield from range(len(images))
        if pingpong:
            for index in range(len(images) - 2, 0, -1):
                yield index

    try:
        model = _load_upscale_model(selected_model)
        indices = list(frame_indices())
        for start in range(0, len(indices), frames_per_chunk):
            batch_indices = indices[start:start + frames_per_chunk]
            frame_batch = images[batch_indices].detach().to(device="cpu", dtype=torch.float32)
            # ImageUpscaleWithModel performs its model-loading check per call.
            # Pass a bounded batch so long videos do not reload/check RRDBNet
            # once for every frame or flood the ComfyUI log.
            result = _upscale_image_with_model(model, frame_batch)
            result = _resize_cpu_chunk(result, int(target_width), int(target_height), method="lanczos")
            if profile == _SMART_UPSCALE_PROFILE:
                baseline = _resize_cpu_chunk(
                    frame_batch, int(target_width), int(target_height), method="lanczos"
                )
                result = (
                    result * _SMART_UPSCALE_AI_WEIGHT
                    + baseline * (1.0 - _SMART_UPSCALE_AI_WEIGHT)
                ).clamp(0.0, 1.0)
            yield result.contiguous()
    finally:
        if model is not None:
            _release_upscale_model(model)


def _unwrap_node_result(result, index=0):
    result = getattr(result, "result", result)
    if isinstance(result, (tuple, list)):
        return result[index]
    if index == 0:
        return result
    raise IndexError(f"节点结果没有第 {index} 个输出")


def _iter_video_sr_frame_chunks(
    images,
    target_width,
    target_height,
    seed=42,
    plan=None,
    max_chunk_frames=4,
):
    """Run SeedVR2 diffusion video SR once, then yield bounded CPU chunks.

    SeedVR2 batches and offloads internally, so the full clip is submitted in
    one call with a hardware-tiered plan; the upscaled tensor is yielded back
    in small CPU chunks to keep encoder feeding bounded.
    """
    callables = resolve_seedvr2_callables()
    if callables is None:
        raise RuntimeError(
            "缺少 SeedVR2 视频超分节点：请安装 ComfyUI-SeedVR2_VideoUpscaler，"
            "或在导演台把最终输出改为通用 AI 超分"
        )
    upscale_fn, dit_loader_fn, vae_loader_fn = callables
    plan = dict(plan or {})
    dit_config = _unwrap_node_result(
        dit_loader_fn(
            model=plan.get("dit_model"),
            device="cuda:0",
            offload_device=plan.get("dit_offload_device", "cpu"),
            blocks_to_swap=int(plan.get("blocks_to_swap", 0)),
            swap_io_components=bool(plan.get("swap_io_components", False)),
        )
    )
    vae_config = _unwrap_node_result(
        vae_loader_fn(
            model=plan.get("vae_model"),
            device="cuda:0",
            offload_device="cpu",
            encode_tiled=bool(plan.get("encode_tiled", False)),
            decode_tiled=bool(plan.get("decode_tiled", False)),
        )
    )
    result = _unwrap_node_result(
        upscale_fn(
            image=images.detach().to(device="cpu", dtype=torch.float32),
            dit=dit_config,
            vae=vae_config,
            seed=int(seed),
            resolution=int(min(target_width, target_height)),
            max_resolution=int(max(target_width, target_height)),
            batch_size=int(plan.get("batch_size", 5)),
            uniform_batch_size=False,
            temporal_overlap=int(plan.get("temporal_overlap", 0)),
            color_correction=str(plan.get("color_correction", "lab")),
            offload_device="cpu",
        )
    )
    if not isinstance(result, torch.Tensor) or result.ndim != 4:
        raise RuntimeError("SeedVR2 视频超分返回了无效结果")
    step = max(1, int(max_chunk_frames))
    for start in range(0, int(result.shape[0]), step):
        yield _resize_cpu_chunk(
            result[start:start + step], int(target_width), int(target_height),
            method="lanczos",
        ).clamp(0.0, 1.0)


def _save_video_sr_frame(source, index, target_width, target_height, seed, plan, output_path, suffix):
    single_plan = dict(plan or {})
    single_plan["batch_size"] = 1
    chunks = _iter_video_sr_frame_chunks(
        source[index:index + 1], target_width, target_height,
        seed=seed, plan=single_plan,
    )
    try:
        frame = next(chunks)[0]
    finally:
        close = getattr(chunks, "close", None)
        if callable(close):
            close()
    pixels = torch.round(frame[..., :3].clamp(0.0, 1.0) * 255).to(torch.uint8).numpy()
    path = f"{os.path.splitext(output_path)[0]}-{suffix}-frame.png"
    Image.fromarray(pixels, mode="RGB").save(path, "PNG")
    return path


def _iter_native_frame_chunks(images, max_frames=4, pingpong=False):
    """Yield source-size CPU chunks without duplicating the complete batch."""
    max_frames = max(1, min(4, int(max_frames)))

    def emit(start, stop):
        return images[start:stop].detach().to(device="cpu", dtype=torch.float32).contiguous()

    for start in range(0, len(images), max_frames):
        yield emit(start, min(len(images), start + max_frames))
    if pingpong:
        for stop in range(len(images) - 1, 1, -max_frames):
            start = max(1, stop - max_frames)
            yield emit(start, stop).flip(0)


def _iter_vsr_frame_chunks(
    images,
    target_width,
    target_height,
    quality="HIGH",
    max_chunk_bytes=64 * 1024 * 1024,
    bytes_per_channel=1,
    pingpong=False,
    device_id=0,
    deblur_before_upscale=False,
):
    """Run RTX VSR one frame at a time and yield bounded CPU HWC chunks."""
    def frame_stream():
        yield from (images[index] for index in range(len(images)))
        if pingpong:
            yield from (images[index] for index in range(len(images) - 2, 0, -1))

    yield from _iter_vsr_frame_stream(
        frame_stream(),
        target_width,
        target_height,
        quality=quality,
        max_chunk_bytes=max_chunk_bytes,
        bytes_per_channel=bytes_per_channel,
        device_id=device_id,
        deblur_before_upscale=deblur_before_upscale,
    )


def _iter_vsr_frame_stream(
    frames,
    target_width,
    target_height,
    quality="HIGH",
    max_chunk_bytes=64 * 1024 * 1024,
    bytes_per_channel=1,
    device_id=0,
    deblur_before_upscale=False,
):
    """Apply one VSR processor to an arbitrary lazy HWC frame stream."""
    target_frame_bytes = max(1, int(target_width) * int(target_height) * 3 * int(bytes_per_channel))
    frames_per_chunk = max(1, min(4, int(max_chunk_bytes) // target_frame_bytes))
    processor = None

    pending = []
    try:
        try:
            api = load_vsr_api()
        except Exception as exc:
            raise _VsrProcessingError(str(exc)) from exc
        for frame in frames:
            if deblur_before_upscale:
                try:
                    processor_frame = _center_crop_to_target_aspect(
                        frame, target_width, target_height
                    )
                except Exception as exc:
                    raise _VsrProcessingError(str(exc)) from exc
            else:
                processor_frame = frame
            if processor is None:
                try:
                    if deblur_before_upscale:
                        processor = DeblurVsrFrameProcessor(
                            api,
                            quality,
                            int(device_id),
                            int(processor_frame.shape[1]),
                            int(processor_frame.shape[0]),
                            int(target_width),
                            int(target_height),
                        )
                    else:
                        processor = VsrFrameProcessor(
                            api, quality, int(device_id), int(target_width), int(target_height)
                        )
                except Exception as exc:
                    raise _VsrProcessingError(str(exc)) from exc
            # The processor owns the CUDA conversion; keep only one source
            # frame and a small CPU output chunk alive at a time.
            chw_frame = processor_frame[..., :3].detach().movedim(-1, 0).contiguous()
            try:
                pending.append(processor.process(chw_frame).detach().to(device="cpu"))
            except Exception as exc:
                raise _VsrProcessingError(str(exc)) from exc
            if len(pending) >= frames_per_chunk:
                yield torch.stack(pending, dim=0).contiguous()
                pending.clear()
        if pending:
            yield torch.stack(pending, dim=0).contiguous()
    finally:
        active_error = sys.exc_info()[1]
        upstream_error = None
        close_frames = getattr(frames, "close", None)
        if callable(close_frames):
            try:
                close_frames()
            except Exception as exc:
                upstream_error = exc
        if processor is not None:
            try:
                processor.close()
            except Exception as exc:
                if active_error is None or isinstance(active_error, GeneratorExit):
                    raise _VsrProcessingError(str(exc)) from exc
        if upstream_error is not None and (
            active_error is None or isinstance(active_error, GeneratorExit)
        ):
            raise upstream_error


def _resolve_postprocess_path(guide, source_width, source_height):
    """Resolve explicit guide paths while retaining legacy guide behavior."""
    path = guide.get("postprocess_path")
    if path in {
        "native_bypass",
        "downscale",
        "balanced_fhd_downscale",
        "lanczos",
        "ai_upscale",
        "video_sr",
        "rtx_vsr",
    }:
        # Equal native/target dimensions are always a bypass, even if an old
        # guide requested RTX VSR as a mode rather than a resolved path.
        target_width = int(guide.get("target_width") or source_width)
        target_height = int(guide.get("target_height") or source_height)
        if target_width == int(source_width) and target_height == int(source_height):
            return "native_bypass"
        if path == "balanced_fhd_downscale":
            return path
        if target_width < int(source_width) or target_height < int(source_height):
            return "downscale"
        return path

    target_width = int(guide.get("target_width") or source_width)
    target_height = int(guide.get("target_height") or source_height)
    if target_width == int(source_width) and target_height == int(source_height):
        return "native_bypass"
    # Legacy workflows used upscale_required plus CPU bicubic for all size
    # changes. Keep that behavior under the downscale iterator name.
    return "downscale"


def release_sampling_models():
    """Release H3 patchers before constructing the final AI model."""
    import comfy.model_management as model_management

    model_management.unload_all_models()
    model_management.soft_empty_cache()
    LOGGER.info("[H3 output] H3 sampling models released before final AI reconstruction")


def _prepare_postprocess_runtime(guide, postprocess_path):
    preset = str((guide or {}).get("performance_preset") or "")
    if preset in _TRAINED_TWO_STAGE_PRESETS and postprocess_path in {"rtx_vsr", "ai_upscale", "video_sr"}:
        release_sampling_models()


def _resolve_upscale_profile(guide, postprocess_path):
    """Resolve the conservative profile for smart AI reconstruction only."""
    if postprocess_path != "ai_upscale":
        return "standard"
    requested = str((guide or {}).get("requested_performance_preset") or "")
    explicit = str((guide or {}).get("upscale_profile") or "")
    if explicit == _SMART_UPSCALE_PROFILE or requested == "smart_free_1080p":
        return _SMART_UPSCALE_PROFILE
    return "standard"


def _resolved_encode_quality(guide, postprocess_path, requested_quality):
    """Use a visually lossless CRF ceiling for high-quality reconstruction routes."""
    quality = int(requested_quality)
    preset = str((guide or {}).get("performance_preset") or "")
    requested_preset = str((guide or {}).get("requested_performance_preset") or "")
    if preset in _TRAINED_TWO_STAGE_PRESETS or requested_preset == "smart_free_1080p":
        return min(quality, 16)
    return quality


def _tag_h264_bt709(ffmpeg, output_path):
    """Write BT.709 limited-range VUI by stream-copy without re-encoding media."""
    output = Path(output_path)
    temporary = output.with_name(
        f"{output.stem}.bt709-{uuid.uuid4().hex}{output.suffix}"
    )
    metadata_filter = (
        "h264_metadata=video_full_range_flag=0:colour_primaries=1:"
        "transfer_characteristics=1:matrix_coefficients=1"
    )
    command = [
        str(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(output),
        "-map",
        "0",
        "-map_metadata",
        "0",
        "-c",
        "copy",
        "-bsf:v",
        metadata_filter,
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
        os.replace(temporary, output)
        LOGGER.info("[H3 output] H.264 已无损写入 BT.709 limited-range 标记")
        return True
    except Exception as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        detail = str(getattr(exc, "stderr", "") or exc).strip()
        LOGGER.warning(
            "[H3 output] BT.709 无损标记失败，已保留原编码视频：%s",
            detail,
        )
        return False


def _save_resized_frame(source, index, target_width, target_height, output_path, suffix):
    frame = _resize_cpu_chunk(source[index:index + 1], target_width, target_height)[0]
    pixels = torch.round(frame[..., :3] * 255).to(torch.uint8).numpy()
    path = f"{os.path.splitext(output_path)[0]}-{suffix}-frame.png"
    Image.fromarray(pixels, mode="RGB").save(path, "PNG")
    return path


def _save_lanczos_frame(source, index, target_width, target_height, output_path, suffix):
    frame = _resize_cpu_chunk(
        source[index:index + 1], target_width, target_height, method="lanczos"
    )[0]
    pixels = torch.round(frame[..., :3] * 255).to(torch.uint8).numpy()
    path = f"{os.path.splitext(output_path)[0]}-{suffix}-frame.png"
    Image.fromarray(pixels, mode="RGB").save(path, "PNG")
    return path


def _save_balanced_fhd_downscale_frame(
    source, index, target_width, target_height, output_path, suffix
):
    chunks = _iter_balanced_fhd_downscale_frame_chunks(
        source[index:index + 1], target_width, target_height
    )
    try:
        frame = next(chunks)[0]
    finally:
        close = getattr(chunks, "close", None)
        if callable(close):
            close()
    pixels = torch.round(frame[..., :3] * 255).to(torch.uint8).numpy()
    path = f"{os.path.splitext(output_path)[0]}-{suffix}-frame.png"
    Image.fromarray(pixels, mode="RGB").save(path, "PNG")
    return path


def _save_ai_upscale_frame(
    source, index, target_width, target_height, model_name, output_path, suffix,
    profile="standard",
):
    chunks = _iter_ai_upscale_frame_chunks(
        source[index:index + 1], target_width, target_height,
        model_name=model_name, profile=profile,
    )
    try:
        frame = next(chunks)[0]
    finally:
        close = getattr(chunks, "close", None)
        if callable(close):
            close()
    pixels = torch.round(frame[..., :3].clamp(0.0, 1.0) * 255).to(torch.uint8).numpy()
    path = f"{os.path.splitext(output_path)[0]}-{suffix}-frame.png"
    Image.fromarray(pixels, mode="RGB").save(path, "PNG")
    return path


def _save_native_frame(source, index, output_path, suffix):
    frame = source[index].detach().to(device="cpu", dtype=torch.float32).clamp(0.0, 1.0)
    pixels = torch.round(frame[..., :3] * 255).to(torch.uint8).numpy()
    path = f"{os.path.splitext(output_path)[0]}-{suffix}-frame.png"
    Image.fromarray(pixels, mode="RGB").save(path, "PNG")
    return path


def _save_vsr_frame(
    source,
    index,
    target_width,
    target_height,
    quality,
    device_id,
    output_path,
    suffix,
    deblur_before_upscale=False,
):
    """Run one exported frame through the same RTX VSR path as the video."""
    processor_frame = source[index]
    if deblur_before_upscale:
        processor_frame = _center_crop_to_target_aspect(
            processor_frame, target_width, target_height
        )
    api = load_vsr_api()
    if deblur_before_upscale:
        processor = DeblurVsrFrameProcessor(
            api,
            quality,
            int(device_id),
            int(processor_frame.shape[1]),
            int(processor_frame.shape[0]),
            int(target_width),
            int(target_height),
        )
    else:
        processor = VsrFrameProcessor(
            api, quality, int(device_id), int(target_width), int(target_height)
        )
    try:
        try:
            frame = processor.process(
                processor_frame[..., :3].detach().movedim(-1, 0).contiguous()
            )
        except Exception as exc:
            raise _VsrProcessingError(str(exc)) from exc
    finally:
        active_error = sys.exc_info()[1]
        try:
            processor.close()
        except Exception as exc:
            if active_error is None:
                raise _VsrProcessingError(str(exc)) from exc
            LOGGER.warning("RTX VSR 导出帧次级清理失败：%s", exc)
    pixels = torch.round(frame[..., :3].clamp(0.0, 1.0) * 255).to(torch.uint8).numpy()
    path = f"{os.path.splitext(output_path)[0]}-{suffix}-frame.png"
    Image.fromarray(pixels, mode="RGB").save(path, "PNG")
    return path


class MiniMaxH3StreamingVideoCombine:
    """DaSiWa-compatible output node with optional streaming final-size resize."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"description": "H3 解码视频帧。"}),
                "guide": ("MINIMAX_H3_DIRECTOR_PLUS_GUIDE",),
                "frame_rate": ("FLOAT", {"default": 24.0, "min": 0.1, "max": 240.0, "step": 0.01}),
                "codec": (["Auto", "AV1", "VP9", "H.265 (HEVC)", "H.264"], {"default": "Auto"}),
                "container": (["Auto", "WebM", "MKV", "MP4", "Animated WebP", "Animated AVIF"], {"default": "Auto"}),
                "bit_depth": (["Auto", "8-bit", "10-bit"], {"default": "Auto"}),
                "quality": ("INT", {"default": 20, "min": 0, "max": 51}),
                "log_level": (["Standard", "Verbose"], {"default": "Standard"}),
                "pingpong": ("BOOLEAN", {"default": False}),
                "save_metadata": ("BOOLEAN", {"default": True}),
                "filename_prefix": ("STRING", {"default": "video_%date:hhmmss%"}),
                "save_output": ("BOOLEAN", {"default": True}),
                "pass_frames": ("BOOLEAN", {"default": False}),
                "crop_to_audio": ("BOOLEAN", {"default": False}),
                "audio_codec": (["Auto", "AAC", "Opus", "MP3"], {"default": "Auto"}),
                "audio_bitrate": (["64k", "96k", "128k", "160k", "192k", "256k", "320k"], {"default": "192k"}),
                "save_first_frame": ("BOOLEAN", {"default": False}),
                "save_last_frame": ("BOOLEAN", {"default": False}),
            },
            "optional": {"audio": ("AUDIO",)},
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("frames", "filename")
    FUNCTION = "combine"
    OUTPUT_NODE = True
    CATEGORY = "MiniMax H3 导演台 Plus"

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def combine(
        self,
        images,
        guide,
        frame_rate,
        codec,
        container,
        bit_depth,
        quality,
        log_level,
        pingpong,
        save_metadata,
        filename_prefix,
        save_output,
        pass_frames,
        crop_to_audio,
        audio_codec,
        audio_bitrate,
        save_first_frame,
        save_last_frame,
        audio=None,
        prompt=None,
        extra_pnginfo=None,
    ):
        if images.ndim != 4 or images.shape[-1] < 3:
            raise ValueError("images 必须是 [帧数, 高, 宽, 通道] 的 IMAGE 批次")

        dasiwa = _dasiwa_video_module()
        source = images.detach()
        source_width = int(source.shape[2])
        source_height = int(source.shape[1])
        target_width = int(guide.get("target_width") or source.shape[2])
        target_height = int(guide.get("target_height") or source.shape[1])
        postprocess_path = _resolve_postprocess_path(guide, source_width, source_height)
        upscale_profile = _resolve_upscale_profile(guide, postprocess_path)
        encode_quality = _resolved_encode_quality(guide, postprocess_path, quality)
        motion_smoothing = str(guide.get("motion_smoothing") or "off")
        if motion_smoothing not in {"off", "rife_x2"}:
            raise ValueError(f"不支持的运动平滑路径：{motion_smoothing}")
        performance_preset = str(guide.get("performance_preset") or "")
        deblur_before_upscale = _should_use_deblur_before_upscale(
            guide, postprocess_path
        )
        if motion_smoothing == "rife_x2" and performance_preset in _TRAINED_TWO_STAGE_PRESETS:
            preset_label = (
                "低显存二采"
                if performance_preset in {"low_vram_two_stage", "低显存二采"}
                else "质量优先二采样"
            )
            raise ValueError(f"{preset_label}不兼容 RIFE 运动平滑，必须保留原始 24 FPS 以避免重影")
        if motion_smoothing == "rife_x2" and postprocess_path != "rtx_vsr":
            raise ValueError("RIFE 2x 运动平滑只能在 RTX VSR 输出链路中执行")
        # Fail before creating an output path/encoder when the strict VSR
        # dependency is unavailable. The frame iterator is still responsible
        # for creating and closing the per-attempt processor.
        if postprocess_path == "rtx_vsr":
            try:
                load_vsr_api()
            except Exception as exc:
                raise _VsrProcessingError(str(exc)) from exc
        if motion_smoothing == "rife_x2":
            probe_rife_capability(guide.get("rife_model", DEFAULT_RIFE_MODEL))
        if postprocess_path == "video_sr":
            validate_frames_for_reconstruction(source)
            if resolve_seedvr2_callables() is None:
                raise RuntimeError(
                    "缺少 SeedVR2 视频超分节点：请安装 ComfyUI-SeedVR2_VideoUpscaler，"
                    "或在导演台把最终输出改为通用 AI 超分"
                )
        if postprocess_path == "ai_upscale":
            validate_frames_for_reconstruction(source)
            try:
                resolve_upscale_model_name(
                    guide.get("ai_upscale_model", "auto"),
                    max(
                        float(target_width) / max(1, source_width),
                        float(target_height) / max(1, source_height),
                    ),
                )
            except Exception as exc:
                raise RuntimeError(f"通用 AI 超分前置检查失败：{exc}") from exc
        _prepare_postprocess_runtime(guide, postprocess_path)
        if postprocess_path == "downscale":
            source = source.to(device="cpu", dtype=torch.float32)
        selected_bit_depth = dasiwa._selected_bit_depth(codec, bit_depth, source)
        output_dir = dasiwa.folder_paths.get_output_directory() if save_output else dasiwa.folder_paths.get_temp_directory()
        output_type = "output" if save_output else "temp"
        filename_prefix = dasiwa._format_filename_prefix(filename_prefix)
        output_folder, filename, counter, subfolder, _ = dasiwa.folder_paths.get_save_image_path(
            filename_prefix, output_dir, target_width, target_height,
        )
        ffmpeg = dasiwa.find_ffmpeg()
        if not ffmpeg:
            raise RuntimeError("未找到 FFmpeg，无法保存 MP4")

        source_frame_count = dasiwa._encoded_frame_count(source, pingpong)
        frame_multiplier = 2 if motion_smoothing == "rife_x2" else 1
        total_frames = smoothed_frame_count(source_frame_count, frame_multiplier)
        output_frame_rate = float(frame_rate) * frame_multiplier
        try:
            import comfy.utils

            progress_bar = comfy.utils.ProgressBar(total_frames)
        except ImportError:
            progress_bar = None

        def report_encode_progress(encoded_seconds):
            if progress_bar is not None:
                progress_bar.update_absolute(min(total_frames, max(0, int(encoded_seconds * output_frame_rate))))

        ai_edge_frames = {}

        def frame_chunks():
            chunk_kwargs = {
                "max_chunk_bytes": dasiwa._MAX_RAW_FRAME_CHUNK_BYTES,
                "bytes_per_channel": 2 if selected_bit_depth == 10 else 1,
                "pingpong": pingpong,
            }
            if postprocess_path == "native_bypass":
                chunks = _iter_native_frame_chunks(source, max_frames=4, pingpong=pingpong)
            elif postprocess_path == "downscale":
                chunks = _iter_resized_frame_chunks(
                    source, target_width, target_height, **chunk_kwargs
                )
            elif postprocess_path == "balanced_fhd_downscale":
                chunks = _iter_balanced_fhd_downscale_frame_chunks(
                    source, target_width, target_height, **chunk_kwargs
                )
            elif postprocess_path == "lanczos":
                chunks = _iter_lanczos_frame_chunks(
                    source, target_width, target_height, **chunk_kwargs
                )
            elif postprocess_path == "video_sr":
                chunks = _iter_video_sr_frame_chunks(
                    source,
                    target_width,
                    target_height,
                    seed=int(guide.get("seed", 42) or 42),
                    plan=guide.get("video_sr_plan"),
                )
            elif postprocess_path == "ai_upscale":
                chunks = _iter_ai_upscale_frame_chunks(
                    source,
                    target_width,
                    target_height,
                    model_name=guide.get("ai_upscale_model", "auto"),
                    profile=upscale_profile,
                    **chunk_kwargs,
                )
            else:
                if motion_smoothing == "rife_x2":
                    rife_frames = iter_rife_frames(
                        source,
                        model_name=guide.get("rife_model", DEFAULT_RIFE_MODEL),
                        pingpong=pingpong,
                    )
                    chunks = _iter_vsr_frame_stream(
                        rife_frames,
                        target_width,
                        target_height,
                        quality=guide.get("rtx_quality", "HIGH"),
                        device_id=guide.get("device_id", 0),
                        max_chunk_bytes=chunk_kwargs["max_chunk_bytes"],
                        bytes_per_channel=chunk_kwargs["bytes_per_channel"],
                        deblur_before_upscale=False,
                    )
                else:
                    chunks = _iter_vsr_frame_chunks(
                        source,
                        target_width,
                        target_height,
                        quality=guide.get("rtx_quality", "HIGH"),
                        device_id=guide.get("device_id", 0),
                        deblur_before_upscale=deblur_before_upscale,
                        **chunk_kwargs,
                    )
            try:
                for chunk in chunks:
                    if postprocess_path in {"ai_upscale", "video_sr"} and len(chunk):
                        if "first" not in ai_edge_frames:
                            ai_edge_frames["first"] = chunk[0].detach().cpu().clone()
                        ai_edge_frames["last"] = chunk[-1].detach().cpu().clone()
                    yield dasiwa._frame_bytes(chunk, selected_bit_depth)
            finally:
                close = getattr(chunks, "close", None)
                if callable(close):
                    close()

        metadata_path = dasiwa._metadata_file(prompt, extra_pnginfo) if save_metadata else None
        audio_loudness = str(guide.get("audio_loudness", "original") or "original")
        audio_cleanup = "disabled"
        audio_cleanup_reason = "original_mode"
        if audio_loudness == "auto":
            try:
                output_audio = _normalize_output_audio(audio, audio_loudness)
                audio_cleanup = "auto_gate_peak_limit"
                audio_cleanup_reason = "applied"
            except ValueError:
                raise
            except Exception as exc:
                LOGGER.warning("[H3 audio] cleanup failed; using original audio: %s", exc)
                output_audio = audio
                audio_cleanup = "bypass_error"
                audio_cleanup_reason = str(exc)
        else:
            output_audio = _normalize_output_audio(audio, audio_loudness)
        audio_path, audio_duration = dasiwa._audio_file(output_audio)
        attempts = []
        output_path = None
        selected_container = None
        selected_codec = None
        encoder = None
        try:
            animated_settings = dasiwa._animated_image_settings(container)
            if animated_settings:
                output_path = os.path.join(output_folder, dasiwa._output_filename(filename, counter, animated_settings[0], False))
                encoder = dasiwa._encode_animated_image(
                    ffmpeg, container, selected_bit_depth, target_width, target_height,
                    output_frame_rate, frame_chunks, output_path, encode_quality, report_encode_progress,
                )
                selected_container = container
                selected_codec = container
            else:
                for selected_codec in dasiwa._codec_candidates(codec):
                    candidates = dasiwa._auto_container_candidates(selected_codec, container) if codec == "Auto" else dasiwa._container_candidates(selected_codec, container)
                    for selected_container in candidates:
                        output_path = os.path.join(
                            output_folder,
                            dasiwa._output_filename(filename, counter, dasiwa._CONTAINER_EXTENSIONS[selected_container], audio_path is not None),
                        )
                        try:
                            encoder = dasiwa._encode_with_available_encoder(
                                ffmpeg, selected_codec, selected_bit_depth, target_width, target_height,
                                output_frame_rate, frame_chunks, output_path, selected_container, encode_quality, encode_quality,
                                metadata_path, audio_path, audio_duration, crop_to_audio,
                                audio_codec, audio_bitrate, report_encode_progress,
                            )
                            break
                        except RuntimeError as error:
                            attempts.append(f"{selected_codec}/{selected_container}: {error}")
                            if isinstance(error, (_VsrProcessingError, RifeProcessingError)):
                                raise
                            if codec != "Auto":
                                raise
                    else:
                        continue
                    break
                else:
                    selected_codec = "H.264"
                    selected_container = "MP4"
                    output_path = os.path.join(
                        output_folder,
                        dasiwa._output_filename(filename, counter, ".mp4", audio_path is not None),
                    )
                    encoder = dasiwa._encode_with_available_encoder(
                        ffmpeg, selected_codec, selected_bit_depth, target_width, target_height,
                        output_frame_rate, frame_chunks, output_path, selected_container, encode_quality, encode_quality,
                        metadata_path, audio_path, audio_duration, crop_to_audio,
                        audio_codec, audio_bitrate, report_encode_progress,
                    )
        finally:
            if metadata_path:
                os.unlink(metadata_path)
            if audio_path:
                os.unlink(audio_path[0])

        bt709_tagged = False
        if (
            output_path
            and selected_codec == "H.264"
            and selected_container == "MP4"
            and performance_preset in _TRAINED_TWO_STAGE_PRESETS
        ):
            bt709_tagged = _tag_h264_bt709(ffmpeg, output_path)

        frame_exports = []
        if output_path:
            if save_first_frame:
                if postprocess_path == "native_bypass":
                    frame_exports.append(_save_native_frame(source, 0, output_path, "first"))
                elif postprocess_path == "downscale":
                    frame_exports.append(_save_resized_frame(source, 0, target_width, target_height, output_path, "first"))
                elif postprocess_path == "balanced_fhd_downscale":
                    frame_exports.append(_save_balanced_fhd_downscale_frame(source, 0, target_width, target_height, output_path, "first"))
                elif postprocess_path == "lanczos":
                    frame_exports.append(_save_lanczos_frame(source, 0, target_width, target_height, output_path, "first"))
                elif postprocess_path == "video_sr":
                    if "first" in ai_edge_frames:
                        frame_exports.append(_save_native_frame(ai_edge_frames["first"].unsqueeze(0), 0, output_path, "first"))
                    else:
                        frame_exports.append(_save_video_sr_frame(source, 0, target_width, target_height, guide.get("seed", 42), guide.get("video_sr_plan"), output_path, "first"))
                elif postprocess_path == "ai_upscale":
                    if "first" in ai_edge_frames:
                        frame_exports.append(_save_native_frame(ai_edge_frames["first"].unsqueeze(0), 0, output_path, "first"))
                    else:
                        frame_exports.append(_save_ai_upscale_frame(source, 0, target_width, target_height, guide.get("ai_upscale_model", "auto"), output_path, "first", profile=upscale_profile))
                else:
                    frame_exports.append(_save_vsr_frame(
                        source, 0, target_width, target_height,
                        guide.get("rtx_quality", "HIGH"), guide.get("device_id", 0),
                        output_path, "first", deblur_before_upscale=deblur_before_upscale,
                    ))
            if save_last_frame:
                last_index = 1 if pingpong and len(source) >= 3 else len(source) - 1
                if postprocess_path == "native_bypass":
                    frame_exports.append(_save_native_frame(source, last_index, output_path, "last"))
                elif postprocess_path == "downscale":
                    frame_exports.append(_save_resized_frame(source, last_index, target_width, target_height, output_path, "last"))
                elif postprocess_path == "balanced_fhd_downscale":
                    frame_exports.append(_save_balanced_fhd_downscale_frame(source, last_index, target_width, target_height, output_path, "last"))
                elif postprocess_path == "lanczos":
                    frame_exports.append(_save_lanczos_frame(source, last_index, target_width, target_height, output_path, "last"))
                elif postprocess_path == "video_sr":
                    if "last" in ai_edge_frames:
                        frame_exports.append(_save_native_frame(ai_edge_frames["last"].unsqueeze(0), 0, output_path, "last"))
                    else:
                        frame_exports.append(_save_video_sr_frame(source, last_index, target_width, target_height, guide.get("seed", 42), guide.get("video_sr_plan"), output_path, "last"))
                elif postprocess_path == "ai_upscale":
                    if "last" in ai_edge_frames:
                        frame_exports.append(_save_native_frame(ai_edge_frames["last"].unsqueeze(0), 0, output_path, "last"))
                    else:
                        frame_exports.append(_save_ai_upscale_frame(source, last_index, target_width, target_height, guide.get("ai_upscale_model", "auto"), output_path, "last", profile=upscale_profile))
                else:
                    frame_exports.append(_save_vsr_frame(
                        source, last_index, target_width, target_height,
                        guide.get("rtx_quality", "HIGH"), guide.get("device_id", 0),
                        output_path, "last", deblur_before_upscale=deblur_before_upscale,
                    ))

        if pass_frames and postprocess_path == "native_bypass":
            output_frames = source
        else:
            output_frames = torch.empty(
                (0, target_height, target_width, 3),
                dtype=torch.float32,
                device="cpu",
            )
        mime_types = {
            "WebM": "video/webm", "MKV": "video/x-matroska", "MP4": "video/mp4",
            **{name: settings[2] for name, settings in dasiwa._ANIMATED_IMAGE_SETTINGS.items()},
        }
        output_mime_type = mime_types[selected_container]
        assets = [{
            "filename": os.path.basename(output_path), "subfolder": subfolder, "type": output_type,
            "format": output_mime_type, "width": target_width, "height": target_height,
            "codec": selected_codec, "bit_depth": selected_bit_depth, "container": selected_container,
            "postprocess_path": postprocess_path, "encode_quality": encode_quality,
            "upscale_profile": upscale_profile,
            "color_metadata": "bt709" if bt709_tagged else "unchanged",
        }]
        assets.extend({
            "filename": os.path.basename(path), "subfolder": subfolder, "type": output_type,
            "format": "image/png", "width": target_width, "height": target_height,
            "postprocess_path": postprocess_path,
        } for path in frame_exports)
        ui = {
            "images": assets,
            "postprocess_path": postprocess_path,
            "audio_cleanup": audio_cleanup,
            "audio_cleanup_reason": audio_cleanup_reason,
        }
        if not dasiwa._animated_image_settings(selected_container):
            ui["gifs"] = [{
                "filename": os.path.basename(output_path), "subfolder": subfolder, "type": output_type,
                "format": output_mime_type, "codec": selected_codec, "bit_depth": selected_bit_depth,
                "container": selected_container, "width": target_width, "height": target_height, "fps": output_frame_rate,
                "source_fps": float(frame_rate), "motion_smoothing": motion_smoothing,
                 "audio_loudness": audio_loudness,
                 "audio_cleanup": audio_cleanup,
                 "audio_cleanup_reason": audio_cleanup_reason,
                 "audio_sample_rate": audio_path[1] if audio_path else None,
                 "audio_channels": audio_path[2] if audio_path else None,
                 "audio_bitrate": str(audio_bitrate) if audio_path else None,
                 "postprocess_path": postprocess_path, "encode_quality": encode_quality,
                 "upscale_profile": upscale_profile,
                 "color_metadata": "bt709" if bt709_tagged else "unchanged",
             }]
        LOGGER.info(
            "[H3 output] source=%sx%s final=%sx%s frames=%s fps=%.3f video=%s/%s "
            "quality=%s bt709=%s upscale_profile=%s audio=%s cleanup=%s path=%s",
            source_width,
            source_height,
            target_width,
            target_height,
            total_frames,
            output_frame_rate,
            selected_codec,
            selected_container,
            encode_quality,
            bt709_tagged,
            upscale_profile,
            audio_codec if audio_path is not None else "none",
            audio_cleanup,
            output_path,
        )
        return {"ui": ui, "result": (output_frames, output_path)}
