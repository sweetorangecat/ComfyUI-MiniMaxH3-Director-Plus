"""Stream final-size H3 frames to FFmpeg without keeping a full 4K batch."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from .rtx_vsr_stream import VsrFrameProcessor, load_vsr_api


class _VsrProcessingError(RuntimeError):
    """Distinguish VSR dependency/runtime failures from FFmpeg retry errors."""


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


def _resize_cpu_chunk(images, target_width, target_height):
    chunk = images.detach().to(device="cpu", dtype=torch.float32)
    if chunk.shape[1] == target_height and chunk.shape[2] == target_width:
        return chunk.clamp(0.0, 1.0)
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
):
    """Yield small target-size frame tensors; never allocate the full target batch."""
    target_frame_bytes = max(1, int(target_width) * int(target_height) * 3 * int(bytes_per_channel))
    frames_per_chunk = max(1, min(4, int(max_chunk_bytes) // target_frame_bytes))

    def resized(frame_chunk):
        return _resize_cpu_chunk(frame_chunk, int(target_width), int(target_height))

    for start in range(0, len(images), frames_per_chunk):
        yield resized(images[start:start + frames_per_chunk])
    if pingpong:
        for stop in range(len(images) - 1, 1, -frames_per_chunk):
            start = max(1, stop - frames_per_chunk)
            yield resized(images[start:stop].flip(0))


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
):
    """Run RTX VSR one frame at a time and yield bounded CPU HWC chunks."""
    target_frame_bytes = max(1, int(target_width) * int(target_height) * 3 * int(bytes_per_channel))
    frames_per_chunk = max(1, min(4, int(max_chunk_bytes) // target_frame_bytes))
    processor = None

    def frame_indices():
        yield from range(len(images))
        if pingpong:
            for index in range(len(images) - 2, 0, -1):
                yield index

    pending = []
    try:
        try:
            api = load_vsr_api()
            processor = VsrFrameProcessor(
                api, quality, device_id, int(target_width), int(target_height)
            )
        except RuntimeError as exc:
            raise _VsrProcessingError(str(exc)) from exc
        for index in frame_indices():
            # The processor owns the CUDA conversion; keep only one source
            # frame and a small CPU output chunk alive at a time.
            chw_frame = images[index, ..., :3].detach().movedim(-1, 0).contiguous()
            try:
                pending.append(processor.process(chw_frame))
            except RuntimeError as exc:
                raise _VsrProcessingError(str(exc)) from exc
            if len(pending) >= frames_per_chunk:
                yield torch.stack(pending, dim=0).contiguous()
                pending.clear()
        if pending:
            yield torch.stack(pending, dim=0).contiguous()
    finally:
        if processor is not None:
            processor.close()


def _resolve_postprocess_path(guide, source_width, source_height):
    """Resolve explicit guide paths while retaining legacy guide behavior."""
    path = guide.get("postprocess_path")
    if path in {"native_bypass", "downscale", "rtx_vsr"}:
        # Equal native/target dimensions are always a bypass, even if an old
        # guide requested RTX VSR as a mode rather than a resolved path.
        native_width = int(guide.get("native_width") or source_width)
        native_height = int(guide.get("native_height") or source_height)
        target_width = int(guide.get("target_width") or source_width)
        target_height = int(guide.get("target_height") or source_height)
        if target_width == native_width and target_height == native_height:
            return "native_bypass"
        return path

    target_width = int(guide.get("target_width") or source_width)
    target_height = int(guide.get("target_height") or source_height)
    if target_width == int(source_width) and target_height == int(source_height):
        return "native_bypass"
    # Legacy workflows used upscale_required plus CPU bicubic for all size
    # changes. Keep that behavior under the downscale iterator name.
    return "downscale"


def _save_resized_frame(source, index, target_width, target_height, output_path, suffix):
    frame = _resize_cpu_chunk(source[index:index + 1], target_width, target_height)[0]
    pixels = torch.round(frame[..., :3] * 255).to(torch.uint8).numpy()
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
):
    """Run one exported frame through the same RTX VSR path as the video."""
    processor = VsrFrameProcessor(
        load_vsr_api(), quality, int(device_id), int(target_width), int(target_height)
    )
    try:
        frame = processor.process(
            source[index, ..., :3].detach().movedim(-1, 0).contiguous()
        )
    finally:
        processor.close()
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

        total_frames = dasiwa._encoded_frame_count(source, pingpong)
        try:
            import comfy.utils

            progress_bar = comfy.utils.ProgressBar(total_frames)
        except ImportError:
            progress_bar = None

        def report_encode_progress(encoded_seconds):
            if progress_bar is not None:
                progress_bar.update_absolute(min(total_frames, max(0, int(encoded_seconds * frame_rate))))

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
            else:
                chunks = _iter_vsr_frame_chunks(
                    source,
                    target_width,
                    target_height,
                    quality=guide.get("rtx_quality", "HIGH"),
                    device_id=guide.get("device_id", 0),
                    **chunk_kwargs,
                )
            try:
                for chunk in chunks:
                    yield dasiwa._frame_bytes(chunk, selected_bit_depth)
            finally:
                close = getattr(chunks, "close", None)
                if callable(close):
                    close()

        metadata_path = dasiwa._metadata_file(prompt, extra_pnginfo) if save_metadata else None
        audio_path, audio_duration = dasiwa._audio_file(audio)
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
                    frame_rate, frame_chunks, output_path, quality, report_encode_progress,
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
                                frame_rate, frame_chunks, output_path, selected_container, quality, quality,
                                metadata_path, audio_path, audio_duration, crop_to_audio,
                                audio_codec, audio_bitrate, report_encode_progress,
                            )
                            break
                        except RuntimeError as error:
                            attempts.append(f"{selected_codec}/{selected_container}: {error}")
                            if isinstance(error, _VsrProcessingError):
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
                        frame_rate, frame_chunks, output_path, selected_container, quality, quality,
                        metadata_path, audio_path, audio_duration, crop_to_audio,
                        audio_codec, audio_bitrate, report_encode_progress,
                    )
        finally:
            if metadata_path:
                os.unlink(metadata_path)
            if audio_path:
                os.unlink(audio_path[0])

        frame_exports = []
        if output_path:
            if save_first_frame:
                if postprocess_path == "native_bypass":
                    frame_exports.append(_save_native_frame(source, 0, output_path, "first"))
                elif postprocess_path == "downscale":
                    frame_exports.append(_save_resized_frame(source, 0, target_width, target_height, output_path, "first"))
                else:
                    frame_exports.append(_save_vsr_frame(
                        source, 0, target_width, target_height,
                        guide.get("rtx_quality", "HIGH"), guide.get("device_id", 0),
                        output_path, "first",
                    ))
            if save_last_frame:
                last_index = 1 if pingpong and len(source) >= 3 else len(source) - 1
                if postprocess_path == "native_bypass":
                    frame_exports.append(_save_native_frame(source, last_index, output_path, "last"))
                elif postprocess_path == "downscale":
                    frame_exports.append(_save_resized_frame(source, last_index, target_width, target_height, output_path, "last"))
                else:
                    frame_exports.append(_save_vsr_frame(
                        source, last_index, target_width, target_height,
                        guide.get("rtx_quality", "HIGH"), guide.get("device_id", 0),
                        output_path, "last",
                    ))

        output_frames = source if pass_frames and postprocess_path == "native_bypass" else source[:0]
        mime_types = {
            "WebM": "video/webm", "MKV": "video/x-matroska", "MP4": "video/mp4",
            **{name: settings[2] for name, settings in dasiwa._ANIMATED_IMAGE_SETTINGS.items()},
        }
        output_mime_type = mime_types[selected_container]
        assets = [{
            "filename": os.path.basename(output_path), "subfolder": subfolder, "type": output_type,
            "format": output_mime_type, "width": target_width, "height": target_height,
            "codec": selected_codec, "bit_depth": selected_bit_depth, "container": selected_container,
            "postprocess_path": postprocess_path,
        }]
        assets.extend({
            "filename": os.path.basename(path), "subfolder": subfolder, "type": output_type,
            "format": "image/png", "width": target_width, "height": target_height,
            "postprocess_path": postprocess_path,
        } for path in frame_exports)
        ui = {"images": assets, "postprocess_path": postprocess_path}
        if not dasiwa._animated_image_settings(selected_container):
            ui["gifs"] = [{
                "filename": os.path.basename(output_path), "subfolder": subfolder, "type": output_type,
                "format": output_mime_type, "codec": selected_codec, "bit_depth": selected_bit_depth,
                "container": selected_container, "width": target_width, "height": target_height, "fps": frame_rate,
                "postprocess_path": postprocess_path,
            }]
        return {"ui": ui, "result": (output_frames, output_path)}
