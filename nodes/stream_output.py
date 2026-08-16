"""Stream final-size H3 frames to FFmpeg without keeping a full 4K batch."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image


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


def _save_resized_frame(source, index, target_width, target_height, output_path, suffix):
    frame = _resize_cpu_chunk(source[index:index + 1], target_width, target_height)[0]
    pixels = torch.round(frame[..., :3] * 255).to(torch.uint8).numpy()
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
        # Keep only the low-resolution decoded batch on CPU when resizing is
        # required. For a native-size output, each small source slice is
        # copied by the iterator, avoiding an unnecessary full CPU duplicate.
        source = images.detach()
        target_width = int(guide.get("target_width") or source.shape[2])
        target_height = int(guide.get("target_height") or source.shape[1])
        resizing = bool(guide.get("upscale_required")) or (
            target_width != int(source.shape[2]) or target_height != int(source.shape[1])
        )
        if resizing:
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
            for chunk in _iter_resized_frame_chunks(
                source,
                target_width,
                target_height,
                max_chunk_bytes=dasiwa._MAX_RAW_FRAME_CHUNK_BYTES,
                bytes_per_channel=2 if selected_bit_depth == 10 else 1,
                pingpong=pingpong,
            ):
                yield dasiwa._frame_bytes(chunk, selected_bit_depth)

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
                frame_exports.append(_save_resized_frame(source, 0, target_width, target_height, output_path, "first"))
            if save_last_frame:
                last_index = 1 if pingpong and len(source) >= 3 else len(source) - 1
                frame_exports.append(_save_resized_frame(source, last_index, target_width, target_height, output_path, "last"))

        output_frames = source if pass_frames and not resizing else source[:0]
        mime_types = {
            "WebM": "video/webm", "MKV": "video/x-matroska", "MP4": "video/mp4",
            **{name: settings[2] for name, settings in dasiwa._ANIMATED_IMAGE_SETTINGS.items()},
        }
        output_mime_type = mime_types[selected_container]
        assets = [{
            "filename": os.path.basename(output_path), "subfolder": subfolder, "type": output_type,
            "format": output_mime_type, "width": target_width, "height": target_height,
            "codec": selected_codec, "bit_depth": selected_bit_depth, "container": selected_container,
        }]
        assets.extend({
            "filename": os.path.basename(path), "subfolder": subfolder, "type": output_type,
            "format": "image/png", "width": target_width, "height": target_height,
        } for path in frame_exports)
        ui = {"images": assets}
        if not dasiwa._animated_image_settings(selected_container):
            ui["gifs"] = [{
                "filename": os.path.basename(output_path), "subfolder": subfolder, "type": output_type,
                "format": output_mime_type, "codec": selected_codec, "bit_depth": selected_bit_depth,
                "container": selected_container, "width": target_width, "height": target_height, "fps": frame_rate,
            }]
        return {"ui": ui, "result": (output_frames, output_path)}
