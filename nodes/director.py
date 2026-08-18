"""Director Plus controller node."""

from __future__ import annotations

import json
import os

from .prompting import build_reference_prompt
from .resolution import ASPECTS, MEGAPIXELS, calculate_resolution, h3_native_canvas
from .rtx_vsr_stream import probe_vsr_capability
from .schema import PERFORMANCE_PRESETS, RequestError, low_vram_target_limit, normalize_request
from .upscale import _available_upscale_models, resolve_upscale_model_name


BASE_MODES = {"T2VA", "I2VA", "FL2VA", "L2VA"}


def _uploaded_files(content_types):
    try:
        import folder_paths

        input_dir = folder_paths.get_input_directory()
        files = [
            os.path.relpath(os.path.join(root, name), input_dir).replace(os.sep, "/")
            for root, _, names in os.walk(input_dir)
            for name in names
        ]
        files = folder_paths.filter_files_content_types(files, content_types)
        return ["", *sorted(files)]
    except (ImportError, OSError):
        return [""]


def load_uploaded_image(filename):
    if not str(filename or "").strip():
        return None
    try:
        import nodes as comfy_nodes

        return comfy_nodes.LoadImage().load_image(filename)[0]
    except (AttributeError, ImportError, OSError, ValueError) as exc:
        raise RequestError(f"无法加载导演台图片“{filename}”：{exc}") from exc


def load_uploaded_audio(filename):
    if not str(filename or "").strip():
        return None
    try:
        from comfy_extras.nodes_audio import LoadAudio

        result = LoadAudio.execute(filename)
        return result[0]
    except (AttributeError, ImportError, OSError, ValueError) as exc:
        raise RequestError(f"无法加载导演台音频“{filename}”：{exc}") from exc


def align_frame_count(frame_count):
    frame_count = max(5, int(frame_count))
    return frame_count + ((5 - frame_count) % 17)


def native_resolution_for_request(
    requested_width,
    requested_height,
    duration,
    performance_preset,
    aspect_ratio,
    custom_width=16,
    custom_height=9,
):
    """Choose a safe H3 sampling size while retaining the requested output target."""
    official_width, official_height = h3_native_canvas(
        aspect_ratio,
        int(custom_width),
        int(custom_height),
    )

    # The official H3 node supports a 768px short edge and 768x1344 area cap.
    # Apply that guard to every performance preset. The requested dimensions
    # remain the final output target and are handled after sampling.
    requested_area = int(requested_width) * int(requested_height)
    if requested_area > official_width * official_height:
        native_width, native_height = official_width, official_height
        capped = True
    else:
        native_width, native_height = int(requested_width), int(requested_height)
        capped = False

    if performance_preset != "low_vram":
        return native_width, native_height, capped

    # H3's QKV peak grows with both spatial tokens and frame count. Keep a
    # conservative native grid for RTX 3070-class 8GB cards, then restore the
    # user-selected output size after decoding with CPU chunked interpolation.
    duration_presets = (
        (4, "1.00 MP"),
        (6, "0.65 MP"),
        (8, "0.50 MP"),
        (10, "0.36 MP"),
        (12, "0.30 MP"),
        (15, "0.26 MP"),
    )
    native_preset = next(
        preset for maximum_duration, preset in duration_presets
        if int(duration) <= maximum_duration
    )
    native_width, native_height = calculate_resolution(
        native_preset,
        aspect_ratio,
        int(custom_width),
        int(custom_height),
    )
    target_area = int(requested_width) * int(requested_height)
    native_area = int(native_width) * int(native_height)
    if target_area <= native_area:
        return int(requested_width), int(requested_height), capped
    return native_width, native_height, True


def require_contiguous_slots(values, label):
    seen_empty = False
    for value in values:
        if value is None:
            seen_empty = True
        elif seen_empty:
            raise RequestError(f"{label}必须从 1 开始连续上传，不能跳过中间编号")


class MiniMaxH3DirectorPlus:
    @classmethod
    def INPUT_TYPES(cls):
        image_files = _uploaded_files(["image"])
        audio_files = _uploaded_files(["audio", "video"])
        return {
            "required": {
                "mode": (["T2VA", "I2VA", "FL2VA", "L2VA", "REF2VA"], {"default": "FL2VA", "tooltip": "生成模式"}),
                "prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "视频提示词"}),
                "duration": ("INT", {"default": 5, "min": 4, "max": 15, "tooltip": "H3 原生视频时长（4-15 秒）"}),
                "width": ("INT", {"default": 1344, "min": 32, "max": 8192, "step": 32, "tooltip": "输出宽度"}),
                "height": ("INT", {"default": 768, "min": 32, "max": 8192, "step": 32, "tooltip": "输出高度"}),
                "aspect_ratio": ([*ASPECTS, "CUSTOM"], {"default": "16:9", "tooltip": "画面比例，包含横向与竖向比例"}),
                "resolution_preset": (list(MEGAPIXELS), {"default": "0.83 MP", "tooltip": "最终输出目标分辨率档位；支持 2K QHD/4K UHD，低显存模式会按时长降低 H3 原生采样尺寸，再在最终编码阶段流式放大到此目标"}),
                "custom_width": ("INT", {"default": 16, "min": 1, "max": 8192, "tooltip": "自定义比例宽"}),
                "custom_height": ("INT", {"default": 9, "min": 1, "max": 8192, "tooltip": "自定义比例高"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": "seed_mode", "tooltip": "噪音种子；可选择固定、递增、递减或随机"}),
                "voice_mode": (["none", "h3_reference", "fish_lock"], {"default": "none", "tooltip": "无音色 / H3原生参考 / Fish高级锁定"}),
                "fish_model_path": (["s2-pro-w4a16 (auto download)", "s2-pro (auto download)"], {"default": "s2-pro-w4a16 (auto download)", "tooltip": "Fish S2 模型；量化版约需 8GB 显存"}),
                "ref_image_size": (["match", "max"], {"default": "match", "tooltip": "参考图尺寸策略"}),
                "performance_preset": (list(PERFORMANCE_PRESETS)[:5], {"default": "稳定质量", "tooltip": "性能预设"}),
                "postprocess_mode": (["native", "lanczos", "ai_upscale", "rtx_vsr"], {"default": "native", "tooltip": "原生直出 / Lanczos / 通用 AI 超分 / AI 细节重建（RTX VSR）"}),
                "rtx_quality": (["HIGH", "ULTRA"], {"default": "HIGH", "tooltip": "RTX VSR 质量"}),
                "ai_upscale_model": (["auto", *_available_upscale_models()], {"default": "auto", "tooltip": "通用 AI 超分模型；自动选择或指定已安装模型"}),
                "timeline_data": ("STRING", {"default": "{\"version\":1,\"items\":[]}", "multiline": False}),
                "target_dialogue": ("STRING", {"default": "", "multiline": True, "tooltip": "Fish高级音色锁定的目标对白"}),
                "reference_transcript": ("STRING", {"default": "", "multiline": True, "tooltip": "音色样本对应文本，可留空"}),
                "voice_reference_name_1": ("STRING", {"default": "", "tooltip": "音色参考1对应的角色名，可留空"}),
                "voice_reference_name_2": ("STRING", {"default": "", "tooltip": "音色参考2对应的角色名，可留空"}),
                "voice_reference_name_3": ("STRING", {"default": "", "tooltip": "音色参考3对应的角色名，可留空"}),
            },
            "optional": {
                "first_image_file": (image_files, {"image_upload": True, "tooltip": "在导演台内上传首帧"}),
                "last_image_file": (image_files, {"image_upload": True, "tooltip": "在导演台内上传尾帧"}),
                "voice_reference_audio_file": (audio_files, {"audio_upload": True, "tooltip": "在导演台内上传音色样本"}),
                "voice_reference_audio_2_file": (audio_files, {"audio_upload": True, "tooltip": "在导演台内上传音色参考2"}),
                "voice_reference_audio_3_file": (audio_files, {"audio_upload": True, "tooltip": "在导演台内上传音色参考3"}),
                "reference_image_1_file": (image_files, {"image_upload": True}),
                "reference_image_2_file": (image_files, {"image_upload": True}),
                "reference_image_3_file": (image_files, {"image_upload": True}),
                "reference_image_4_file": (image_files, {"image_upload": True}),
                "reference_image_5_file": (image_files, {"image_upload": True}),
                "reference_image_6_file": (image_files, {"image_upload": True}),
                "reference_image_7_file": (image_files, {"image_upload": True}),
                "reference_image_8_file": (image_files, {"image_upload": True}),
                "reference_image_9_file": (image_files, {"image_upload": True}),
                "first_image": ("IMAGE",),
                "last_image": ("IMAGE",),
                "voice_reference_audio": ("AUDIO",),
                "voice_reference_audio_2": ("AUDIO",),
                "voice_reference_audio_3": ("AUDIO",),
                "reference_image_1": ("IMAGE",),
                "reference_image_2": ("IMAGE",),
                "reference_image_3": ("IMAGE",),
                "reference_image_4": ("IMAGE",),
                "reference_image_5": ("IMAGE",),
                "reference_image_6": ("IMAGE",),
                "reference_image_7": ("IMAGE",),
                "reference_image_8": ("IMAGE",),
                "reference_image_9": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("MINIMAX_H3_DIRECTOR_PLUS_GUIDE", "INT", "STRING", "STRING", "STRING", "AUDIO", "STRING", "BOOLEAN", "INT")
    RETURN_NAMES = ("导演指南", "帧数", "最终提示词", "实际后端", "警告", "Fish音色样本", "Fish目标对白", "FL2VA硬端点", "噪音种子")
    FUNCTION = "build"
    CATEGORY = "MiniMax H3 导演台 Plus"

    def build(
        self,
        mode,
        prompt,
        duration,
        width,
        height,
        voice_mode,
        ref_image_size,
        performance_preset,
        timeline_data,
        target_dialogue,
        reference_transcript,
        postprocess_mode="native",
        rtx_quality="HIGH",
        ai_upscale_model="auto",
        fish_model_path="s2-pro-w4a16 (auto download)",
        aspect_ratio="16:9",
        resolution_preset="0.83 MP",
        custom_width=16,
        custom_height=9,
        seed=0,
        voice_reference_name_1="",
        voice_reference_name_2="",
        voice_reference_name_3="",
        first_image=None,
        last_image=None,
        voice_reference_audio=None,
        voice_reference_audio_2=None,
        voice_reference_audio_3=None,
        reference_image_1=None,
        reference_image_2=None,
        reference_image_3=None,
        reference_image_4=None,
        reference_image_5=None,
        reference_image_6=None,
        reference_image_7=None,
        reference_image_8=None,
        reference_image_9=None,
        first_image_file="",
        last_image_file="",
        voice_reference_audio_file="",
        voice_reference_audio_2_file="",
        voice_reference_audio_3_file="",
        reference_image_1_file="",
        reference_image_2_file="",
        reference_image_3_file="",
        reference_image_4_file="",
        reference_image_5_file="",
        reference_image_6_file="",
        reference_image_7_file="",
        reference_image_8_file="",
        reference_image_9_file="",
    ):
        requested_width, requested_height = calculate_resolution(
            resolution_preset,
            aspect_ratio,
            int(custom_width),
            int(custom_height),
        )
        try:
            timeline = json.loads(timeline_data or "{}")
        except (TypeError, json.JSONDecodeError) as exc:
            raise RequestError(f"素材时间线 JSON 无效：{exc}") from exc
        if not isinstance(timeline, dict):
            raise RequestError("素材时间线必须是 JSON 对象")

        first_image = first_image if first_image is not None else load_uploaded_image(first_image_file)
        last_image = last_image if last_image is not None else load_uploaded_image(last_image_file)
        voice_reference_audio = (
            voice_reference_audio
            if voice_reference_audio is not None
            else load_uploaded_audio(voice_reference_audio_file)
        )
        voice_reference_audio_2 = (
            voice_reference_audio_2
            if voice_reference_audio_2 is not None
            else load_uploaded_audio(voice_reference_audio_2_file)
        )
        voice_reference_audio_3 = (
            voice_reference_audio_3
            if voice_reference_audio_3 is not None
            else load_uploaded_audio(voice_reference_audio_3_file)
        )
        connected_references = (
            reference_image_1,
            reference_image_2,
            reference_image_3,
            reference_image_4,
            reference_image_5,
            reference_image_6,
            reference_image_7,
            reference_image_8,
            reference_image_9,
        )
        uploaded_reference_files = (
            reference_image_1_file,
            reference_image_2_file,
            reference_image_3_file,
            reference_image_4_file,
            reference_image_5_file,
            reference_image_6_file,
            reference_image_7_file,
            reference_image_8_file,
            reference_image_9_file,
        )
        reference_images = [
            connected if connected is not None else load_uploaded_image(filename)
            for connected, filename in zip(connected_references, uploaded_reference_files)
        ]
        if mode == "REF2VA" and (first_image is not None or last_image is not None):
            require_contiguous_slots((first_image, last_image, *reference_images), "参考图")
        else:
            require_contiguous_slots(reference_images, "参考图")

        voice_slots = (voice_reference_audio, voice_reference_audio_2, voice_reference_audio_3)
        require_contiguous_slots(voice_slots, "音色参考")
        voice_references = [item for item in voice_slots if item is not None]
        request = normalize_request({
            "mode": mode,
            "prompt": prompt,
            "duration": duration,
            "aspect_ratio": aspect_ratio,
            "resolution_preset": resolution_preset,
            "custom_width": custom_width,
            "custom_height": custom_height,
            "first_image": first_image,
            "last_image": last_image,
            "voice_mode": voice_mode,
            "voice_reference_audio": voice_reference_audio,
            "voice_reference_audios": voice_references,
            "target_dialogue": str(target_dialogue or "").strip(),
            "reference_transcript": str(reference_transcript or "").strip(),
            "fish_model_path": str(fish_model_path or "s2-pro-w4a16 (auto download)"),
            "references": [item for item in reference_images if item is not None],
            "ref_image_size": ref_image_size,
            "performance_preset": performance_preset,
            "postprocess_mode": postprocess_mode,
            "rtx_quality": rtx_quality,
            "ai_upscale_model": ai_upscale_model,
        })

        if request["performance_preset"] == "low_vram":
            limit_width, limit_height = low_vram_target_limit(request["duration"])
            requested_sides = sorted((int(requested_width), int(requested_height)), reverse=True)
            limit_sides = sorted((limit_width, limit_height), reverse=True)
            if any(requested > limit for requested, limit in zip(requested_sides, limit_sides)):
                if requested_width < requested_height:
                    limit_width, limit_height = limit_height, limit_width
                raise RequestError(
                    f"低显存模式下 {request['duration']} 秒视频的最终输出目标最大为 "
                    f"{limit_width}×{limit_height}"
                )

        native_width, native_height, native_capped = native_resolution_for_request(
            requested_width,
            requested_height,
            duration,
            request["performance_preset"],
            aspect_ratio,
            custom_width,
            custom_height,
        )
        if requested_width == native_width and requested_height == native_height:
            postprocess_path = "native_bypass"
        elif requested_width < native_width or requested_height < native_height:
            postprocess_path = "downscale"
        elif request["postprocess_mode"] in {"lanczos", "ai_upscale", "rtx_vsr"}:
            postprocess_path = request["postprocess_mode"]
        else:
            postprocess_path = "native_bypass"

        if postprocess_path == "native_bypass" and (
            requested_width != native_width or requested_height != native_height
        ):
            request["warnings"].append(
                "当前为原生尺寸直出，2K/4K 最终目标不会放大；"
                f"实际保存尺寸为 {native_width}×{native_height}。"
            )

        if postprocess_path == "rtx_vsr":
            try:
                probe_vsr_capability(request["rtx_quality"], device_id=0)
            except Exception as exc:
                raise RequestError(f"RTX VSR 前置检查失败，尚未开始 H3 视频生成：{exc}") from exc
        if postprocess_path == "ai_upscale":
            try:
                request["ai_upscale_model"] = resolve_upscale_model_name(
                    request["ai_upscale_model"],
                    max(
                        float(requested_width) / max(1, int(native_width)),
                        float(requested_height) / max(1, int(native_height)),
                    ),
                )
            except Exception as exc:
                raise RequestError(f"通用 AI 超分前置检查失败，尚未开始 H3 视频生成：{exc}") from exc

        if postprocess_path == "native_bypass":
            final_target_width, final_target_height = native_width, native_height
            final_upscale_required = False
        elif postprocess_path == "downscale":
            final_target_width, final_target_height = requested_width, requested_height
            final_upscale_required = False
        else:
            final_target_width, final_target_height = requested_width, requested_height
            final_upscale_required = True

        if native_capped:
            request["warnings"].append(
                f"H3 原生采样受官方画布上限限制为 {native_width}×{native_height}；最终目标为 {requested_width}×{requested_height}。"
            )
            if postprocess_mode == "rtx_vsr" and requested_width != native_width:
                request["warnings"][-1] += " 将在生成后通过 RTX VSR 重建。"

        ref_images = {}
        ref_audios = {}
        first_frame = first_image
        last_frame = last_image
        resolved_prompt = str(prompt or "").strip()

        if request["resolved_backend"] == "ref2va_model":
            first_frame = None
            last_frame = None
            if first_image is not None:
                ref_images[f"ref_image_{len(ref_images) + 1}"] = first_image
            if last_image is not None:
                ref_images[f"ref_image_{len(ref_images) + 1}"] = last_image
            for reference in request["references"]:
                ref_images[f"ref_image_{len(ref_images) + 1}"] = reference
            if voice_mode == "h3_reference":
                for index, audio in enumerate(voice_references, 1):
                    ref_audios[f"ref_audio_{index}"] = audio
            resolved_prompt = build_reference_prompt(
                mode=mode,
                detail=prompt,
                duration=duration,
                has_first=first_image is not None,
                has_last=last_image is not None,
                has_audio=voice_mode != "none",
                extra_reference_count=len(request["references"]),
                audio_count=len(voice_references),
                audio_names=(voice_reference_name_1, voice_reference_name_2, voice_reference_name_3),
            )

        length = align_frame_count(int(duration) * 24)
        guide = {
            "version": 1,
            "mode": mode,
            "voice_mode": voice_mode,
            "resolved_backend": request["resolved_backend"],
            "prompt": resolved_prompt,
            "width": int(native_width),
            "height": int(native_height),
            "native_width": int(native_width),
            "native_height": int(native_height),
            "native_cap_applied": bool(native_capped),
            "requested_width": int(requested_width),
            "requested_height": int(requested_height),
            "target_width": int(final_target_width),
            "target_height": int(final_target_height),
            "upscale_required": final_upscale_required,
            "postprocess_mode": request["postprocess_mode"],
            "rtx_quality": request["rtx_quality"],
            "ai_upscale_model": request["ai_upscale_model"],
            "postprocess_path": postprocess_path,
            "upscale_method": {
                "rtx_vsr": "rtx_vsr",
                "ai_upscale": "comfy_upscale_model",
                "lanczos": "lanczos",
                "downscale": "cpu_bicubic",
                "native_bypass": "none",
            }[postprocess_path],
            "length": length,
            "ref_image_size": ref_image_size,
            "first_frame": first_frame,
            "last_frame": last_frame,
            "ref_images": ref_images,
            "ref_videos": {},
            "ref_video_audios": {},
            "ref_audios": ref_audios,
            "performance_preset": request["performance_preset"],
            "timeline": timeline,
            "warnings": request["warnings"],
            "reference_transcript": reference_transcript,
        }
        warning_text = "\n".join(request["warnings"])
        exported_voice = voice_reference_audio if voice_mode == "fish_lock" else None
        exported_dialogue = str(target_dialogue or "").strip() if voice_mode == "fish_lock" else ""
        return (
            guide,
            length,
            resolved_prompt,
            request["resolved_backend"],
            warning_text,
            exported_voice,
            exported_dialogue,
            request["resolved_backend"] == "fl2va_model",
            int(seed),
        )
