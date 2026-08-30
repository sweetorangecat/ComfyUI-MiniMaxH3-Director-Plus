"""Director Plus controller node."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from .prompting import build_reference_prompt
from .resolution import ASPECTS, MEGAPIXELS, calculate_resolution, h3_native_canvas
from .rtx_vsr_stream import probe_vsr_capability
from .rife_stream import DEFAULT_RIFE_MODEL, probe_rife_capability
from .schema import (
    TWO_STAGE_PERFORMANCE_PRESETS,
    USER_PERFORMANCE_PRESET_LABELS,
    RequestError,
    low_vram_target_limit,
    normalize_request,
)
from .smart_1080p import SMART_PRESET, resolve_smart_1080p_plan, smart_1080p_target
from .two_stage_assets import dependency_report, resolve_two_stage_route
from .upscale import (
    _available_upscale_models,
    is_x2_upscale_model_name,
    resolve_upscale_model_name,
)
from .vram_budget import plan_two_stage_dimensions


BASE_MODES = {"T2VA", "I2VA", "FL2VA", "L2VA"}

TWO_STAGE_IMAGE_SCALE = 1.5
LOGGER = logging.getLogger("MiniMaxH3.DirectorPlus")


def _cuda_memory_gb():
    """Return current free/total CUDA memory for the active generation GPU."""
    try:
        import torch

        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        return total_bytes / 2**30, free_bytes / 2**30
    except Exception:
        return 0.0, 0.0


def _trained_two_stage_dependency_report(route):
    try:
        import nodes as comfy_nodes

        mappings = getattr(comfy_nodes, "NODE_CLASS_MAPPINGS", {})
    except (AttributeError, ImportError):
        mappings = {}
    comfy_root = Path(__file__).resolve().parents[3]
    return dependency_report(comfy_root, route, mappings)


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
                "performance_preset": (list(USER_PERFORMANCE_PRESET_LABELS), {"default": "免费智能 1080p", "tooltip": "性能预设；默认本地免费智能 1080p"}),
                "postprocess_mode": (["native", "lanczos", "ai_upscale", "rtx_vsr"], {"default": "ai_upscale", "tooltip": "原生直出 / Lanczos / 通用 AI 超分 / AI 细节重建（RTX VSR）"}),
                "rtx_quality": (["HIGH", "ULTRA", "HIGHBITRATE_ULTRA"], {"default": "HIGH", "tooltip": "RTX VSR 质量；质量优先二采样自动使用原画源最高保真档"}),
                "ai_upscale_model": (["auto", *_available_upscale_models()], {"default": "RealESRGAN_x2plus.pth", "tooltip": "通用 AI 超分模型；默认本地 RealESRGAN X2"}),
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
                "motion_smoothing": (["off", "rife_x2"], {"default": "off", "tooltip": "运动平滑：默认关闭以保留原始帧；需要 48 FPS 时手动启用流式 RIFE 2x；低显存模式只允许关闭"}),
                "audio_loudness": (["auto", "original"], {"default": "auto", "tooltip": "最终音频：自动增强过小的 H3 音量，或保持原始响度"}),
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
        postprocess_mode="ai_upscale",
        rtx_quality="HIGH",
        ai_upscale_model="RealESRGAN_x2plus.pth",
        motion_smoothing="off",
        audio_loudness="auto",
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

        # Do not even load hidden stale widgets that the selected mode cannot
        # consume.  This prevents a mode switch from turning an old FL2VA
        # endpoint into a T2VA first frame, and avoids errors for deleted files.
        first_allowed = mode in {"I2VA", "FL2VA", "REF2VA"}
        last_allowed = mode in {"FL2VA", "L2VA", "REF2VA"}
        def supplied(value, filename):
            return value is not None or bool(str(filename or "").strip())

        ignored_media = []
        if not first_allowed and supplied(first_image, first_image_file):
            ignored_media.append("首帧图片")
        if not last_allowed and supplied(last_image, last_image_file):
            ignored_media.append("尾帧图片")
        if mode != "REF2VA" and any(
            supplied(value, filename)
            for value, filename in zip(
                (reference_image_1, reference_image_2, reference_image_3, reference_image_4,
                 reference_image_5, reference_image_6, reference_image_7, reference_image_8,
                 reference_image_9),
                (reference_image_1_file, reference_image_2_file, reference_image_3_file,
                 reference_image_4_file, reference_image_5_file, reference_image_6_file,
                 reference_image_7_file, reference_image_8_file, reference_image_9_file),
            )
        ):
            ignored_media.append("参考图")
        first_image = (
            first_image
            if first_allowed and first_image is not None
            else load_uploaded_image(first_image_file) if first_allowed else None
        )
        last_image = (
            last_image
            if last_allowed and last_image is not None
            else load_uploaded_image(last_image_file) if last_allowed else None
        )
        voice_reference_audio = (
            voice_reference_audio
            if voice_mode != "none" and voice_reference_audio is not None
            else load_uploaded_audio(voice_reference_audio_file) if voice_mode != "none" else None
        )
        voice_reference_audio_2 = (
            voice_reference_audio_2
            if voice_mode == "h3_reference" and voice_reference_audio_2 is not None
            else load_uploaded_audio(voice_reference_audio_2_file) if voice_mode == "h3_reference" else None
        )
        voice_reference_audio_3 = (
            voice_reference_audio_3
            if voice_mode == "h3_reference" and voice_reference_audio_3 is not None
            else load_uploaded_audio(voice_reference_audio_3_file) if voice_mode == "h3_reference" else None
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
        if mode == "REF2VA":
            reference_images = [
                connected if connected is not None else load_uploaded_image(filename)
                for connected, filename in zip(connected_references, uploaded_reference_files)
            ]
        else:
            reference_images = [None] * len(connected_references)
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
            "motion_smoothing": motion_smoothing,
            "audio_loudness": audio_loudness,
            "ignored_media": ignored_media,
        })

        requested_performance_preset = request["performance_preset"]
        smart_mode = requested_performance_preset == SMART_PRESET
        smart_plan = None
        smart_vram = None
        if smart_mode:
            smart_vram = _cuda_memory_gb()
            total_vram_gb, free_vram_gb = smart_vram
            smart_plan = resolve_smart_1080p_plan(
                request["resolved_backend"], request["duration"], total_vram_gb, free_vram_gb
            )
            request["performance_preset"] = smart_plan["performance_preset"]
            request["postprocess_mode"] = smart_plan["postprocess_mode"]
            request["ai_upscale_model"] = smart_plan["ai_upscale_model"]
            request["motion_smoothing"] = smart_plan["motion_smoothing"]
            if smart_plan["warning"]:
                request["warnings"].append(smart_plan["warning"])
            if aspect_ratio == "CUSTOM":
                target_ratio = (int(custom_width), int(custom_height))
            else:
                target_ratio = ASPECTS[aspect_ratio]
            requested_width, requested_height = smart_1080p_target(*target_ratio)

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

        two_stage_plan = None
        resolved_two_stage_route = "bypass"
        required_assets = []
        if request["performance_preset"] in TWO_STAGE_PERFORMANCE_PRESETS:
            resolved_two_stage_route = resolve_two_stage_route(request)
            if smart_vram is None:
                total_vram_gb, free_vram_gb = _cuda_memory_gb()
            else:
                total_vram_gb, free_vram_gb = smart_vram
            if total_vram_gb <= 0:
                raise RequestError("无法读取当前 GPU 显存，已阻止训练型二采启动")
            two_stage_plan = plan_two_stage_dimensions(
                requested_width,
                requested_height,
                duration,
                total_vram_gb,
                free_vram_gb,
                profile=(
                    "low_vram"
                    if request["performance_preset"] == "low_vram_two_stage"
                    else "quality"
                ),
            )
            if not two_stage_plan["allowed"]:
                raise RequestError(f"训练型二采显存前置检查失败：{two_stage_plan['reason']}")
            dependencies = _trained_two_stage_dependency_report(resolved_two_stage_route)
            if not dependencies.get("ready"):
                missing = "、".join(str(item) for item in dependencies.get("missing", []))
                raise RequestError(f"训练型二采依赖缺失：{missing}")
            native_width = int(two_stage_plan["first_stage_width"])
            native_height = int(two_stage_plan["first_stage_height"])
            native_capped = (native_width, native_height) != (
                int(requested_width),
                int(requested_height),
            )
            required_assets = list(dependencies.get("required_assets", []))
        else:
            native_width, native_height, native_capped = native_resolution_for_request(
                requested_width,
                requested_height,
                duration,
                request["performance_preset"],
                aspect_ratio,
                custom_width,
                custom_height,
            )
        postprocess_source_width = int(
            two_stage_plan["second_stage_width"] if two_stage_plan else native_width
        )
        postprocess_source_height = int(
            two_stage_plan["second_stage_height"] if two_stage_plan else native_height
        )
        if two_stage_plan is not None and (
            two_stage_plan.get("balanced_fhd_supersample")
            or two_stage_plan.get("conservative_fhd_supersample")
        ):
            postprocess_path = "balanced_fhd_downscale"
        elif requested_width == postprocess_source_width and requested_height == postprocess_source_height:
            postprocess_path = "native_bypass"
        elif requested_width < postprocess_source_width or requested_height < postprocess_source_height:
            postprocess_path = "downscale"
        elif request["postprocess_mode"] in {"lanczos", "ai_upscale", "rtx_vsr"}:
            postprocess_path = request["postprocess_mode"]
        else:
            postprocess_path = "native_bypass"
        # DEBLUR_LOW was removed from the automatic quality route after the
        # NVIDIA effect produced corrupted RGB frames on a real server GPU.
        # Keep the guide field for old workflows, but force the safe single-VSR
        # path so a stale value cannot re-enable the broken chain.
        request["rtx_deblur_mode"] = "off"

        if two_stage_plan is not None and postprocess_path == "balanced_fhd_downscale":
            if two_stage_plan.get("conservative_fhd_supersample"):
                fhd_warning_prefix = "1080p 保守 FHD 二采已启用："
            else:
                # Keep the original label stable for existing UI/API consumers.
                fhd_warning_prefix = "1080p 平衡二采已启用（FHD）："
            request["warnings"].append(
                fhd_warning_prefix +
                f"首采 {native_width}×{native_height}，神经二采 "
                f"{two_stage_plan['second_stage_width']}×{two_stage_plan['second_stage_height']}，"
                f"最终中心等比裁切并 Lanczos 缩小到 {requested_width}×{requested_height}；"
                "保留 4+4/4+5 加速和原始时长，不执行 RTX VSR。"
            )
        elif two_stage_plan is not None and postprocess_path == "rtx_vsr":
            request["warnings"].append(
                "训练型 H3 latent 二采已通过显存预算："
                f"首采 {native_width}×{native_height}，神经二采 "
                f"{two_stage_plan['second_stage_width']}×{two_stage_plan['second_stage_height']}，"
                f"最终 RTX VSR 约 {two_stage_plan['final_scale']:.2f} 倍。"
            )
            request["warnings"].append(
                "质量二采使用单次 HIGHBITRATE_ULTRA RTX VSR；"
                "已关闭不稳定的 DEBLUR_LOW 双效果链，避免彩条、灰屏和伪影。"
            )
        elif two_stage_plan is not None and postprocess_path == "ai_upscale":
            request["warnings"].append(
                "低显存训练型二采已通过显存预算："
                f"首采 {native_width}×{native_height}，神经二采 "
                f"{two_stage_plan['second_stage_width']}×{two_stage_plan['second_stage_height']}，"
                f"最终 AI X2 细节重建约 {two_stage_plan['final_scale']:.2f} 倍。"
            )
        elif two_stage_plan is not None and postprocess_path == "native_bypass":
            request["warnings"].append(
                "训练型神经 latent 二采已达到最终目标尺寸，无需重复执行 RTX VSR。"
            )

        if postprocess_path == "native_bypass" and (
            requested_width != postprocess_source_width
            or requested_height != postprocess_source_height
        ):
            request["warnings"].append(
                "当前为原生尺寸直出，2K/4K 最终目标不会放大；"
                f"实际保存尺寸为 {postprocess_source_width}×{postprocess_source_height}。"
            )

        if postprocess_path == "rtx_vsr":
            try:
                probe_vsr_capability(request["rtx_quality"], device_id=0)
            except Exception as exc:
                raise RequestError(str(exc)) from exc
        if request["motion_smoothing"] == "rife_x2":
            if postprocess_path != "rtx_vsr":
                if request.get("motion_smoothing_requested") == "auto":
                    request["motion_smoothing"] = "off"
                else:
                    raise RequestError("RIFE 2x 运动平滑只能搭配实际执行的 RTX VSR 输出")
            else:
                try:
                    probe_rife_capability(DEFAULT_RIFE_MODEL)
                except Exception as exc:
                    raise RequestError(
                        f"RIFE 运动平滑前置检查失败，尚未开始 H3 视频生成：{exc}"
                    ) from exc
        if postprocess_path == "ai_upscale":
            try:
                request["ai_upscale_model"] = resolve_upscale_model_name(
                    request["ai_upscale_model"],
                    max(
                        float(requested_width) / max(1, postprocess_source_width),
                        float(requested_height) / max(1, postprocess_source_height),
                    ),
                )
                if (
                    request["performance_preset"] == "low_vram_two_stage"
                    and not is_x2_upscale_model_name(request["ai_upscale_model"])
                ):
                    raise ValueError(
                        "低显存二采只允许 X2 超分模型，"
                        f"当前解析为 {request['ai_upscale_model']}"
                    )
                if smart_mode and Path(request["ai_upscale_model"]).name.lower() != "realesrgan_x2plus.pth".lower():
                    raise ValueError(
                        "免费智能 1080p 只允许 RealESRGAN_x2plus.pth，"
                        f"当前解析为 {request['ai_upscale_model']}"
                    )
                required_assets.append(request["ai_upscale_model"])
            except Exception as exc:
                raise RequestError(f"通用 AI 超分前置检查失败，尚未开始 H3 视频生成：{exc}") from exc

        if postprocess_path == "native_bypass":
            final_target_width, final_target_height = postprocess_source_width, postprocess_source_height
            final_upscale_required = False
        elif postprocess_path in {"downscale", "balanced_fhd_downscale"}:
            final_target_width, final_target_height = requested_width, requested_height
            final_upscale_required = False
        else:
            final_target_width, final_target_height = requested_width, requested_height
            final_upscale_required = True

        if native_capped and two_stage_plan is None:
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
            "postprocess_source_width": int(postprocess_source_width),
            "postprocess_source_height": int(postprocess_source_height),
            "native_cap_applied": bool(native_capped),
            "two_stage_image_scale": TWO_STAGE_IMAGE_SCALE,
            "resolved_two_stage_route": resolved_two_stage_route,
            "first_stage_width": int(native_width),
            "first_stage_height": int(native_height),
            "second_stage_width": int(
                two_stage_plan["second_stage_width"] if two_stage_plan else native_width
            ),
            "second_stage_height": int(
                two_stage_plan["second_stage_height"] if two_stage_plan else native_height
            ),
            "final_upscale_scale_x": float(
                two_stage_plan["final_scale_x"] if two_stage_plan else 1.0
            ),
            "final_upscale_scale_y": float(
                two_stage_plan["final_scale_y"] if two_stage_plan else 1.0
            ),
            "final_upscale_scale": float(
                two_stage_plan["final_scale"] if two_stage_plan else 1.0
            ),
            "max_final_vsr_scale": (
                two_stage_plan.get("max_final_vsr_scale")
                if two_stage_plan
                else None
            ),
            "vram_safety_tier": (
                two_stage_plan["vram_safety_tier"] if two_stage_plan else "not_applicable"
            ),
            "quality_basis": (
                two_stage_plan["quality_basis"] if two_stage_plan else "H3 原生"
            ),
            "required_assets": required_assets,
            "requested_width": int(requested_width),
            "requested_height": int(requested_height),
            "target_width": int(final_target_width),
            "target_height": int(final_target_height),
            "final_width": int(final_target_width),
            "final_height": int(final_target_height),
            "upscale_required": final_upscale_required,
            "postprocess_mode": request["postprocess_mode"],
            "rtx_quality_requested": request["rtx_quality_requested"],
            "rtx_quality": request["rtx_quality"],
            "rtx_deblur_mode": request["rtx_deblur_mode"],
            "ai_upscale_model": request["ai_upscale_model"],
            "motion_smoothing": request["motion_smoothing"],
            "audio_loudness": request["audio_loudness"],
            "audio_cleanup_requested": (
                "auto_gate_peak_limit"
                if request["audio_loudness"] == "auto"
                else "disabled"
            ),
            "rife_model": DEFAULT_RIFE_MODEL,
            "output_frame_multiplier": 2 if request["motion_smoothing"] == "rife_x2" else 1,
            "postprocess_path": postprocess_path,
            "upscale_profile": (
                "smart_conservative_blend_v1"
                if smart_mode and postprocess_path == "ai_upscale"
                else "standard"
            ),
            "upscale_method": {
                "rtx_vsr": "rtx_vsr",
                "ai_upscale": "comfy_upscale_model",
                "lanczos": "lanczos",
                "balanced_fhd_downscale": "aspect_lanczos_downscale",
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
            "requested_performance_preset": requested_performance_preset,
            "timeline": timeline,
            "warnings": request["warnings"],
            "reference_transcript": reference_transcript,
            "ignored_media": request.get("ignored_media", []),
        }
        LOGGER.info(
            "[H3 director] mode=%s backend=%s preset=%s first_frame=%s last_frame=%s "
            "ref_images=%d ref_audios=%d ignored_media=%s native=%sx%s target=%sx%s postprocess=%s",
            mode,
            request["resolved_backend"],
            request["performance_preset"],
            bool(first_frame is not None),
            bool(last_frame is not None),
            len(ref_images),
            len(ref_audios),
            ",".join(request.get("ignored_media", [])) or "none",
            native_width,
            native_height,
            final_target_width,
            final_target_height,
            postprocess_path,
        )
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
