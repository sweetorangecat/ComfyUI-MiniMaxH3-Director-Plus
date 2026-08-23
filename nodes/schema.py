"""Public request schema shared by canvas nodes and HTTP routes."""

from __future__ import annotations

from copy import deepcopy


MODES = ("T2VA", "I2VA", "FL2VA", "L2VA", "REF2VA")
VOICE_MODES = ("none", "h3_reference", "fish_lock")
POSTPROCESS_MODES = ("native", "lanczos", "ai_upscale", "rtx_vsr")
RTX_QUALITIES = ("HIGH", "ULTRA")
MOTION_SMOOTHING_MODES = ("auto", "off", "rife_x2")
AUDIO_LOUDNESS_MODES = ("auto", "original")
PUBLIC_API_KEYS = (
    "mode", "prompt", "duration", "aspect_ratio", "resolution_preset", "custom_width", "custom_height",
    "seed", "first_image", "last_image", "references", "voice_mode", "voice_reference_audio", "voice_reference_audios", "voice_reference_names",
    "target_dialogue", "reference_transcript", "fish_model_path", "ref_image_size", "performance_preset",
    "postprocess_mode", "rtx_quality", "ai_upscale_model", "motion_smoothing", "audio_loudness",
)
PERFORMANCE_PRESETS = {
    "稳定质量": "quality",
    "质量优先加速": "quality_sage",
    "质量优先二采样": "quality_two_stage",
    "极速4步": "fast_4step",
    "参考图加速": "reference_fast",
    "低显存": "low_vram",
    "自定义": "custom",
    "quality": "quality",
    "quality_sage": "quality_sage",
    "quality_two_stage": "quality_two_stage",
    "fast_4step": "fast_4step",
    "reference_fast": "reference_fast",
    "low_vram": "low_vram",
    "custom": "custom",
}

PERFORMANCE_PRESETS_BY_ROUTE = {
    # The official H3 Turbo LoRA ships with a T2V example workflow and is
    # compatible with the same FL2VA model endpoint used by T2VA/FL2VA/I2VA.
    "t2va": ("quality", "quality_sage", "quality_two_stage", "fast_4step", "low_vram"),
    "endpoint": ("quality", "quality_sage", "quality_two_stage", "fast_4step", "low_vram"),
    "reference": ("quality", "quality_sage", "quality_two_stage", "reference_fast", "fast_4step", "low_vram"),
}

# These routes are intentionally explicit. A quality two-stage pass already
# enlarges and redraws the H3 latent; it is paired with exactly one final RTX
# VSR output stage. Other presets still expose one selectable final-output
# method at a time.
POSTPROCESS_MODES_BY_PERFORMANCE = {
    "quality_two_stage": ("rtx_vsr",),
    "quality": POSTPROCESS_MODES,
    "quality_sage": POSTPROCESS_MODES,
    "fast_4step": POSTPROCESS_MODES,
    "reference_fast": POSTPROCESS_MODES,
    "low_vram": POSTPROCESS_MODES,
    "custom": POSTPROCESS_MODES,
}


def allowed_postprocess_modes(performance_preset):
    """Return final-output methods compatible with a normalized preset."""
    preset = PERFORMANCE_PRESETS.get(performance_preset, performance_preset)
    return POSTPROCESS_MODES_BY_PERFORMANCE.get(preset, POSTPROCESS_MODES)


def allowed_motion_smoothing(performance_preset, postprocess_mode):
    """Return resolved motion-smoothing paths compatible with this route."""
    preset = PERFORMANCE_PRESETS.get(performance_preset, performance_preset)
    if preset in {"low_vram", "quality_two_stage"} or postprocess_mode != "rtx_vsr":
        return ("off",)
    return ("off", "rife_x2")


def allowed_performance_presets(mode, voice_mode="none"):
    """Return the safe, user-facing presets for the active H3 route."""
    if voice_mode == "fish_lock":
        return tuple(item for item in PERFORMANCE_PRESETS_BY_ROUTE["reference"] if item != "quality_two_stage")
    if voice_mode != "none" or mode == "REF2VA":
        return PERFORMANCE_PRESETS_BY_ROUTE["reference"]
    if mode == "T2VA":
        return PERFORMANCE_PRESETS_BY_ROUTE["t2va"]
    return PERFORMANCE_PRESETS_BY_ROUTE["endpoint"]


def low_vram_target_limit(duration):
    """Return the maximum final-output dimensions for low-VRAM generation."""
    try:
        duration = int(duration)
    except (TypeError, ValueError) as exc:
        raise RequestError("视频时长必须是整数秒") from exc
    if duration < 4 or duration > 15:
        raise RequestError("视频时长必须在 4 到 15 秒之间")
    return 1920, 1080


class RequestError(ValueError):
    """A short actionable validation error suitable for the UI and API."""


def _present(value):
    """Check optional ComfyUI media without coercing a multi-element Tensor."""
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    try:
        return len(value) > 0
    except (TypeError, AttributeError):
        return True


DEFAULT_REQUEST = {
    "schema_version": "1.0",
    "mode": "FL2VA",
    "prompt": "",
    "duration": 5,
    "aspect_ratio": "16:9",
    "resolution_preset": "0.83 MP",
    "seed": 0,
    "first_image": None,
    "last_image": None,
    "references": [],
    "custom_width": 16,
    "custom_height": 9,
    "voice_mode": "none",
    "voice_reference_audio": None,
    "voice_reference_audios": [],
    "voice_reference_names": [],
    "target_dialogue": "",
    "reference_transcript": "",
    "fish_model_path": "s2-pro-w4a16 (auto download)",
    "ref_image_size": "match",
    "performance_preset": "quality",
    "postprocess_mode": "native",
    "rtx_quality": "HIGH",
    "ai_upscale_model": "auto",
    "motion_smoothing": "off",
    "audio_loudness": "auto",
    "postprocess": {},
    "output": {},
}


def normalize_request(raw=None):
    request = deepcopy(DEFAULT_REQUEST)
    request.update(dict(raw or {}))

    if not isinstance(request["references"], (list, tuple)):
        raise RequestError("额外参考图必须是列表")
    reference_total = len(request["references"]) + int(_present(request["first_image"])) + int(_present(request["last_image"]))
    if reference_total > 9:
        raise RequestError("首帧、尾帧和额外参考图合计最多支持 9 张")

    audio_references = list(request.get("voice_reference_audios") or [])
    if _present(request.get("voice_reference_audio")) and not audio_references:
        audio_references = [request["voice_reference_audio"]]
    if len(audio_references) > 3:
        raise RequestError("音色参考最多支持 3 路")
    if request["voice_mode"] == "fish_lock" and len(audio_references) > 1:
        raise RequestError("Fish 高级音色锁定只使用音色参考 1，请只上传 1 路音频")
    request["voice_reference_audios"] = audio_references
    voice_names = list(request.get("voice_reference_names") or [])
    if len(voice_names) > 3:
        raise RequestError("音色参考角色名最多支持 3 个")
    request["voice_reference_names"] = [str(name or "").strip() for name in voice_names]

    if request["mode"] not in MODES:
        raise RequestError(f"不支持的生成模式：{request['mode']}")
    if request["voice_mode"] not in VOICE_MODES:
        raise RequestError(f"不支持的音色模式：{request['voice_mode']}")
    if request["postprocess_mode"] not in POSTPROCESS_MODES:
        raise RequestError(f"不支持的后处理模式：{request['postprocess_mode']}")
    if request["rtx_quality"] not in RTX_QUALITIES:
        raise RequestError(f"不支持的 RTX VSR 质量：{request['rtx_quality']}")
    requested_motion_smoothing = str(request.get("motion_smoothing") or "off")
    if requested_motion_smoothing not in MOTION_SMOOTHING_MODES:
        raise RequestError(f"不支持的运动平滑模式：{requested_motion_smoothing}")
    request["motion_smoothing_requested"] = requested_motion_smoothing
    request["audio_loudness"] = str(request.get("audio_loudness") or "auto")
    if request["audio_loudness"] not in AUDIO_LOUDNESS_MODES:
        raise RequestError(f"不支持的最终音频响度模式：{request['audio_loudness']}")
    request["ai_upscale_model"] = str(request.get("ai_upscale_model") or "auto").strip() or "auto"

    try:
        duration = int(request["duration"])
    except (TypeError, ValueError) as exc:
        raise RequestError("视频时长必须是整数秒") from exc
    if duration < 4 or duration > 15:
        raise RequestError("视频时长必须在 4 到 15 秒之间")
    request["duration"] = duration

    if request["mode"] == "I2VA" and not _present(request["first_image"]):
        raise RequestError("I2VA 需要首帧图片")
    if request["mode"] == "FL2VA" and not (_present(request["first_image"]) or _present(request["last_image"])):
        raise RequestError("FL2VA 至少需要一张端点图片")
    if request["mode"] == "L2VA" and not _present(request["last_image"]):
        raise RequestError("L2VA 需要尾帧图片")

    if request["voice_mode"] != "none" and not audio_references:
        raise RequestError("当前音色模式需要音色参考音频")
    if request["voice_mode"] == "fish_lock" and not str(request["target_dialogue"]).strip():
        raise RequestError("Fish 高级音色锁定需要目标对白")

    preset = PERFORMANCE_PRESETS.get(request["performance_preset"])
    if preset is None:
        raise RequestError(f"不支持的性能预设：{request['performance_preset']}")
    request["performance_preset"] = preset

    allowed_postprocess = allowed_postprocess_modes(preset)
    if request["postprocess_mode"] not in allowed_postprocess:
        if preset == "quality_two_stage":
            raise RequestError(
                "质量优先二采样已包含 H3 latent 放大重绘，只能搭配 RTX VSR；"
                "请将最终输出切换为 RTX VSR（HIGH 或 ULTRA）"
            )
        raise RequestError(
            f"性能预设 {request['performance_preset']} 不支持后处理模式 "
            f"{request['postprocess_mode']}"
        )

    request["warnings"] = []
    allowed = allowed_performance_presets(request["mode"], request["voice_mode"])
    if preset not in allowed and preset != "custom":
        request["performance_preset"] = "quality"
        request["warnings"].append(
            f"{request['mode']} / {request['voice_mode']} 不支持性能预设 {preset}，已自动切换为稳定质量。"
        )

    resolved_preset = request["performance_preset"]
    if requested_motion_smoothing == "auto":
        request["motion_smoothing"] = "off"
    else:
        request["motion_smoothing"] = requested_motion_smoothing
    if request["motion_smoothing"] not in allowed_motion_smoothing(
        resolved_preset,
        request["postprocess_mode"],
    ):
        if resolved_preset == "quality_two_stage":
            raise RequestError("质量优先二采样固定关闭 RIFE 运动平滑，以避免动态云雾和大视差建筑产生重影")
        if resolved_preset == "low_vram":
            raise RequestError("低显存模式不支持 RIFE 运动平滑，请将运动平滑切换为关闭")
        raise RequestError("RIFE 2x 运动平滑只能搭配 RTX VSR 最终输出")

    request["resolved_backend"] = (
        "ref2va_model"
        if request["mode"] == "REF2VA" or request["voice_mode"] != "none"
        else "fl2va_model"
    )
    if request["resolved_backend"] == "ref2va_model" and request["mode"] != "REF2VA":
        request["warnings"].append(
            "已因音色参考切换到 REF2VA；首尾图片属于提示词约束，不是硬端点。"
        )
    return request


def public_schema():
    return {
        "version": "1.0",
        "properties": {
            "mode": {"中文名称": "生成模式", "enum": list(MODES), "default": "FL2VA"},
            "prompt": {"中文名称": "视频提示词", "type": "string", "default": ""},
            "duration": {"中文名称": "视频时长", "type": "integer", "minimum": 4, "maximum": 15, "default": 5},
            "aspect_ratio": {"中文名称": "画面比例", "type": "string", "default": "16:9"},
            "resolution_preset": {"中文名称": "最终输出目标分辨率档位", "type": "string", "default": "0.83 MP"},
            "seed": {"中文名称": "随机种子", "type": "integer", "minimum": 0, "default": 0},
            "first_image": {"中文名称": "首帧图片", "type": ["string", "null"]},
            "last_image": {"中文名称": "尾帧图片", "type": ["string", "null"]},
            "references": {"中文名称": "多媒体参考素材", "type": "array", "default": []},
            "voice_mode": {"中文名称": "音色模式", "enum": list(VOICE_MODES), "default": "none"},
            "voice_reference_audio": {"中文名称": "音色参考音频", "type": ["string", "null"]},
            "voice_reference_audios": {"中文名称": "编号音色参考音频", "type": "array", "maxItems": 3, "default": []},
            "voice_reference_names": {"中文名称": "编号音色对应角色", "type": "array", "maxItems": 3, "default": []},
            "target_dialogue": {"中文名称": "目标对白", "type": "string", "default": ""},
            "reference_transcript": {"中文名称": "音色样本文本", "type": "string", "default": ""},
            "fish_model_path": {"中文名称": "Fish S2 模型", "type": "string", "default": "s2-pro-w4a16 (auto download)"},
            "ref_image_size": {"中文名称": "参考图尺寸策略", "enum": ["match", "max"], "default": "match"},
            "performance_preset": {
                "中文名称": "性能预设",
                "enum": list(PERFORMANCE_PRESETS)[:7],
                "default": "稳定质量",
                "allowed_by_route": {
                    "T2VA": ["稳定质量", "质量优先加速", "质量优先二采样", "极速4步", "低显存"],
                    "I2VA / FL2VA / L2VA": ["稳定质量", "质量优先加速", "质量优先二采样", "极速4步", "低显存"],
                    "REF2VA": ["稳定质量", "质量优先加速", "质量优先二采样", "参考图加速", "极速4步", "低显存"],
                    "I2VA + 音色参考": ["稳定质量", "质量优先加速", "质量优先二采样", "参考图加速", "极速4步", "低显存"],
                    "FL2VA + 音色参考": ["稳定质量", "质量优先加速", "质量优先二采样", "参考图加速", "极速4步", "低显存"],
                    "L2VA + 音色参考": ["稳定质量", "质量优先加速", "质量优先二采样", "参考图加速", "极速4步", "低显存"],
                    "T2VA + 音色参考": ["稳定质量", "质量优先加速", "质量优先二采样", "参考图加速", "极速4步", "低显存"],
                },
            },
            "postprocess_mode": {
                "中文名称": "最终输出后处理模式",
                "enum": list(POSTPROCESS_MODES),
                "default": "native",
                "description": "四种最终输出路线：原生尺寸直出、Lanczos 快速放大、通用 AI 自动超分、NVIDIA RTX VSR AI 细节重建。",
                "allowed_by_performance": {
                    "质量优先二采样": ["rtx_vsr"],
                    "其他性能预设": list(POSTPROCESS_MODES),
                },
            },
            "rtx_quality": {
                "中文名称": "RTX VSR 质量",
                "enum": list(RTX_QUALITIES),
                "default": "HIGH",
                "description": "RTX VSR 的质量级别，仅在后处理模式为 rtx_vsr 时生效。",
            },
            "ai_upscale_model": {
                "中文名称": "通用 AI 超分模型",
                "type": "string",
                "default": "auto",
                "description": "自动选择或指定 models/upscale_models 中的通用 AI 超分模型，仅在 ai_upscale 模式生效。",
            },
            "motion_smoothing": {
                "中文名称": "运动平滑",
                "enum": list(MOTION_SMOOTHING_MODES),
                "default": "off",
                "description": "默认关闭以保留 H3 原始帧；可手动启用流式 RIFE 2x，旧 auto 值按关闭处理。质量优先二采样和低显存模式只允许关闭。",
                "allowed_by_performance": {
                    "质量优先二采样 + RTX VSR": ["auto", "off"],
                    "低显存": ["off"],
                    "其他 RTX VSR": ["auto", "off", "rife_x2"],
                    "非 RTX VSR": ["auto", "off"],
                },
            },
            "audio_loudness": {
                "中文名称": "最终音频响度",
                "enum": list(AUDIO_LOUDNESS_MODES),
                "default": "auto",
                "description": "自动模式在最终编码前安全提升过小的 H3 音频响度；original 保持原始波形。",
            },
            "custom_width": {"中文名称": "自定义宽度", "type": "integer", "minimum": 1, "maximum": 8192, "default": 16},
            "custom_height": {"中文名称": "自定义高度", "type": "integer", "minimum": 1, "maximum": 8192, "default": 9},
        },
        "resolved_outputs": {
            "resolved_two_stage_route": {"中文名称": "实际训练型二采路线", "type": "string", "readOnly": True},
            "first_stage_width": {"中文名称": "H3首采宽度", "type": "integer", "readOnly": True},
            "first_stage_height": {"中文名称": "H3首采高度", "type": "integer", "readOnly": True},
            "second_stage_width": {"中文名称": "神经latent二采宽度", "type": "integer", "readOnly": True},
            "second_stage_height": {"中文名称": "神经latent二采高度", "type": "integer", "readOnly": True},
            "final_upscale_scale_x": {"中文名称": "最终横向放大倍率", "type": "number", "readOnly": True},
            "final_upscale_scale_y": {"中文名称": "最终纵向放大倍率", "type": "number", "readOnly": True},
            "vram_safety_tier": {"中文名称": "显存安全档位", "type": "string", "readOnly": True},
            "quality_basis": {"中文名称": "清晰度基础", "type": "string", "readOnly": True},
            "required_assets": {"中文名称": "本次所需模型资产", "type": "array", "readOnly": True},
        },
    }
