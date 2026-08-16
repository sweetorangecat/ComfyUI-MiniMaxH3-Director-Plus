"""Public request schema shared by canvas nodes and HTTP routes."""

from __future__ import annotations

from copy import deepcopy


MODES = ("T2VA", "I2VA", "FL2VA", "L2VA", "REF2VA")
VOICE_MODES = ("none", "h3_reference", "fish_lock")
PUBLIC_API_KEYS = (
    "mode", "prompt", "duration", "aspect_ratio", "resolution_preset", "custom_width", "custom_height",
    "seed", "first_image", "last_image", "references", "voice_mode", "voice_reference_audio", "voice_reference_audios", "voice_reference_names",
    "target_dialogue", "reference_transcript", "fish_model_path", "ref_image_size", "performance_preset",
)
PERFORMANCE_PRESETS = {
    "稳定质量": "quality",
    "质量优先加速": "quality_sage",
    "极速4步": "fast_4step",
    "参考图加速": "reference_fast",
    "低显存": "low_vram",
    "自定义": "custom",
    "quality": "quality",
    "quality_sage": "quality_sage",
    "fast_4step": "fast_4step",
    "reference_fast": "reference_fast",
    "low_vram": "low_vram",
    "custom": "custom",
}

PERFORMANCE_PRESETS_BY_ROUTE = {
    # The official H3 Turbo LoRA ships with a T2V example workflow and is
    # compatible with the same FL2VA model endpoint used by T2VA/FL2VA/I2VA.
    "t2va": ("quality", "quality_sage", "fast_4step", "low_vram"),
    "endpoint": ("quality", "quality_sage", "fast_4step", "low_vram"),
    "reference": ("quality", "quality_sage", "reference_fast", "fast_4step", "low_vram"),
}


def allowed_performance_presets(mode, voice_mode="none"):
    """Return the safe, user-facing presets for the active H3 route."""
    if voice_mode != "none" or mode == "REF2VA":
        return PERFORMANCE_PRESETS_BY_ROUTE["reference"]
    if mode == "T2VA":
        return PERFORMANCE_PRESETS_BY_ROUTE["t2va"]
    return PERFORMANCE_PRESETS_BY_ROUTE["endpoint"]


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

    request["warnings"] = []
    allowed = allowed_performance_presets(request["mode"], request["voice_mode"])
    if preset not in allowed and preset != "custom":
        request["performance_preset"] = "quality"
        request["warnings"].append(
            f"{request['mode']} / {request['voice_mode']} 不支持性能预设 {preset}，已自动切换为稳定质量。"
        )

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
                "enum": list(PERFORMANCE_PRESETS)[:6],
                "default": "稳定质量",
                "allowed_by_route": {
                    "T2VA": ["稳定质量", "质量优先加速", "极速4步", "低显存"],
                    "I2VA / FL2VA / L2VA": ["稳定质量", "质量优先加速", "极速4步", "低显存"],
                    "REF2VA": ["稳定质量", "质量优先加速", "参考图加速", "极速4步", "低显存"],
                    "I2VA + 音色参考": ["稳定质量", "质量优先加速", "参考图加速", "极速4步", "低显存"],
                    "FL2VA + 音色参考": ["稳定质量", "质量优先加速", "参考图加速", "极速4步", "低显存"],
                    "L2VA + 音色参考": ["稳定质量", "质量优先加速", "参考图加速", "极速4步", "低显存"],
                    "T2VA + 音色参考": ["稳定质量", "质量优先加速", "参考图加速", "极速4步", "低显存"],
                },
            },
            "custom_width": {"中文名称": "自定义宽度", "type": "integer", "minimum": 1, "maximum": 8192, "default": 16},
            "custom_height": {"中文名称": "自定义高度", "type": "integer", "minimum": 1, "maximum": 8192, "default": 9},
        },
    }
