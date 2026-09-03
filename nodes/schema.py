"""Public request schema shared by canvas nodes and HTTP routes."""

from __future__ import annotations

import re
from copy import deepcopy


MODES = ("T2VA", "I2VA", "FL2VA", "L2VA", "REF2VA")
VOICE_MODES = ("none", "h3_reference", "fish_lock")
POSTPROCESS_MODES = ("native", "lanczos", "ai_upscale", "video_sr", "rtx_vsr")
RTX_QUALITIES = ("HIGH", "ULTRA", "HIGHBITRATE_ULTRA")
MOTION_SMOOTHING_MODES = ("auto", "off", "rife_x2")
AUDIO_LOUDNESS_MODES = ("auto", "original")
PUBLIC_API_KEYS = (
    "mode", "prompt", "duration", "aspect_ratio", "resolution_preset", "custom_width", "custom_height",
    "seed", "first_image", "last_image", "references", "voice_mode", "voice_reference_audio", "voice_reference_audios", "voice_reference_names",
    "voice_gender", "target_dialogue", "reference_transcript", "fish_model_path", "ref_image_size", "performance_preset",
    "postprocess_mode", "rtx_quality", "ai_upscale_model", "motion_smoothing", "audio_loudness",
)
PERFORMANCE_PRESETS = {
    "免费智能 1080p": "smart_free_1080p",
    "稳定质量": "quality",
    "质量优先加速": "quality_sage",
    "质量优先二采样": "quality_two_stage",
    "高清快速（v4 8步）": "fl_quality_fast_v4",
    "极速4步": "fast_4step",
    "参考图加速": "reference_fast",
    "参考高清（原生20步）": "ref_quality_native",
    "参考极速（官方4步）": "ref_fast_4step",
    "低显存": "low_vram",
    "低显存二采": "low_vram_two_stage",
    "自定义": "custom",
    "quality": "quality",
    "quality_sage": "quality_sage",
    "quality_two_stage": "quality_two_stage",
    "fl_quality_fast_v4": "fl_quality_fast_v4",
    "fast_4step": "fast_4step",
    "reference_fast": "reference_fast",
    "ref_quality_native": "ref_quality_native",
    "ref_fast_4step": "ref_fast_4step",
    "low_vram": "low_vram",
    "low_vram_two_stage": "low_vram_two_stage",
    "custom": "custom",
    "smart_free_1080p": "smart_free_1080p",
}

USER_PERFORMANCE_PRESET_LABELS = (
    "免费智能 1080p",
    "稳定质量",
    "质量优先加速",
    "质量优先二采样",
    "高清快速（v4 8步）",
    "极速4步",
    "参考图加速",
    "参考高清（原生20步）",
    "参考极速（官方4步）",
    "低显存",
    "低显存二采",
    "自定义",
)

PERFORMANCE_PRESET_LABELS_BY_KEY = {
    "smart_free_1080p": "免费智能 1080p",
    "quality": "稳定质量",
    "quality_sage": "质量优先加速",
    "quality_two_stage": "质量优先二采样",
    "fl_quality_fast_v4": "高清快速（v4 8步）",
    "fast_4step": "极速4步",
    "reference_fast": "参考图加速",
    "ref_quality_native": "参考高清（原生20步）",
    "ref_fast_4step": "参考极速（官方4步）",
    "low_vram": "低显存",
    "low_vram_two_stage": "低显存二采",
    "custom": "自定义",
}

REFERENCE_UNSAFE_FALLBACKS = {
    # quality_two_stage is now safe on REF2VA: the U22-validated turbo v4
    # 8+4 recipe drives the trained latent redraw while the AV split/concat
    # contract preserves the audio (voice) latent.
    "reference_fast": "quality_sage",
    "fl_quality_fast_v4": "quality_sage",
    "low_vram_two_stage": "low_vram",
}

PERFORMANCE_PRESETS_BY_ROUTE = {
    # The official H3 Turbo LoRA ships with a T2V example workflow and is
    # compatible with the same FL2VA model endpoint used by T2VA/FL2VA/I2VA.
    "t2va": ("smart_free_1080p", "quality", "quality_sage", "quality_two_stage", "fl_quality_fast_v4", "fast_4step", "low_vram", "low_vram_two_stage"),
    "endpoint": ("smart_free_1080p", "quality", "quality_sage", "quality_two_stage", "fl_quality_fast_v4", "fast_4step", "low_vram", "low_vram_two_stage"),
    "reference": ("smart_free_1080p", "quality", "quality_sage", "quality_two_stage", "ref_quality_native", "ref_fast_4step", "fast_4step", "low_vram", "custom"),
}
# The public selector now shows exactly one preset per route: 免费智能 1080p.
# It auto-resolves to the best verified path for the device — trained latent
# two-stage direct 1080p (U22 8+4 recipe) when VRAM and dependencies allow,
# otherwise 20-step SageAttention + SeedVR2/AI upscale, with a safe low-VRAM
# policy on small cards. The broader route map above is retained for loading
# older API payloads and saved workflows.
VISIBLE_PERFORMANCE_PRESETS_BY_ROUTE = {
    "t2va": ("smart_free_1080p",),
    "endpoint": ("smart_free_1080p",),
    "reference": ("smart_free_1080p",),
}

TWO_STAGE_PERFORMANCE_PRESETS = frozenset({"quality_two_stage", "low_vram_two_stage"})

# These routes are intentionally explicit. A quality two-stage pass already
# enlarges and redraws the H3 latent; each preset is paired with exactly one
# verified final reconstruction route. Other presets still expose one
# selectable final-output method at a time.
POSTPROCESS_MODES_BY_PERFORMANCE = {
    "smart_free_1080p": ("ai_upscale", "video_sr"),
    "quality_two_stage": ("video_sr", "rtx_vsr"),
    "low_vram_two_stage": ("ai_upscale",),
    "quality": POSTPROCESS_MODES,
    "quality_sage": POSTPROCESS_MODES,
    "fast_4step": POSTPROCESS_MODES,
    "reference_fast": POSTPROCESS_MODES,
    "low_vram": POSTPROCESS_MODES,
    "custom": POSTPROCESS_MODES,
}

# The user-facing selector shows exactly one final-output route per preset:
# SeedVR2 diffusion video SR, the measured quality leader for AI-generated
# content. Everything above stays accepted for old workflows and API payloads;
# when SeedVR2 is not installed the director degrades video_sr to ai_upscale
# with a warning instead of failing.
VISIBLE_POSTPROCESS_MODES_BY_PERFORMANCE = {
    "smart_free_1080p": ("video_sr",),
    "quality_two_stage": ("video_sr",),
    "low_vram_two_stage": ("ai_upscale",),
}

RTX_QUALITIES_BY_PERFORMANCE = {
    "quality_two_stage": ("HIGHBITRATE_ULTRA",),
}


def allowed_postprocess_modes(performance_preset):
    """Return final-output methods compatible with a normalized preset."""
    preset = PERFORMANCE_PRESETS.get(performance_preset, performance_preset)
    return POSTPROCESS_MODES_BY_PERFORMANCE.get(preset, POSTPROCESS_MODES)


def visible_postprocess_modes(performance_preset):
    """Return the single curated final-output route shown in the selector."""
    preset = PERFORMANCE_PRESETS.get(performance_preset, performance_preset)
    return VISIBLE_POSTPROCESS_MODES_BY_PERFORMANCE.get(preset, ("video_sr",))


def allowed_rtx_qualities(performance_preset):
    """Return RTX VSR qualities compatible with a normalized preset."""
    preset = PERFORMANCE_PRESETS.get(performance_preset, performance_preset)
    return RTX_QUALITIES_BY_PERFORMANCE.get(preset, ("HIGH", "ULTRA"))


def allowed_motion_smoothing(performance_preset, postprocess_mode):
    """Return resolved motion-smoothing paths compatible with this route."""
    preset = PERFORMANCE_PRESETS.get(performance_preset, performance_preset)
    if preset in {"smart_free_1080p", "low_vram", *TWO_STAGE_PERFORMANCE_PRESETS} or postprocess_mode != "rtx_vsr":
        return ("off",)
    return ("off", "rife_x2")


def allowed_performance_presets(mode, voice_mode="none"):
    """Return the safe, user-facing presets for the active H3 route."""
    if voice_mode != "none" or mode == "REF2VA":
        presets = PERFORMANCE_PRESETS_BY_ROUTE["reference"]
        if voice_mode == "fish_lock":
            # Fish S2 and the trained latent redraw are mutually exclusive.
            presets = tuple(
                preset for preset in presets
                if preset not in TWO_STAGE_PERFORMANCE_PRESETS
            )
        return presets
    if mode == "T2VA":
        return PERFORMANCE_PRESETS_BY_ROUTE["t2va"]
    return PERFORMANCE_PRESETS_BY_ROUTE["endpoint"]


def visible_performance_presets(mode, voice_mode="none"):
    """Return the curated presets shown by the user-facing selector."""
    if voice_mode != "none" or mode == "REF2VA":
        return VISIBLE_PERFORMANCE_PRESETS_BY_ROUTE["reference"]
    if mode == "T2VA":
        return VISIBLE_PERFORMANCE_PRESETS_BY_ROUTE["t2va"]
    return VISIBLE_PERFORMANCE_PRESETS_BY_ROUTE["endpoint"]


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
    "voice_gender": "auto",
    "target_dialogue": "",
    "reference_transcript": "",
    "fish_model_path": "s2-pro-w4a16 (auto download)",
    "ref_image_size": "match",
    "performance_preset": "smart_free_1080p",
    "postprocess_mode": "video_sr",
    "rtx_quality": "HIGH",
    "ai_upscale_model": "auto",
    "motion_smoothing": "off",
    "audio_loudness": "auto",
    "ignored_media": [],
    "postprocess": {},
    "output": {},
}


def normalize_request(raw=None):
    request = deepcopy(DEFAULT_REQUEST)
    request.update(dict(raw or {}))

    if not isinstance(request["references"], (list, tuple)):
        raise RequestError("额外参考图必须是列表")
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
    request["voice_gender"] = str(request.get("voice_gender") or "auto").strip().lower()
    if request["voice_gender"] not in {"auto", "male", "female", "neutral"}:
        request["voice_gender"] = "auto"

    if request["mode"] not in MODES:
        raise RequestError(f"不支持的生成模式：{request['mode']}")
    if request["voice_mode"] not in VOICE_MODES:
        raise RequestError(f"不支持的音色模式：{request['voice_mode']}")
    if request["postprocess_mode"] not in POSTPROCESS_MODES:
        raise RequestError(f"不支持的后处理模式：{request['postprocess_mode']}")
    if request["rtx_quality"] not in RTX_QUALITIES:
        raise RequestError(f"不支持的 RTX VSR 质量：{request['rtx_quality']}")

    # A saved Director node keeps hidden upload widgets when the user switches
    # modes.  Enforce the native H3 media contract here so stale endpoints can
    # never become an accidental first/last frame or an ignored extra input.
    ignored_media = list(request.get("ignored_media") or [])

    def discard(field, label):
        if _present(request.get(field)):
            ignored_media.append(label)
            request[field] = None

    if request["mode"] == "T2VA":
        discard("first_image", "首帧图片")
        discard("last_image", "尾帧图片")
        if request["references"]:
            ignored_media.append("参考图")
            request["references"] = []
    elif request["mode"] == "I2VA":
        discard("last_image", "尾帧图片")
        if request["references"]:
            ignored_media.append("参考图")
            request["references"] = []
    elif request["mode"] == "L2VA":
        discard("first_image", "首帧图片")
        if request["references"]:
            ignored_media.append("参考图")
            request["references"] = []
    elif request["mode"] == "FL2VA" and request["references"]:
        ignored_media.append("参考图")
        request["references"] = []
    request["ignored_media"] = ignored_media

    reference_total = len(request["references"]) + int(_present(request["first_image"])) + int(_present(request["last_image"]))
    if reference_total > 9:
        raise RequestError("首帧、尾帧和额外参考图合计最多支持 9 张")
    request["rtx_quality_requested"] = request["rtx_quality"]
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
    # Reverse check: a prompt that binds <Audio N> without any uploaded voice
    # reference makes H3 invent a voice from the text prior — the run is
    # guaranteed to miss the intended timbre, so fail before sampling starts.
    if re.search(r"<Audio\s*\d+>", str(request.get("prompt") or "")) and not audio_references:
        raise RequestError(
            "提示词引用了 <Audio N> 音色标记，但没有绑定任何音色参考音频"
            "（音色模式为“不使用音色”或未上传样本），H3 将凭空生成嗓音。"
            "请将音色模式切换为 H3 原生参考或 Fish 高级音色锁定并上传样本，"
            "或删除提示词中的 <Audio N> 标记。"
        )
    if request["voice_mode"] == "fish_lock" and not str(request["target_dialogue"]).strip():
        raise RequestError("Fish 高级音色锁定需要目标对白")
    preset = PERFORMANCE_PRESETS.get(request["performance_preset"])
    if preset is None:
        raise RequestError(f"不支持的性能预设：{request['performance_preset']}")
    request["performance_preset"] = preset

    request["warnings"] = []
    if request["voice_mode"] == "h3_reference" and request["voice_gender"] in {"male", "female"}:
        request["warnings"].append(
            "H3 原生音色参考仅提供性别/音域约束，不保证严格声纹一致；要尽量保持本人声线请使用 Fish S2。"
        )
    if ignored_media:
        request["warnings"].append(
            f"已忽略与 {request['mode']} / 当前音色设置不兼容的素材输入：{'、'.join(ignored_media)}。"
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
    if request["voice_mode"] == "h3_reference" and audio_references:
        audio_markers = tuple(f"<Audio {index}>" for index in range(1, len(audio_references) + 1))
        if not any(marker in str(request.get("prompt") or "") for marker in audio_markers):
            request["warnings"].append(
                "已上传音色参考，但提示词没有把 <Audio 1> 等音色绑定到角色对白；请明确写出“角色使用 <Audio 1> 的音色说：……”否则可能只生成环境声。"
            )
    if request["resolved_backend"] == "ref2va_model":
        fallback = REFERENCE_UNSAFE_FALLBACKS.get(preset)
        if fallback:
            request["performance_preset"] = fallback
            request["warnings"].append(
                f"{PERFORMANCE_PRESET_LABELS_BY_KEY[preset]} ({preset}) 已回退为 "
                f"{PERFORMANCE_PRESET_LABELS_BY_KEY[fallback]} ({fallback})。"
            )
            if preset == "quality_two_stage" and request["rtx_quality"] == "HIGHBITRATE_ULTRA":
                request["rtx_quality"] = "HIGH"
                request["warnings"].append(
                    "质量优先二采样专属 HIGHBITRATE_ULTRA 已随 REF 回退协调为 HIGH。"
                )

    resolved_preset = request["performance_preset"]
    allowed = allowed_performance_presets(request["mode"], request["voice_mode"])
    if resolved_preset not in allowed and resolved_preset != "custom":
        request["performance_preset"] = "quality"
        resolved_preset = "quality"
        request["warnings"].append(
            f"{request['mode']} / {request['voice_mode']} 不支持性能预设 {preset}，已自动切换为稳定质量。"
        )

    allowed_postprocess = allowed_postprocess_modes(resolved_preset)
    if request["postprocess_mode"] not in allowed_postprocess:
        if resolved_preset == "quality_two_stage":
            raise RequestError(
                "质量优先二采样已包含 H3 latent 放大重绘，只能搭配 SeedVR2 视频超分或 RTX VSR；"
                "请将最终输出切换为 SeedVR2 视频超分（推荐）或 RTX VSR"
            )
        if resolved_preset == "low_vram_two_stage":
            raise RequestError(
                "低显存二采只能搭配 AI 自动超分的 X2 细节重建；"
                "请将最终输出切换为 AI 自动超分"
            )
        raise RequestError(
            f"性能预设 {request['performance_preset']} 不支持后处理模式 "
            f"{request['postprocess_mode']}"
        )

    allowed_rtx = allowed_rtx_qualities(resolved_preset)
    if resolved_preset == "quality_two_stage" and request["postprocess_mode"] == "rtx_vsr":
        if request["rtx_quality"] != "HIGHBITRATE_ULTRA":
            request["warnings"].append(
                "质量优先二采样使用干净的 H3 VAE 输出，RTX VSR 已自动切换为 "
                "HIGHBITRATE_ULTRA。"
            )
        request["rtx_quality"] = "HIGHBITRATE_ULTRA"
    elif resolved_preset == "quality_two_stage":
        pass  # SeedVR2 路线不经过 RTX VSR，无需校验 RTX 质量档位
    elif request["rtx_quality"] not in allowed_rtx:
        raise RequestError(
            "HIGHBITRATE_ULTRA 仅用于质量优先二采样；"
            "其他性能预设请选择 HIGH 或 ULTRA"
        )
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
        if resolved_preset == "low_vram_two_stage":
            raise RequestError("低显存二采固定关闭 RIFE 运动平滑，以避免增加显存峰值与重影")
        if resolved_preset == "low_vram":
            raise RequestError("低显存模式不支持 RIFE 运动平滑，请将运动平滑切换为关闭")
        raise RequestError("RIFE 2x 运动平滑只能搭配 RTX VSR 最终输出")

    return request


def public_schema():
    def labels_for(mode, voice_mode="none"):
        return [
            PERFORMANCE_PRESET_LABELS_BY_KEY[preset]
            for preset in visible_performance_presets(mode, voice_mode)
        ]

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
            "voice_gender": {"中文名称": "音色性别约束", "enum": ["auto", "male", "female", "neutral"], "default": "auto"},
            "target_dialogue": {"中文名称": "目标对白", "type": "string", "default": ""},
            "reference_transcript": {"中文名称": "音色样本文本", "type": "string", "default": ""},
            "fish_model_path": {"中文名称": "Fish S2 模型", "type": "string", "default": "s2-pro-w4a16 (auto download)"},
            "ref_image_size": {"中文名称": "参考图尺寸策略", "enum": ["match", "max"], "default": "match"},
            "performance_preset": {
                "中文名称": "性能预设",
                "enum": list(USER_PERFORMANCE_PRESET_LABELS),
                "default": "免费智能 1080p",
                "allowed_by_route": {
                    "T2VA": labels_for("T2VA"),
                    "I2VA / FL2VA / L2VA": labels_for("I2VA"),
                    "REF2VA": labels_for("REF2VA"),
                    "I2VA + 音色参考": labels_for("I2VA", "h3_reference"),
                    "FL2VA + 音色参考": labels_for("FL2VA", "h3_reference"),
                    "L2VA + 音色参考": labels_for("L2VA", "h3_reference"),
                    "T2VA + 音色参考": labels_for("T2VA", "h3_reference"),
                },
            },
            "postprocess_mode": {
                "中文名称": "最终输出后处理模式",
                "enum": list(POSTPROCESS_MODES),
                "default": "video_sr",
                "description": "导演台只展示一条最清晰路线：SeedVR2 扩散视频超分（7B sharp 优先，时间一致性最好）；未安装 SeedVR2 时自动回退通用 AI 超分并给出警告。原生直出、Lanczos、RTX VSR 等历史值仅为旧工作流兼容保留。",
                "allowed_by_performance": {
                    "免费智能 1080p": ["video_sr"],
                    "质量优先二采样": ["video_sr"],
                    "低显存二采": ["ai_upscale"],
                    "其他性能预设": ["video_sr"],
                },
            },
            "rtx_quality": {
                "中文名称": "RTX VSR 质量",
                "enum": list(RTX_QUALITIES),
                "default": "HIGH",
                "description": "RTX VSR 的质量级别，仅在后处理模式为 rtx_vsr 时生效。质量优先二采样固定使用高码率原画源档。",
                "allowed_by_performance": {
                    "质量优先二采样": ["HIGHBITRATE_ULTRA"],
                    "其他性能预设": ["HIGH", "ULTRA"],
                },
            },
            "ai_upscale_model": {
                "中文名称": "通用 AI 超分模型",
                "type": "string",
                "default": "auto",
                "description": "默认 auto 按实际放大倍率自动选择 models/upscale_models 中的 X2/X4 模型（≤2 倍优先 X2，避免 X4 放大再缩回的浪费），仅在 ai_upscale 模式生效。",
            },
            "motion_smoothing": {
                "中文名称": "运动平滑",
                "enum": list(MOTION_SMOOTHING_MODES),
                "default": "off",
                "description": "默认关闭以保留 H3 原始帧；可手动启用流式 RIFE 2x，旧 auto 值按关闭处理。质量优先二采样和低显存模式只允许关闭。",
                "allowed_by_performance": {
                    "免费智能 1080p + AI 自动超分": ["auto", "off"],
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
            "final_upscale_scale": {"中文名称": "最终最大放大倍率", "type": "number", "readOnly": True},
            "max_final_vsr_scale": {"中文名称": "低显存清晰度倍率上限", "type": ["number", "null"], "readOnly": True},
            "vram_safety_tier": {"中文名称": "显存安全档位", "type": "string", "readOnly": True},
            "quality_basis": {"中文名称": "清晰度基础", "type": "string", "readOnly": True},
            "required_assets": {"中文名称": "本次所需模型资产", "type": "array", "readOnly": True},
        },
    }
