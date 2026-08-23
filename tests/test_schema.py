import pytest
import nodes.schema as schema_module

from nodes.schema import (
    PUBLIC_API_KEYS,
    RequestError,
    allowed_performance_presets,
    allowed_postprocess_modes,
    low_vram_target_limit,
    normalize_request,
    public_schema,
)


def test_public_schema_lists_every_public_api_key():
    assert set(PUBLIC_API_KEYS) <= set(public_schema()["properties"])


def test_schema_exposes_final_postprocess_controls():
    schema = public_schema()["properties"]

    assert schema["postprocess_mode"]["enum"] == ["native", "lanczos", "ai_upscale", "rtx_vsr"]
    assert schema["rtx_quality"]["enum"] == ["HIGH", "ULTRA"]
    assert schema["ai_upscale_model"]["default"] == "auto"
    assert schema["postprocess_mode"]["allowed_by_performance"]["质量优先二采样"] == ["rtx_vsr"]
    assert schema["motion_smoothing"]["enum"] == ["auto", "off", "rife_x2"]
    assert schema["motion_smoothing"]["default"] == "off"
    assert schema["audio_loudness"]["enum"] == ["auto", "original"]
    assert schema["audio_loudness"]["default"] == "auto"


def test_normalize_request_defaults_to_native_postprocess():
    request = normalize_request({"mode": "T2VA", "duration": 4})

    assert request["postprocess_mode"] == "native"
    assert request["rtx_quality"] == "HIGH"
    assert request["ai_upscale_model"] == "auto"
    assert request["motion_smoothing"] == "off"
    assert request["audio_loudness"] == "auto"


def test_normalize_request_accepts_generic_upscale_model_override():
    request = normalize_request({
        "mode": "T2VA",
        "postprocess_mode": "ai_upscale",
        "ai_upscale_model": "RealESRGAN_x2plus.pth",
    })

    assert request["postprocess_mode"] == "ai_upscale"
    assert request["ai_upscale_model"] == "RealESRGAN_x2plus.pth"


def test_quality_two_stage_only_allows_rtx_vsr_postprocess():
    assert allowed_postprocess_modes("quality_two_stage") == ("rtx_vsr",)


def test_public_schema_exposes_read_only_two_stage_execution_metadata():
    resolved = public_schema()["resolved_outputs"]
    assert resolved["resolved_two_stage_route"]["中文名称"] == "实际训练型二采路线"
    assert resolved["first_stage_width"]["中文名称"] == "H3首采宽度"
    assert resolved["second_stage_width"]["中文名称"] == "神经latent二采宽度"
    assert resolved["final_upscale_scale_x"]["中文名称"] == "最终横向放大倍率"
    assert resolved["vram_safety_tier"]["中文名称"] == "显存安全档位"
    assert resolved["quality_basis"]["中文名称"] == "清晰度基础"
    assert resolved["required_assets"]["中文名称"] == "本次所需模型资产"


@pytest.mark.parametrize("postprocess_mode", ["native", "lanczos", "ai_upscale"])
def test_quality_two_stage_rejects_incompatible_postprocess(postprocess_mode):
    with pytest.raises(RequestError, match="质量优先二采样.*RTX VSR"):
        normalize_request({
            "mode": "T2VA",
            "performance_preset": "质量优先二采样",
            "postprocess_mode": postprocess_mode,
        })


def test_quality_two_stage_accepts_rtx_vsr_postprocess():
    request = normalize_request({
        "mode": "T2VA",
        "performance_preset": "质量优先二采样",
        "postprocess_mode": "rtx_vsr",
        "rtx_quality": "ULTRA",
    })
    assert request["performance_preset"] == "quality_two_stage"
    assert request["postprocess_mode"] == "rtx_vsr"


def test_legacy_auto_motion_smoothing_resolves_to_off():
    request = normalize_request({
        "mode": "T2VA",
        "performance_preset": "质量优先二采样",
        "postprocess_mode": "rtx_vsr",
        "motion_smoothing": "auto",
    })

    assert request["motion_smoothing"] == "off"
    allowed_motion_smoothing = getattr(schema_module, "allowed_motion_smoothing", None)
    assert callable(allowed_motion_smoothing), "缺少运动平滑兼容矩阵"
    assert allowed_motion_smoothing("quality_two_stage", "rtx_vsr") == ("off",)


def test_quality_two_stage_rejects_forced_rife_motion_smoothing():
    with pytest.raises(RequestError, match="质量优先二采样.*RIFE"):
        normalize_request({
            "mode": "T2VA",
            "performance_preset": "质量优先二采样",
            "postprocess_mode": "rtx_vsr",
            "motion_smoothing": "rife_x2",
        })


def test_normalize_request_rejects_unknown_audio_loudness_mode():
    with pytest.raises(RequestError, match="音频响度"):
        normalize_request({"mode": "T2VA", "audio_loudness": "maximum"})


def test_normal_preset_auto_motion_smoothing_stays_off():
    request = normalize_request({
        "mode": "T2VA",
        "performance_preset": "稳定质量",
        "postprocess_mode": "rtx_vsr",
        "motion_smoothing": "auto",
    })

    assert request["motion_smoothing"] == "off"


def test_low_vram_rejects_forced_rife_motion_smoothing():
    with pytest.raises(RequestError, match="低显存.*运动平滑"):
        normalize_request({
            "mode": "T2VA",
            "performance_preset": "低显存",
            "postprocess_mode": "rtx_vsr",
            "motion_smoothing": "rife_x2",
        })


def test_rife_motion_smoothing_requires_rtx_vsr():
    with pytest.raises(RequestError, match="RIFE 2x.*RTX VSR"):
        normalize_request({
            "mode": "T2VA",
            "performance_preset": "稳定质量",
            "postprocess_mode": "native",
            "motion_smoothing": "rife_x2",
        })


def test_normalize_request_rejects_unknown_postprocess_mode():
    with pytest.raises(RequestError, match="后处理模式"):
        normalize_request({"mode": "T2VA", "postprocess_mode": "magic"})


def test_normalize_request_rejects_unknown_rtx_quality():
    with pytest.raises(RequestError, match="RTX VSR 质量"):
        normalize_request({"mode": "T2VA", "rtx_quality": "MEDIUM"})


@pytest.mark.parametrize(
    ("duration", "expected"),
    [
        (4, (1920, 1080)),
        (6, (1920, 1080)),
        (7, (1920, 1080)),
        (8, (1920, 1080)),
        (15, (1920, 1080)),
    ],
)
def test_low_vram_target_limit_follows_duration(duration, expected):
    assert low_vram_target_limit(duration) == expected


@pytest.mark.parametrize("duration", [3, 16])
def test_low_vram_target_limit_rejects_duration_outside_request_range(duration):
    with pytest.raises(RequestError, match="4 到 15"):
        low_vram_target_limit(duration)


def test_public_schema_exposes_fish_model_choice():
    schema = public_schema()
    assert "fish_model_path" in PUBLIC_API_KEYS
    assert schema["properties"]["fish_model_path"]["中文名称"] == "Fish S2 模型"


def test_public_schema_documents_route_performance_options():
    property_schema = public_schema()["properties"]["performance_preset"]
    assert "质量优先二采样" in property_schema["enum"]
    assert "自定义" in property_schema["enum"]
    assert property_schema["allowed_by_route"]["T2VA"] == ["稳定质量", "质量优先加速", "质量优先二采样", "极速4步", "低显存"]
    assert property_schema["allowed_by_route"]["I2VA + 音色参考"] == [
        "稳定质量", "质量优先加速", "质量优先二采样", "参考图加速", "极速4步", "低显存"
    ]


class TensorLikeImage:
    def __bool__(self):
        raise RuntimeError("ambiguous tensor boolean")

    def __len__(self):
        return 1


def test_tensor_like_image_is_accepted_without_boolean_coercion():
    request = normalize_request({"mode": "I2VA", "first_image": TensorLikeImage()})
    assert request["resolved_backend"] == "fl2va_model"


def test_fl2va_without_voice_keeps_fl_backend():
    request = normalize_request({
        "mode": "FL2VA",
        "first_image": "opening.png",
        "last_image": "closing.png",
    })

    assert request["resolved_backend"] == "fl2va_model"


def test_i2va_with_h3_reference_uses_ref_backend():
    request = normalize_request({
        "mode": "I2VA",
        "first_image": "opening.png",
        "voice_mode": "h3_reference",
        "voice_reference_audio": "voice.wav",
    })

    assert request["resolved_backend"] == "ref2va_model"


def test_fl2va_with_fish_lock_uses_ref_backend():
    request = normalize_request({
        "mode": "FL2VA",
        "first_image": "opening.png",
        "last_image": "closing.png",
        "voice_mode": "fish_lock",
        "voice_reference_audio": "voice.wav",
        "target_dialogue": "我们回家。",
    })

    assert request["resolved_backend"] == "ref2va_model"


def test_fish_lock_requires_target_dialogue():
    with pytest.raises(RequestError, match="目标对白"):
        normalize_request({
            "mode": "I2VA",
            "first_image": "opening.png",
            "voice_mode": "fish_lock",
            "voice_reference_audio": "voice.wav",
        })


def test_i2va_requires_first_image():
    with pytest.raises(RequestError, match="首帧图片"):
        normalize_request({"mode": "I2VA"})


def test_reference_mode_never_accepts_copy_semantics():
    with pytest.raises(RequestError, match="音色模式"):
        normalize_request({"mode": "REF2VA", "voice_mode": "fully_copy"})


@pytest.mark.parametrize(
    ("preset", "expected"),
    [
        ("稳定质量", "quality"),
        ("极速4步", "fast_4step"),
        ("参考图加速", "reference_fast"),
        ("质量优先加速", "quality_sage"),
        ("质量优先二采样", "quality_two_stage"),
        ("低显存", "low_vram"),
        ("自定义", "custom"),
    ],
)
def test_chinese_performance_presets_normalize_to_stable_keys(preset, expected):
    request = normalize_request({
        "mode": "REF2VA",
        "performance_preset": preset,
        "postprocess_mode": "rtx_vsr" if expected == "quality_two_stage" else "native",
    })

    assert request["performance_preset"] == expected


@pytest.mark.parametrize(
    ("mode", "voice_mode", "expected"),
    [
        ("T2VA", "none", ("quality", "quality_sage", "quality_two_stage", "fast_4step", "low_vram")),
        ("I2VA", "none", ("quality", "quality_sage", "quality_two_stage", "fast_4step", "low_vram")),
        ("FL2VA", "none", ("quality", "quality_sage", "quality_two_stage", "fast_4step", "low_vram")),
        ("L2VA", "none", ("quality", "quality_sage", "quality_two_stage", "fast_4step", "low_vram")),
        ("REF2VA", "none", ("quality", "quality_sage", "quality_two_stage", "reference_fast", "fast_4step", "low_vram")),
        ("I2VA", "h3_reference", ("quality", "quality_sage", "quality_two_stage", "reference_fast", "fast_4step", "low_vram")),
        ("FL2VA", "fish_lock", ("quality", "quality_sage", "reference_fast", "fast_4step", "low_vram")),
        ("L2VA", "h3_reference", ("quality", "quality_sage", "quality_two_stage", "reference_fast", "fast_4step", "low_vram")),
        ("T2VA", "h3_reference", ("quality", "quality_sage", "quality_two_stage", "reference_fast", "fast_4step", "low_vram")),
    ],
)
def test_allowed_performance_presets_follow_mode_and_voice(mode, voice_mode, expected):
    assert allowed_performance_presets(mode, voice_mode) == expected


def test_invalid_t2va_reference_acceleration_falls_back_with_warning():
    request = normalize_request({"mode": "T2VA", "performance_preset": "reference_fast"})
    assert request["performance_preset"] == "quality"
    assert any("T2VA" in warning and "稳定质量" in warning for warning in request["warnings"])


def test_voice_reference_uses_ref2va_performance_set_even_in_fl2va():
    request = normalize_request({
        "mode": "FL2VA",
        "first_image": "opening.png",
        "voice_mode": "h3_reference",
        "voice_reference_audio": "voice.wav",
    })
    assert request["resolved_backend"] == "ref2va_model"
    assert "reference_fast" in allowed_performance_presets("FL2VA", "h3_reference")


def test_duration_uses_h3_native_four_to_fifteen_second_range():
    assert normalize_request({"mode": "T2VA", "duration": 4})["duration"] == 4
    assert normalize_request({"mode": "T2VA", "duration": 15})["duration"] == 15
    with pytest.raises(RequestError, match="4 到 15"):
        normalize_request({"mode": "T2VA", "duration": 3})


def test_fish_voice_mode_rejects_unused_secondary_reference_audio():
    with pytest.raises(RequestError, match="Fish"):
        normalize_request({
            "mode": "T2VA",
            "voice_mode": "fish_lock",
            "voice_reference_audios": ["voice-1", "voice-2"],
            "target_dialogue": "对白",
        })


def test_reference_limits_match_h3_native_caps():
    assert len(normalize_request({"mode": "REF2VA", "references": list(range(9))})["references"]) == 9
    with pytest.raises(RequestError, match="最多支持 9 张"):
        normalize_request({"mode": "REF2VA", "references": list(range(10))})
    with pytest.raises(RequestError, match="最多支持 3 路"):
        normalize_request({"mode": "REF2VA", "voice_reference_audios": list(range(4))})
