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


def test_two_stage_presets_remain_available_only_on_non_reference_routes():
    assert "low_vram" in allowed_performance_presets("T2VA", "none")
    assert "low_vram_two_stage" in allowed_performance_presets("T2VA", "none")
    assert "low_vram_two_stage" not in allowed_performance_presets("REF2VA", "h3_reference")
    assert "low_vram_two_stage" not in allowed_performance_presets("T2VA", "fish_lock")


def test_schema_exposes_final_postprocess_controls():
    schema = public_schema()["properties"]

    assert schema["postprocess_mode"]["enum"] == ["native", "lanczos", "ai_upscale", "video_sr", "rtx_vsr"]
    assert schema["rtx_quality"]["enum"] == ["HIGH", "ULTRA", "HIGHBITRATE_ULTRA"]
    assert schema["rtx_quality"]["allowed_by_performance"] == {
        "质量优先二采样": ["HIGHBITRATE_ULTRA"],
        "其他性能预设": ["HIGH", "ULTRA"],
    }
    assert schema["ai_upscale_model"]["default"] == "4x-UltraSharpV2.safetensors"
    assert schema["postprocess_mode"]["allowed_by_performance"]["质量优先二采样"] == ["rtx_vsr"]
    assert schema["postprocess_mode"]["allowed_by_performance"]["低显存二采"] == ["ai_upscale"]
    assert schema["motion_smoothing"]["enum"] == ["auto", "off", "rife_x2"]
    assert schema["motion_smoothing"]["default"] == "off"
    assert schema["audio_loudness"]["enum"] == ["auto", "original"]
    assert schema["audio_loudness"]["default"] == "auto"


def test_normalize_request_defaults_to_smart_free_1080p():
    request = normalize_request({"mode": "T2VA", "duration": 4})

    assert request["performance_preset"] == "smart_free_1080p"
    assert request["postprocess_mode"] == "ai_upscale"
    assert request["rtx_quality"] == "HIGH"
    assert request["ai_upscale_model"] == "4x-UltraSharpV2.safetensors"
    assert request["motion_smoothing"] == "off"
    assert request["audio_loudness"] == "auto"
    assert request["voice_gender"] == "auto"


def test_h3_reference_preserves_explicit_voice_gender_constraint_with_warning():
    request = normalize_request({
        "mode": "REF2VA",
        "voice_mode": "h3_reference",
        "voice_gender": "male",
        "voice_reference_audio": object(),
    })

    assert request["voice_gender"] == "male"
    assert any("性别/音域约束" in warning and "Fish S2" in warning for warning in request["warnings"])


def test_h3_reference_warns_when_prompt_does_not_bind_audio_to_dialogue():
    request = normalize_request({
        "mode": "REF2VA",
        "prompt": "安静的室内环境声，门铃保持静音。",
        "voice_mode": "h3_reference",
        "voice_reference_audio": object(),
    })

    assert any("<Audio 1>" in warning and "对白" in warning for warning in request["warnings"])


def test_normalize_request_normalizes_unknown_voice_gender_to_auto():
    """Unknown voice_gender values (from widget misalignment) normalize to auto."""
    request = normalize_request({"mode": "T2VA", "duration": 4, "voice_gender": "baritone"})
    assert request["voice_gender"] == "auto"


def test_normalize_request_normalizes_shifted_voice_gender_from_widget_misalignment():
    """Old saved workflows may shift widget values by position, causing voice_gender
    to receive values from other fields like fish_model_path.  These must normalize
    to auto instead of raising, so the node output is not silently ignored."""
    request = normalize_request({"mode": "T2VA", "duration": 4, "voice_gender": "s2-pro-w4a16 (auto download)"})
    assert request["voice_gender"] == "auto"


def test_normalize_request_normalizes_blank_voice_gender_from_old_workflow():
    """ComfyUI strips empty strings from combo lists, so the cf1a6c2 trick of
    adding empty string to the combo did not work.  A blank value must normalize to auto."""
    request = normalize_request({"mode": "T2VA", "duration": 4, "voice_gender": ""})
    assert request["voice_gender"] == "auto"


@pytest.mark.parametrize(
    ("mode", "expected_first", "expected_last", "expected_refs"),
    [
        ("T2VA", False, False, 0),
        ("I2VA", True, False, 0),
        ("FL2VA", True, True, 0),
        ("L2VA", False, True, 0),
        ("REF2VA", True, True, 1),
    ],
)
def test_normalize_request_drops_media_incompatible_with_selected_mode(
    mode, expected_first, expected_last, expected_refs
):
    request = normalize_request({
        "mode": mode,
        "first_image": object(),
        "last_image": object(),
        "references": [object()],
    })

    assert bool(request["first_image"]) is expected_first
    assert bool(request["last_image"]) is expected_last
    assert len(request["references"]) == expected_refs
    if mode != "REF2VA":
        assert any("不兼容" in warning for warning in request["warnings"])


def test_normalize_request_accepts_generic_upscale_model_override():
    request = normalize_request({
        "mode": "T2VA",
        "postprocess_mode": "ai_upscale",
        "ai_upscale_model": "RealESRGAN_x2plus.pth",
    })

    assert request["postprocess_mode"] == "ai_upscale"
    assert request["ai_upscale_model"] == "RealESRGAN_x2plus.pth"


def test_smart_free_1080p_aliases_and_output_controls_are_locked():
    for preset in ("免费智能 1080p", "smart_free_1080p"):
        request = normalize_request({"mode": "T2VA", "performance_preset": preset})
        assert request["performance_preset"] == "smart_free_1080p"
    assert allowed_postprocess_modes("smart_free_1080p") == ("ai_upscale", "video_sr")
    assert schema_module.allowed_motion_smoothing("smart_free_1080p", "ai_upscale") == ("off",)

    with pytest.raises(RequestError, match="smart_free_1080p.*后处理模式"):
        normalize_request({
            "mode": "T2VA",
            "performance_preset": "smart_free_1080p",
            "postprocess_mode": "native",
        })
    with pytest.raises(RequestError, match="RIFE"):
        normalize_request({
            "mode": "T2VA",
            "performance_preset": "smart_free_1080p",
            "motion_smoothing": "rife_x2",
        })


def test_quality_two_stage_only_allows_rtx_vsr_postprocess():
    assert allowed_postprocess_modes("quality_two_stage") == ("rtx_vsr",)
    allowed_rtx_qualities = getattr(schema_module, "allowed_rtx_qualities", None)
    assert callable(allowed_rtx_qualities), "缺少 RTX VSR 质量兼容矩阵"
    assert allowed_rtx_qualities("quality_two_stage") == ("HIGHBITRATE_ULTRA",)
    assert allowed_rtx_qualities("quality") == ("HIGH", "ULTRA")


def test_low_vram_two_stage_only_allows_ai_x2_reconstruction_without_rife():
    assert allowed_postprocess_modes("low_vram_two_stage") == ("ai_upscale",)
    assert schema_module.allowed_rtx_qualities("low_vram_two_stage") == ("HIGH", "ULTRA")
    assert schema_module.allowed_motion_smoothing("low_vram_two_stage", "ai_upscale") == ("off",)

    request = normalize_request({
        "mode": "T2VA",
        "duration": 4,
        "performance_preset": "低显存二采",
        "postprocess_mode": "ai_upscale",
        "ai_upscale_model": "auto",
    })

    assert request["performance_preset"] == "low_vram_two_stage"
    assert request["postprocess_mode"] == "ai_upscale"
    assert request["ai_upscale_model"] == "auto"
    assert request["motion_smoothing"] == "off"


def test_low_vram_two_stage_rejects_non_ai_reconstruction_postprocess():
    with pytest.raises(RequestError, match="低显存二采.*AI 自动超分"):
        normalize_request({
            "mode": "T2VA",
            "duration": 4,
            "performance_preset": "低显存二采",
            "postprocess_mode": "native",
        })


def test_public_schema_exposes_read_only_two_stage_execution_metadata():
    resolved = public_schema()["resolved_outputs"]
    assert resolved["resolved_two_stage_route"]["中文名称"] == "实际训练型二采路线"
    assert resolved["first_stage_width"]["中文名称"] == "H3首采宽度"
    assert resolved["second_stage_width"]["中文名称"] == "神经latent二采宽度"
    assert resolved["final_upscale_scale_x"]["中文名称"] == "最终横向放大倍率"
    assert resolved["final_upscale_scale"]["中文名称"] == "最终最大放大倍率"
    assert resolved["max_final_vsr_scale"]["中文名称"] == "低显存清晰度倍率上限"
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
    assert request["rtx_quality_requested"] == "ULTRA"
    assert request["rtx_quality"] == "HIGHBITRATE_ULTRA"
    assert any("HIGHBITRATE_ULTRA" in warning for warning in request["warnings"])


def test_non_two_stage_rejects_exclusive_high_bitrate_quality():
    with pytest.raises(RequestError, match="HIGHBITRATE_ULTRA.*质量优先二采样"):
        normalize_request({
            "mode": "T2VA",
            "performance_preset": "稳定质量",
            "postprocess_mode": "rtx_vsr",
            "rtx_quality": "HIGHBITRATE_ULTRA",
        })


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
    assert property_schema["enum"][0] == "免费智能 1080p"
    assert property_schema["default"] == "免费智能 1080p"
    assert "质量优先二采样" in property_schema["enum"]
    assert "自定义" in property_schema["enum"]
    assert property_schema["allowed_by_route"]["T2VA"] == [
        "免费智能 1080p", "稳定质量"
    ]
    assert property_schema["allowed_by_route"]["I2VA + 音色参考"] == [
        "免费智能 1080p", "参考高清（原生20步）"
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
        ("免费智能 1080p", "smart_free_1080p"),
        ("高清快速（v4 8步）", "fl_quality_fast_v4"),
        ("参考高清（原生20步）", "ref_quality_native"),
        ("参考极速（官方4步）", "ref_fast_4step"),
        ("稳定质量", "quality"),
        ("极速4步", "fast_4step"),
        ("质量优先加速", "quality_sage"),
        ("质量优先二采样", "quality_two_stage"),
        ("低显存", "low_vram"),
        ("低显存二采", "low_vram_two_stage"),
        ("自定义", "custom"),
    ],
)
def test_chinese_performance_presets_normalize_to_stable_keys(preset, expected):
    request = normalize_request({
        "mode": "T2VA" if expected in {"quality_two_stage", "low_vram_two_stage", "fl_quality_fast_v4"} else "REF2VA",
        "performance_preset": preset,
        "postprocess_mode": (
            "rtx_vsr" if expected == "quality_two_stage"
            else "ai_upscale" if expected in {"smart_free_1080p", "low_vram_two_stage"}
            else "native"
        ),
    })

    assert request["performance_preset"] == expected


@pytest.mark.parametrize(
    ("mode", "voice_mode", "expected"),
    [
        ("T2VA", "none", ("smart_free_1080p", "quality", "quality_sage", "quality_two_stage", "fl_quality_fast_v4", "fast_4step", "low_vram", "low_vram_two_stage")),
        ("I2VA", "none", ("smart_free_1080p", "quality", "quality_sage", "quality_two_stage", "fl_quality_fast_v4", "fast_4step", "low_vram", "low_vram_two_stage")),
        ("FL2VA", "none", ("smart_free_1080p", "quality", "quality_sage", "quality_two_stage", "fl_quality_fast_v4", "fast_4step", "low_vram", "low_vram_two_stage")),
        ("L2VA", "none", ("smart_free_1080p", "quality", "quality_sage", "quality_two_stage", "fl_quality_fast_v4", "fast_4step", "low_vram", "low_vram_two_stage")),
        ("REF2VA", "none", ("smart_free_1080p", "quality", "quality_sage", "ref_quality_native", "ref_fast_4step", "fast_4step", "low_vram", "custom")),
        ("I2VA", "h3_reference", ("smart_free_1080p", "quality", "quality_sage", "ref_quality_native", "ref_fast_4step", "fast_4step", "low_vram", "custom")),
        ("FL2VA", "fish_lock", ("smart_free_1080p", "quality", "quality_sage", "ref_quality_native", "ref_fast_4step", "fast_4step", "low_vram", "custom")),
        ("L2VA", "h3_reference", ("smart_free_1080p", "quality", "quality_sage", "ref_quality_native", "ref_fast_4step", "fast_4step", "low_vram", "custom")),
        ("T2VA", "h3_reference", ("smart_free_1080p", "quality", "quality_sage", "ref_quality_native", "ref_fast_4step", "fast_4step", "low_vram", "custom")),
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
    assert "smart_free_1080p" in allowed_performance_presets("FL2VA", "h3_reference")
    assert "reference_fast" not in allowed_performance_presets("FL2VA", "h3_reference")


def test_route_specific_presets_are_not_cross_exposed():
    assert "fl_quality_fast_v4" in allowed_performance_presets("FL2VA", "none")
    assert "fl_quality_fast_v4" not in allowed_performance_presets("REF2VA", "none")
    assert "ref_quality_native" in allowed_performance_presets("REF2VA", "none")
    assert "ref_fast_4step" in allowed_performance_presets("REF2VA", "none")
    assert "ref_quality_native" not in allowed_performance_presets("FL2VA", "none")


def test_reference_routes_have_exact_safe_preset_set():
    allowed = set(allowed_performance_presets("REF2VA", "none"))
    assert allowed == {
        "smart_free_1080p", "quality", "quality_sage", "ref_quality_native",
        "ref_fast_4step", "fast_4step", "low_vram", "custom",
    }
    assert not allowed & {"quality_two_stage", "low_vram_two_stage", "reference_fast", "fl_quality_fast_v4"}


@pytest.mark.parametrize(
    ("preset", "fallback", "label", "fallback_label", "postprocess_mode"),
    [
        ("quality_two_stage", "quality_sage", "质量优先二采样", "质量优先加速", "rtx_vsr"),
        ("reference_fast", "quality_sage", "参考图加速", "质量优先加速", "native"),
        ("fl_quality_fast_v4", "quality_sage", "高清快速（v4 8步）", "质量优先加速", "native"),
        ("low_vram_two_stage", "low_vram", "低显存二采", "低显存", "ai_upscale"),
    ],
)
def test_ref2va_unsafe_legacy_preset_falls_back_after_backend_resolution(
    preset, fallback, label, fallback_label, postprocess_mode
):
    request = normalize_request({
        "mode": "REF2VA",
        "performance_preset": preset,
        "postprocess_mode": postprocess_mode,
    })
    assert request["resolved_backend"] == "ref2va_model"
    assert request["performance_preset"] == fallback
    assert any(preset in warning and fallback in warning for warning in request["warnings"])
    assert any(label in warning and fallback_label in warning for warning in request["warnings"])


@pytest.mark.parametrize(
    "payload",
    [
        {
            "mode": "REF2VA",
            "performance_preset": "quality_two_stage",
            "postprocess_mode": "rtx_vsr",
            "rtx_quality": "HIGHBITRATE_ULTRA",
        },
        {
            "mode": "I2VA",
            "first_image": "opening.png",
            "voice_mode": "h3_reference",
            "voice_reference_audio": "voice.wav",
            "performance_preset": "quality_two_stage",
            "postprocess_mode": "rtx_vsr",
            "rtx_quality": "HIGHBITRATE_ULTRA",
        },
    ],
)
def test_complete_legacy_reference_two_stage_state_migrates_to_repeatable_quality_sage(payload):
    request = normalize_request(payload)

    assert request["resolved_backend"] == "ref2va_model"
    assert request["performance_preset"] == "quality_sage"
    assert request["postprocess_mode"] == "rtx_vsr"
    assert request["rtx_quality"] == "HIGH"
    assert any("HIGHBITRATE_ULTRA" in warning and "HIGH" in warning for warning in request["warnings"])

    reloaded = normalize_request(request)
    assert {
        field: reloaded[field]
        for field in ("performance_preset", "postprocess_mode", "rtx_quality")
    } == {
        field: request[field]
        for field in ("performance_preset", "postprocess_mode", "rtx_quality")
    }


def test_h3_reference_backend_falls_back_unsafe_legacy_preset():
    request = normalize_request({
        "mode": "I2VA",
        "first_image": "opening.png",
        "voice_mode": "h3_reference",
        "voice_reference_audio": "voice.wav",
        "performance_preset": "quality_two_stage",
        "postprocess_mode": "rtx_vsr",
    })
    assert request["resolved_backend"] == "ref2va_model"
    assert request["performance_preset"] == "quality_sage"
    assert any("quality_two_stage" in warning and "quality_sage" in warning for warning in request["warnings"])


def test_fish_backend_falls_back_low_vram_two_stage_after_compatibility_resolution():
    request = normalize_request({
        "mode": "FL2VA",
        "first_image": "opening.png",
        "voice_mode": "fish_lock",
        "voice_reference_audio": "voice.wav",
        "target_dialogue": "我们回家。",
        "performance_preset": "low_vram_two_stage",
        "postprocess_mode": "ai_upscale",
    })
    assert request["resolved_backend"] == "ref2va_model"
    assert request["performance_preset"] == "low_vram"
    assert request["postprocess_mode"] == "ai_upscale"


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
