from pathlib import Path

from nodes.schema import allowed_motion_smoothing, normalize_request, visible_performance_presets


SOURCE = Path("js/minimax_h3_director_plus_v10.js")


def test_sampling_menu_exposes_quality_routes_and_low_vram_after_62b_rollback():
    assert visible_performance_presets("T2VA") == (
        "quality",
        "quality_sage",
        "quality_two_stage",
        "low_vram",
    )
    assert visible_performance_presets("FL2VA") == (
        "quality",
        "quality_sage",
        "quality_two_stage",
        "low_vram",
    )
    assert visible_performance_presets("REF2VA") == (
        "ref_quality_native",
        "quality_sage",
        "quality_two_stage",
        "low_vram",
    )


def test_frontend_sampling_menu_matches_quality_routes():
    text = SOURCE.read_text(encoding="utf-8")
    assert 't2va: ["稳定质量", "质量优先加速", "质量优先二采样", "低显存"]' in text
    assert 'endpoint: ["稳定质量", "质量优先加速", "质量优先二采样", "低显存"]' in text
    assert 'reference: ["参考高清（原生20步）", "质量优先加速", "质量优先二采样", "低显存"]' in text
    assert 'preset = performanceOptions[0] || "稳定质量"' in text


def test_director_defaults_to_first_visible_quality_route():
    text = Path("nodes/director.py").read_text(encoding="utf-8")
    assert '"performance_preset": (list(USER_PERFORMANCE_PRESET_LABELS), {"default": "稳定质量"' in text


def test_low_vram_menu_is_available_for_every_mode_and_voice_route():
    for mode in ("T2VA", "I2VA", "FL2VA", "L2VA", "REF2VA"):
        for voice in ("none", "h3_reference", "fish_lock", "audio_reuse"):
            assert "low_vram" in visible_performance_presets(mode, voice), (mode, voice)


def test_low_vram_selection_stays_on_single_pass_without_rife():
    for mode in ("T2VA", "REF2VA"):
        request = normalize_request({
            "mode": mode,
            "performance_preset": "低显存",
            "postprocess_mode": "ai_upscale",
            "duration": 4,
        })
        assert request["performance_preset"] == "low_vram"
        assert request["motion_smoothing"] == "off"
        assert allowed_motion_smoothing("low_vram", "ai_upscale") == ("off",)
