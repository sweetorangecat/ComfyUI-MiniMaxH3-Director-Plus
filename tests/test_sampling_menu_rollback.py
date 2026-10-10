from pathlib import Path

from nodes.schema import visible_performance_presets


SOURCE = Path("js/minimax_h3_director_plus_v9.js")


def test_sampling_menu_exposes_three_quality_routes_after_62b_rollback():
    assert visible_performance_presets("T2VA") == (
        "quality",
        "quality_sage",
        "quality_two_stage",
    )
    assert visible_performance_presets("FL2VA") == (
        "quality",
        "quality_sage",
        "quality_two_stage",
    )
    assert visible_performance_presets("REF2VA") == (
        "ref_quality_native",
        "quality_sage",
        "quality_two_stage",
    )


def test_frontend_sampling_menu_matches_quality_routes():
    text = SOURCE.read_text(encoding="utf-8")
    assert 't2va: ["稳定质量", "质量优先加速", "质量优先二采样"]' in text
    assert 'endpoint: ["稳定质量", "质量优先加速", "质量优先二采样"]' in text
    assert 'reference: ["参考高清（原生20步）", "质量优先加速", "质量优先二采样"]' in text
    assert 'preset = performanceOptions[0] || "稳定质量"' in text


def test_director_defaults_to_first_visible_quality_route():
    text = Path("nodes/director.py").read_text(encoding="utf-8")
    assert '"performance_preset": (list(USER_PERFORMANCE_PRESET_LABELS), {"default": "稳定质量"' in text
