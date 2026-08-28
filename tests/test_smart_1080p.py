import pytest

from nodes.schema import RequestError
from nodes.smart_1080p import (
    LOW_VRAM_MAX_SECONDS,
    LOW_VRAM_MIN_FREE_GB,
    LOW_VRAM_TOTAL_GB,
    SMART_PRESET,
    SMART_UPSCALE_MODEL,
    resolve_smart_1080p_plan,
    smart_1080p_target,
)


@pytest.mark.parametrize(
    "size, expected",
    [((16, 9), (1920, 1080)), ((9, 16), (1080, 1920)), ((1, 1), (1080, 1080)),
     ((21, 9), (2520, 1080)), ((9, 21), (1080, 2520))],
)
def test_smart_target_keeps_aspect_and_uses_1080_short_edge(size, expected):
    assert smart_1080p_target(*size) == expected
    assert all(axis % 2 == 0 for axis in smart_1080p_target(*size))


@pytest.mark.parametrize("size", [(0, 10), (-1, 10), (10, 0), (10, -1)])
def test_smart_target_rejects_non_positive_dimensions(size):
    with pytest.raises(RequestError):
        smart_1080p_target(*size)


def test_smart_target_rounds_non_integer_long_edge_to_nearest_even():
    width, height = smart_1080p_target(13, 7)
    assert (width, height) == (2006, 1080)
    assert width % 2 == 0 and height % 2 == 0


@pytest.mark.parametrize(
    "backend, preset",
    [("fl2va_model", "quality_sage"), ("ref2va_model", "quality_sage")],
)
def test_normal_vram_selects_backend_preset_and_free_upscale(backend, preset):
    plan = resolve_smart_1080p_plan(backend, 10, 24, 20)
    assert plan["performance_preset"] == preset
    assert plan["postprocess_mode"] == "ai_upscale"
    assert plan["ai_upscale_model"] == SMART_UPSCALE_MODEL
    assert plan["motion_smoothing"] == "off"
    assert plan["use_easycache"] is False
    assert plan["low_vram"] is False
    assert plan["warning"] == ""


def test_low_vram_fl_uses_trained_latent_two_stage():
    plan = resolve_smart_1080p_plan("fl2va_model", 6, 8, 7)
    assert plan["performance_preset"] == "low_vram_two_stage"
    assert plan["two_stage_route"] == "trained_latent_fl"
    assert plan["low_vram"] is True


def test_low_vram_ref_bypasses_two_stage():
    plan = resolve_smart_1080p_plan("ref2va_model", 4, 8, 7)
    assert plan["performance_preset"] == "low_vram"
    assert plan["two_stage_route"] == "bypass"
    assert plan["low_vram"] is True


@pytest.mark.parametrize("backend", ["fl2va_model", "ref2va_model"])
def test_low_vram_common_free_upscale_contract(backend):
    plan = resolve_smart_1080p_plan(backend, 4, 8, 7)
    assert plan["postprocess_mode"] == "ai_upscale"
    assert plan["ai_upscale_model"] == "RealESRGAN_x2plus.pth"
    assert plan["motion_smoothing"] == "off"
    assert plan["use_easycache"] is False
    assert plan["max_duration"] == 6
    assert plan["warning"] == (
        "已启用低显存 1080p 模式。当前显存档位最多支持 6 秒；系统会降低生成阶段分辨率，"
        "并在生成后免费超分到目标 1080p 尺寸。"
    )


@pytest.mark.parametrize("total_vram_gb, low_vram", [(16.0, True), (16.1, False)])
def test_total_vram_boundary_selects_low_vram_at_sixteen_gb(total_vram_gb, low_vram):
    plan = resolve_smart_1080p_plan("fl2va_model", 6, total_vram_gb, 7)
    assert plan["low_vram"] is low_vram


def test_smart_upscale_model_is_the_approved_model_name():
    assert SMART_UPSCALE_MODEL == "RealESRGAN_x2plus.pth"


def test_low_vram_rejects_duration_above_six_with_actionable_chinese_error():
    with pytest.raises(RequestError) as exc:
        resolve_smart_1080p_plan("fl2va_model", 7, 8, 7)
    message = str(exc.value)
    assert "7 秒" in message
    assert "最多支持 6 秒" in message
    assert "总显存 8.0GB" in message
    assert "空闲显存 7.0GB" in message
    assert "缩短或拆段" in message


def test_any_insufficient_free_vram_is_rejected_even_with_large_total_vram():
    with pytest.raises(RequestError) as exc:
        resolve_smart_1080p_plan("fl2va_model", 10, 24, 5.5)
    assert str(exc.value) == (
        "低于最低安全预算：当前空闲显存 5.5GB，至少需要 6.0GB；"
        "请关闭其他任务、等待模型卸载或重启 ComfyUI。"
    )


def test_free_vram_exactly_at_safety_floor_is_allowed():
    plan = resolve_smart_1080p_plan("fl2va_model", 6, 8, 6.0)
    assert plan["low_vram"] is True


def test_free_vram_just_below_safety_floor_is_rejected():
    with pytest.raises(RequestError):
        resolve_smart_1080p_plan("fl2va_model", 6, 8, 5.99)


def test_unknown_backend_is_rejected():
    with pytest.raises(RequestError):
        resolve_smart_1080p_plan("unknown", 10, 24, 20)


@pytest.mark.parametrize("duration", ["6", 6.0])
def test_duration_is_coerced_to_integer(duration):
    plan = resolve_smart_1080p_plan("fl2va_model", duration, 8, 7)
    assert plan["max_duration"] == 6


@pytest.mark.parametrize("duration", [3, 7, 15.1, "bad"])
def test_low_vram_only_allows_four_to_six_integer_seconds(duration):
    with pytest.raises(RequestError):
        resolve_smart_1080p_plan("fl2va_model", duration, 8, 7)


@pytest.mark.parametrize("duration", [3, 16, "bad"])
def test_normal_vram_keeps_public_four_to_fifteen_second_contract(duration):
    with pytest.raises(RequestError):
        resolve_smart_1080p_plan("fl2va_model", duration, 24, 20)


def test_public_constants_match_smart_free_contract():
    assert SMART_PRESET == "smart_free_1080p"
    assert LOW_VRAM_MAX_SECONDS == 6
    assert LOW_VRAM_MIN_FREE_GB == 6.0
    assert LOW_VRAM_TOTAL_GB == 16.0
