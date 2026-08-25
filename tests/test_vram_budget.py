import pytest


from nodes.vram_budget import plan_two_stage_dimensions


def test_32gb_15s_2k_keeps_neural_basis_near_1080p():
    plan = plan_two_stage_dimensions(
        2560,
        1440,
        15,
        total_vram_gb=32,
        free_vram_gb=29,
    )

    assert plan["allowed"] is True
    assert 0.80 <= plan["first_stage_megapixels"] <= 0.95
    assert 1.80 <= plan["second_stage_megapixels"] <= 2.15
    assert 1.25 <= plan["final_scale"] <= 1.45
    assert plan["first_stage_width"] % 32 == 0
    assert plan["second_stage_height"] % 32 == 0


def test_32gb_15s_4k_is_streaming_2x_not_native_4k_latent():
    plan = plan_two_stage_dimensions(
        3840,
        2160,
        15,
        total_vram_gb=32,
        free_vram_gb=29,
    )

    assert plan["allowed"] is True
    assert plan["second_stage_width"] <= 1920
    assert plan["second_stage_height"] <= 1088
    assert 1.95 <= plan["final_scale"] <= 2.10
    assert plan["quality_basis"] == "H3 神经 latent 二采"


def test_8gb_long_video_rejects_two_stage_and_4k():
    plan = plan_two_stage_dimensions(
        3840,
        2160,
        15,
        total_vram_gb=8,
        free_vram_gb=7,
    )

    assert plan["allowed"] is False
    assert plan["max_final_width"] == 1920
    assert plan["max_final_height"] == 1080
    assert "低显存" in plan["reason"]


def test_idle_8gb_allows_four_second_fhd_low_vram_two_stage():
    plan = plan_two_stage_dimensions(
        1920,
        1080,
        4,
        total_vram_gb=8,
        free_vram_gb=7,
        profile="low_vram",
    )

    assert plan["allowed"] is True
    assert plan["vram_safety_tier"] == "8gb_low_vram_two_stage"
    assert plan["first_stage_megapixels"] <= 0.21
    assert plan["second_stage_megapixels"] <= 0.48
    assert plan["max_final_width"] == 1920
    assert plan["max_final_height"] == 1080


def test_8gb_low_vram_two_stage_rejects_longer_clips():
    plan = plan_two_stage_dimensions(
        1920,
        1080,
        5,
        total_vram_gb=8,
        free_vram_gb=7,
        profile="low_vram",
    )

    assert plan["allowed"] is False
    assert "只支持 4 秒" in plan["reason"]


def test_8gb_low_vram_two_stage_rejects_target_above_fhd_area():
    plan = plan_two_stage_dimensions(
        2560,
        1440,
        4,
        total_vram_gb=8,
        free_vram_gb=7,
        profile="low_vram",
    )

    assert plan["allowed"] is False
    assert "最高支持 1080p FHD" in plan["reason"]


def test_8gb_low_vram_two_stage_accepts_ultrawide_with_fhd_pixel_budget():
    plan = plan_two_stage_dimensions(
        2048,
        864,
        4,
        total_vram_gb=8,
        free_vram_gb=7,
        profile="low_vram",
    )

    assert plan["allowed"] is True


def test_8gb_low_vram_two_stage_rejects_busy_gpu_before_sampling():
    plan = plan_two_stage_dimensions(
        1920,
        1080,
        4,
        total_vram_gb=8,
        free_vram_gb=5.9,
        profile="low_vram",
    )

    assert plan["allowed"] is False
    assert "至少需要 6.0GB" in plan["reason"]


def test_standard_quality_profile_still_rejects_8gb_four_second_fhd():
    plan = plan_two_stage_dimensions(
        1920,
        1080,
        4,
        total_vram_gb=8,
        free_vram_gb=7,
    )

    assert plan["allowed"] is False


def test_busy_32gb_gpu_fails_before_sampling():
    plan = plan_two_stage_dimensions(
        2560,
        1440,
        15,
        total_vram_gb=32,
        free_vram_gb=8,
    )

    assert plan["allowed"] is False
    assert "当前可用显存" in plan["reason"]


def test_24gb_allows_short_2k_but_rejects_15_seconds():
    short = plan_two_stage_dimensions(
        2560,
        1440,
        4,
        total_vram_gb=24,
        free_vram_gb=22,
    )
    long = plan_two_stage_dimensions(
        2560,
        1440,
        15,
        total_vram_gb=24,
        free_vram_gb=22,
    )

    assert short["allowed"] is True
    assert short["max_final_width"] == 2560
    assert long["allowed"] is False
    assert "短视频2K" in long["reason"]


def test_portrait_target_preserves_orientation_and_scale():
    plan = plan_two_stage_dimensions(
        1440,
        2560,
        4,
        total_vram_gb=32,
        free_vram_gb=29,
    )

    assert plan["allowed"] is True
    assert plan["first_stage_height"] > plan["first_stage_width"]
    assert plan["second_stage_height"] > plan["second_stage_width"]
    assert plan["final_scale_x"] == pytest.approx(
        1440 / plan["second_stage_width"]
    )
    assert plan["final_scale_y"] == pytest.approx(
        2560 / plan["second_stage_height"]
    )


def test_small_final_target_caps_neural_second_stage_instead_of_downscaling():
    plan = plan_two_stage_dimensions(
        1344,
        768,
        4,
        total_vram_gb=32,
        free_vram_gb=29,
    )

    assert plan["allowed"] is True
    assert (plan["first_stage_width"], plan["first_stage_height"]) == (896, 512)
    assert (plan["second_stage_width"], plan["second_stage_height"]) == (1344, 768)
    assert plan["final_scale"] == pytest.approx(1.0)
