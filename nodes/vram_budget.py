"""Conservative, deterministic workload planning for trained H3 two-stage sampling."""

from __future__ import annotations

import math


LOW_VRAM_TWO_STAGE_MIN_FIRST_MP = 0.20
LOW_VRAM_TWO_STAGE_SCALE = 1.5
LOW_VRAM_TWO_STAGE_MAX_VSR_SCALE = 1.45
LOW_VRAM_TWO_STAGE_MAX_DURATION = 6
LOW_VRAM_TWO_STAGE_768P_MAX_DURATION = 10
LOW_VRAM_TWO_STAGE_768P_MAX_PIXELS = 1344 * 768
LOW_VRAM_TWO_STAGE_768P_MAX_VSR_SCALE = 1.0
BALANCED_FHD_LANDSCAPE = (1920, 1080)
# High-VRAM FHD restores the former detail-preserving grid.  The conservative
# 24GB route below remains 1280x704 to keep its memory bound unchanged.
BALANCED_FHD_FIRST_LANDSCAPE = (1344, 768)
BALANCED_FHD_SECOND_SCALE = 1.5


def split_tile_plan(free_vram_gb):
    """Choose second-stage tiles; larger tiles reduce repeated H3 tile passes.

    The 768px profile is selected only with at least 22GB free before sampling.
    OOM handling in the sampler falls back to 512px, then 256px, without
    changing the requested latent grid or video duration.
    """
    if float(free_vram_gb) >= 22.0:
        return {
            "split_tile_width": 768,
            "split_tile_height": 768,
            "split_chunk_frames": 141,
            "split_temporal_overlap_frames": 39,
            "split_motion_anchor_frames": "39",
        }
    return {
        "split_tile_width": 512,
        "split_tile_height": 512,
        "split_chunk_frames": 73,
        "split_temporal_overlap_frames": 22,
        "split_motion_anchor_frames": "22",
    }


def _aligned_size(width, height, target_mp, alignment=32):
    width = max(1, int(width))
    height = max(1, int(height))
    ratio = float(width) / float(height)
    area = float(target_mp) * 1_000_000.0
    resolved_height = math.sqrt(area / ratio)
    resolved_width = resolved_height * ratio
    aligned_width = max(alignment, int(round(resolved_width / alignment)) * alignment)
    aligned_height = max(alignment, int(round(resolved_height / alignment)) * alignment)
    return aligned_width, aligned_height


def _rejected(reason, max_width, max_height, tier):
    return {
        "allowed": False,
        "reason": reason,
        "vram_safety_tier": tier,
        "max_final_width": int(max_width),
        "max_final_height": int(max_height),
        "quality_basis": "逐帧后处理重建",
    }


def plan_two_stage_dimensions(
    final_width,
    final_height,
    duration,
    total_vram_gb,
    free_vram_gb,
    profile="quality",
    adaptive=False,
):
    """Plan one safe first/second grid without pretending it is native 2K/4K."""
    final_width = int(final_width)
    final_height = int(final_height)
    duration = int(duration)
    total = float(total_vram_gb)
    free = float(free_vram_gb)
    profile = str(profile or "quality")
    long_768p_budget = False
    if final_width <= 0 or final_height <= 0:
        raise ValueError("最终尺寸必须大于 0")
    if duration < 4 or duration > 15:
        raise ValueError("视频时长必须在 4 到 15 秒之间")
    if profile not in {"quality", "low_vram"}:
        raise ValueError(f"未知二采显存预算档位：{profile}")
    if adaptive:
        if not (math.isfinite(total) and math.isfinite(free)):
            return _rejected("显存信息不可用，无法安全规划自适应二采", 0, 0, "unknown")
        adaptive_target = profile == "quality" and (final_width, final_height) in {
            (2560, 1440), (1440, 2560),
        }
        if adaptive_target:
            if total >= 28.0 and free >= 24.0:
                first_width, first_height = ((1280, 736) if final_width >= final_height else (736, 1280))
                second_width, second_height = first_width * 2, first_height * 2
                # Keep the learned 2x grid for detail, then let SeedVR2 do the
                # final diffusion refinement. Direct tiled export was visibly
                # softer even when the canvas fit in VRAM.
                tier, direct = "28gb_plus_qhd", False
            elif total >= 20.0 and free >= 18.0:
                first_width, first_height = ((960, 544) if final_width >= final_height else (544, 960))
                second_width, second_height = first_width * 2, first_height * 2
                tier, direct = "16_24gb_qhd_tiled", False
            else:
                return _rejected("当前空闲显存不足以安全执行自适应 2K 二采", 2560, 1440, "qhd_rejected")
            return {
                "allowed": True, "reason": "自适应 2K 训练型 latent 二采预算通过",
                "vram_safety_tier": tier, "first_stage_width": first_width,
                "first_stage_height": first_height, "second_stage_width": second_width,
                "second_stage_height": second_height,
                "first_stage_megapixels": first_width * first_height / 1_000_000.0,
                "second_stage_megapixels": second_width * second_height / 1_000_000.0,
                "final_scale_x": final_width / second_width, "final_scale_y": final_height / second_height,
                "final_scale": max(final_width / second_width, final_height / second_height),
                "max_final_width": 2560, "max_final_height": 1440,
                "quality_basis": "H3 神经 latent 二采", "budget_profile": profile,
                "qhd_direct": direct, "adaptive_qhd": True, "two_stage_tiling_required": True,
                "balanced_fhd_supersample": False, "conservative_fhd_supersample": False,
                "max_final_vsr_scale": None,
            }

    if profile == "low_vram":
        max_width, max_height = 1920, 1080
        tier = "8gb_low_vram_two_stage"
        if duration > LOW_VRAM_TWO_STAGE_MAX_DURATION:
            long_768p_budget = (
                duration <= LOW_VRAM_TWO_STAGE_768P_MAX_DURATION
                and final_width * final_height <= LOW_VRAM_TWO_STAGE_768P_MAX_PIXELS * 1.02
            )
            if not long_768p_budget:
                return _rejected(
                    "低显存二采只支持 4 到 6 秒 1080p，或最长 10 秒 768p；"
                    "更长时长无法同时保证当前清晰度底线",
                    max_width,
                    max_height,
                    tier,
                )
        if final_width * final_height > max_width * max_height * 1.02:
            return _rejected(
                "低显存二采最高支持 1080p FHD 像素预算的最终输出",
                max_width,
                max_height,
                tier,
            )
        if total < 7.5:
            return _rejected(
                f"低显存二采至少需要 8GB 级显卡，当前总显存 {total:.1f}GB",
                max_width,
                max_height,
                tier,
            )
        if free < 6.0:
            return _rejected(
                f"低显存二采启动前至少需要 6.0GB 空闲显存，当前只有 {free:.1f}GB",
                max_width,
                max_height,
                tier,
            )
        required_free = 6.0
        # Keep the temporal-spatial token budget approximately constant as
        # duration grows. Four-to-six-second FHD retains the validated budget.
        # Long 768p uses a separate 2x tiled grid below. Its redraw reaches the
        # output resolution instead of asking a frame upscaler to invent it.
        duration_factor = 4.0 / float(duration)
        min_first_mp = LOW_VRAM_TWO_STAGE_MIN_FIRST_MP * duration_factor
        max_final_vsr_scale = (
            LOW_VRAM_TWO_STAGE_768P_MAX_VSR_SCALE
            if duration > LOW_VRAM_TWO_STAGE_MAX_DURATION
            else LOW_VRAM_TWO_STAGE_MAX_VSR_SCALE * math.sqrt(float(duration) / 4.0)
        )
        # A final reconstruction model is not a replacement for H3 detail.
        # Derive the first grid from the maximum accepted final scale instead
        # of multiplying the detail floor by the duration factor a second time.
        first_mp = max(
            min_first_mp,
            (final_width * final_height)
            / (
                1_000_000.0
                * (LOW_VRAM_TWO_STAGE_SCALE * max_final_vsr_scale) ** 2
            ),
        )
    elif total < 16.0:
        return _rejected(
            "低显存档位不执行长视频训练型二采；请使用低显存单采并将最终目标限制为1080p",
            1920,
            1080,
            "8_12gb",
        )

    fhd_target = (final_width, final_height) in {
        BALANCED_FHD_LANDSCAPE,
        tuple(reversed(BALANCED_FHD_LANDSCAPE)),
    }
    # Exact FHD gets its own budget on both 24 GB cards and busy 32 GB cards.
    # The conservative grid avoids applying the 2K/4K long-clip margin to a
    # target whose learned second-stage grid is capped around 1080p.
    balanced_fhd_supersample = profile == "quality" and fhd_target and total >= 28.0 and free >= 24.0
    conservative_fhd_supersample = profile == "quality" and fhd_target and total >= 20.0 and not balanced_fhd_supersample
    if profile == "low_vram":
        pass
    elif balanced_fhd_supersample:
        required_free = 24.0
        tier = "28gb_plus"
        max_width, max_height = 1920, 1080
    elif conservative_fhd_supersample:
        required_free = 18.0
        tier = "16_24gb_fhd"
        max_width, max_height = 1920, 1080
    elif total < 28.0:
        if duration > 8:
            return _rejected("短视频2K传统二采最多支持 8 秒；15 秒请使用智能自适应分块链路", 2560, 1440, "16_24gb")
        if final_width > 2560 or final_height > 1440:
            return _rejected(
                "当前显存档位最高开放 2K 训练型二采",
                2560,
                1440,
                "16_24gb",
            )
        required_free = 18.0
        first_mp = 0.50
        tier = "16_24gb"
        max_width, max_height = 2560, 1440
    else:
        required_free = 24.0 if duration >= 12 else 21.0 if duration >= 8 else 18.0
        first_mp = 0.90
        tier = "28gb_plus"
        max_width, max_height = 3840, 2160

    if free < required_free:
        if conservative_fhd_supersample:
            reason = (
                "24GB级显卡的 FHD 训练型二采至少需要 "
                f"{required_free:.1f}GB 空闲显存，当前只有 {free:.1f}GB"
            )
        else:
            reason = f"当前可用显存 {free:.1f}GB 低于安全余量 {required_free:.1f}GB"
        return _rejected(
            reason,
            max_width,
            max_height,
            tier,
        )

    if long_768p_budget:
        # Only the first pass is full-frame. Keep it at one quarter of the
        # target area; the existing 512px / 73-frame tiles bound pass two.
        # Round upward so non-64px targets need at most a small final downscale.
        first_width = max(32, math.ceil(final_width / 64) * 32)
        first_height = max(32, math.ceil(final_height / 64) * 32)
        second_width, second_height = first_width * 2, first_height * 2
    elif balanced_fhd_supersample:
        if final_width > final_height:
            first_width, first_height = BALANCED_FHD_FIRST_LANDSCAPE
        else:
            first_width, first_height = tuple(
                reversed(BALANCED_FHD_FIRST_LANDSCAPE)
            )
        second_width = int(first_width * BALANCED_FHD_SECOND_SCALE)
        second_height = int(first_height * BALANCED_FHD_SECOND_SCALE)
    elif conservative_fhd_supersample:
        if final_width > final_height:
            first_width, first_height = (1280, 704)
            second_width, second_height = (1920, 1056)
        else:
            first_width, first_height = (704, 1280)
            second_width, second_height = (1056, 1920)
    else:
        first_width, first_height = _aligned_size(
            final_width,
            final_height,
            first_mp,
        )
        second_width = max(32, int(round(first_width * 1.5 / 32.0)) * 32)
        second_height = max(32, int(round(first_height * 1.5 / 32.0)) * 32)
    if not (long_768p_budget or balanced_fhd_supersample or conservative_fhd_supersample) and (
        not balanced_fhd_supersample
        and (second_width > final_width or second_height > final_height)
    ):
        # A learned second-stage grid must never exceed the final target. A
        # later downscale would throw away reconstructed detail and ask the
        # final postprocessor to operate on an invalid ratio.
        first_width = max(32, int(math.floor(final_width / 1.5 / 32.0)) * 32)
        first_height = max(32, int(math.floor(final_height / 1.5 / 32.0)) * 32)
        second_width = max(32, int(round(first_width * 1.5 / 32.0)) * 32)
        second_height = max(32, int(round(first_height * 1.5 / 32.0)) * 32)
    scale_x = float(final_width) / float(second_width)
    scale_y = float(final_height) / float(second_height)
    return {
        "allowed": True,
        "reason": "显存、时长与二采网格通过训练型二采安全预算",
        "vram_safety_tier": tier,
        "first_stage_width": first_width,
        "first_stage_height": first_height,
        "second_stage_width": second_width,
        "second_stage_height": second_height,
        "first_stage_megapixels": first_width * first_height / 1_000_000.0,
        "second_stage_megapixels": second_width * second_height / 1_000_000.0,
        "final_scale_x": scale_x,
        "final_scale_y": scale_y,
        "final_scale": max(scale_x, scale_y),
        "max_final_width": max_width,
        "max_final_height": max_height,
        "quality_basis": "H3 神经 latent 二采",
        "budget_profile": profile,
        "two_stage_tiling_required": long_768p_budget,
        "balanced_fhd_supersample": balanced_fhd_supersample,
        "conservative_fhd_supersample": conservative_fhd_supersample,
        "max_final_vsr_scale": (
            max_final_vsr_scale
            if profile == "low_vram"
            else None
        ),
    }
