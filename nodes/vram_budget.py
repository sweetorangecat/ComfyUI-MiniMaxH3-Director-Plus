"""Conservative, deterministic workload planning for trained H3 two-stage sampling."""

from __future__ import annotations

import math


LOW_VRAM_TWO_STAGE_MIN_FIRST_MP = 0.20
LOW_VRAM_TWO_STAGE_SCALE = 1.5
LOW_VRAM_TWO_STAGE_MAX_VSR_SCALE = 1.6


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
):
    """Plan one safe first/second grid without pretending it is native 2K/4K."""
    final_width = int(final_width)
    final_height = int(final_height)
    duration = int(duration)
    total = float(total_vram_gb)
    free = float(free_vram_gb)
    profile = str(profile or "quality")
    if final_width <= 0 or final_height <= 0:
        raise ValueError("最终尺寸必须大于 0")
    if duration < 4 or duration > 15:
        raise ValueError("视频时长必须在 4 到 15 秒之间")
    if profile not in {"quality", "low_vram"}:
        raise ValueError(f"未知二采显存预算档位：{profile}")

    if profile == "low_vram":
        max_width, max_height = 1920, 1080
        tier = "8gb_low_vram_two_stage"
        if duration != 4:
            return _rejected(
                "低显存二采只支持 4 秒视频；更长视频请使用低显存高清单采",
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
        # RTX VSR is a reconstruction finish, not a replacement for real H3
        # detail.  Size the learned second-pass grid so FHD never asks VSR to
        # stretch either axis by much more than 1.6x.  Smaller targets retain
        # the old 0.20 MP fast floor.
        first_mp = max(
            LOW_VRAM_TWO_STAGE_MIN_FIRST_MP,
            (final_width * final_height)
            / (
                1_000_000.0
                * (LOW_VRAM_TWO_STAGE_SCALE * LOW_VRAM_TWO_STAGE_MAX_VSR_SCALE) ** 2
            ),
        )
    elif total < 16.0:
        return _rejected(
            "低显存档位不执行长视频训练型二采；请使用低显存单采并将最终目标限制为1080p",
            1920,
            1080,
            "8_12gb",
        )

    if profile == "low_vram":
        pass
    elif total < 28.0:
        if duration > 8 or final_width > 2560 or final_height > 1440:
            return _rejected(
                "当前显存档位只开放8秒以内的短视频2K训练型二采",
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
        return _rejected(
            f"当前可用显存 {free:.1f}GB 低于安全余量 {required_free:.1f}GB",
            max_width,
            max_height,
            tier,
        )

    first_width, first_height = _aligned_size(
        final_width,
        final_height,
        first_mp,
    )
    second_width = max(32, int(round(first_width * 1.5 / 32.0)) * 32)
    second_height = max(32, int(round(first_height * 1.5 / 32.0)) * 32)
    if second_width > final_width or second_height > final_height:
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
        "max_final_vsr_scale": (
            LOW_VRAM_TWO_STAGE_MAX_VSR_SCALE if profile == "low_vram" else None
        ),
    }
