"""Pure policy helpers for the Smart free 1080p preset."""

from __future__ import annotations

from numbers import Real

from .schema import RequestError


SMART_PRESET = "smart_free_1080p"
SMART_UPSCALE_MODEL = "auto"
SMART_LOW_VRAM_UPSCALE_MODEL = "RealESRGAN_x2plus.pth"
LOW_VRAM_MAX_SECONDS = 6
LOW_VRAM_MIN_FREE_GB = 6.0
LOW_VRAM_TOTAL_GB = 16.0


def smart_1080p_target(width, height):
    """Return an even target size whose short edge is 1080 pixels."""
    if width <= 0 or height <= 0:
        raise RequestError("目标尺寸宽高必须为正数")
    width = float(width)
    height = float(height)
    scale = 1080.0 / min(width, height)

    def even_round(value):
        return max(2, int(round(value / 2.0) * 2))

    return even_round(width * scale), even_round(height * scale)


def _duration_seconds(duration):
    try:
        seconds = int(duration)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RequestError("视频时长必须是整数秒") from exc
    if isinstance(duration, Real) and duration != seconds:
        raise RequestError("视频时长必须是整数秒")
    if isinstance(duration, str):
        try:
            if float(duration) != seconds:
                raise RequestError("视频时长必须是整数秒")
        except ValueError as exc:
            raise RequestError("视频时长必须是整数秒") from exc
    return seconds


def resolve_smart_1080p_plan(
    backend,
    duration,
    total_vram_gb,
    free_vram_gb,
    seedvr2_ready=False,
    two_stage_ready=False,
    target_width=None,
    target_height=None,
    voice_mode="none",
):
    """Resolve Smart generation policy for a backend, VRAM state and target.

    ``target_width``/``target_height`` describe the requested final output.
    Targets whose short edge exceeds 1080 (2K QHD / 4K UHD presets) take the
    clarity-first chain: trained latent two-stage redraw for the detail base,
    then one SeedVR2 diffusion upscale to the final size.  Any unmet
    requirement raises before queueing instead of silently degrading.
    """
    if free_vram_gb < LOW_VRAM_MIN_FREE_GB:
        raise RequestError(
            f"低于最低安全预算：当前空闲显存 {float(free_vram_gb):.1f}GB，"
            f"至少需要 {LOW_VRAM_MIN_FREE_GB:.1f}GB；"
            "请关闭其他任务、等待模型卸载或重启 ComfyUI。"
        )
    if backend not in ("fl2va_model", "ref2va_model"):
        raise RequestError(f"不支持的 Smart 1080p backend：{backend}")

    seconds = _duration_seconds(duration)
    high_res_target = (
        target_width is not None
        and target_height is not None
        and min(int(target_width), int(target_height)) > 1080
    )
    target_label = (
        f"{int(target_width)}×{int(target_height)}" if high_res_target else ""
    )
    low_vram = total_vram_gb <= LOW_VRAM_TOTAL_GB

    if high_res_target:
        target_label = f"{int(target_width)}×{int(target_height)}"
        if voice_mode == "fish_lock":
            raise RequestError(
                "Fish S2 声纹锁定与训练型 latent 二采互斥，智能预设最高输出 1080p；"
                "需要 2K/4K 请改用 H3 原生音色参考或不使用音色。"
            )
        if low_vram:
            raise RequestError(
                f"智能预设的 {target_label} 输出需要 20GB 级以上显卡；当前总显存 "
                f"{float(total_vram_gb):.1f}GB，低显存档位最高输出 1080p。"
            )
        if not two_stage_ready:
            raise RequestError(
                f"智能预设的 {target_label} 输出使用「训练型 latent 二采 + SeedVR2」链，"
                "但训练型二采依赖未就绪（turbo v4 / FL 二采 LoRA、3D latent 放大节点或模型缺失）；"
                "请按使用说明安装，或把最终目标降为 1080p。"
            )
        if not seedvr2_ready:
            raise RequestError(
                f"智能预设的 {target_label} 输出需要 SeedVR2 视频超分完成最后一级扩散重建，"
                "但 SeedVR2 节点或 models/SEEDVR2 权重未就绪；请安装后重试，或把最终目标降为 1080p。"
            )
    if low_vram:
        if not 4 <= seconds <= LOW_VRAM_MAX_SECONDS:
            raise RequestError(
                f"请求 {seconds} 秒超出低显存模式最多支持 {LOW_VRAM_MAX_SECONDS} 秒（总显存 "
                f"{float(total_vram_gb):.1f}GB，空闲显存 {float(free_vram_gb):.1f}GB），请缩短或拆段"
            )
        preset = "low_vram_two_stage" if backend == "fl2va_model" else "low_vram"
        route = "trained_latent_fl" if backend == "fl2va_model" else "bypass"
        warning = (
            "已启用低显存 1080p 模式。当前显存档位最多支持 6 秒；系统会降低生成阶段分辨率，"
            "并在生成后免费超分到目标 1080p 尺寸。"
        )
        max_duration = LOW_VRAM_MAX_SECONDS
    else:
        if not 4 <= seconds <= 15:
            raise RequestError("视频时长必须在 4 到 15 秒之间")
        if high_res_target:
            # 2K/4K clarity chain: the trained two-stage redraw builds the
            # detail base, then one SeedVR2 diffusion pass reaches the final
            # size.  Readiness was enforced above; the VRAM/duration budget
            # gate runs in plan_two_stage_dimensions before queueing.
            preset = "quality_two_stage"
            route = "trained_latent_ref" if backend == "ref2va_model" else "trained_latent_fl"
            warning = (
                f"已启用 {target_label} 智能高清链：训练型 latent 二采（U22 配方 8 步首采 + "
                "训练型 3D latent 放大 + 4 步低 sigma 重绘）构建细节基准，再由 SeedVR2 "
                "扩散超分到最终尺寸；显存与时长预算会在排队前检查。"
            )
        # Clarity-first: when the trained two-stage assets and the FHD VRAM
        # budget are present, use the U22-validated 8+4 latent redraw and
        # output 1080p directly -- no separate SeedVR2/AI upscale pass at all.
        elif (
            two_stage_ready
            and float(total_vram_gb) >= 20.0
            and float(free_vram_gb) >= 18.0
        ):
            preset = "quality_two_stage"
            route = "trained_latent_ref" if backend == "ref2va_model" else "trained_latent_fl"
            warning = (
                "已启用训练型 latent 二采直出 1080p：8 步首采 + 训练型 3D latent 放大 + "
                "4 步低 sigma 重绘（U22 验证配方），最终只裁切对齐到 1920×1080，"
                "不再执行额外的 SeedVR2/AI 超分。"
            )
        else:
            # Keep the full 20-step denoising path and accelerate attention only.
            # Turbo's four-step shortcut is fast, but leaves less native detail for
            # the single local upscale pass to reconstruct.
            preset = "quality_sage"
            route = "bypass"
            warning = ""
        max_duration = 15

    postprocess_mode = "video_sr" if seedvr2_ready else "ai_upscale"
    return {
        "performance_preset": preset,
        "postprocess_mode": postprocess_mode,
        "ai_upscale_model": (
            SMART_LOW_VRAM_UPSCALE_MODEL if low_vram else SMART_UPSCALE_MODEL
        ),
        "seedvr2_ready": bool(seedvr2_ready),
        "motion_smoothing": "off",
        "use_easycache": False,
        "low_vram": low_vram,
        "max_duration": max_duration,
        "two_stage_route": route,
        "warning": warning,
    }
