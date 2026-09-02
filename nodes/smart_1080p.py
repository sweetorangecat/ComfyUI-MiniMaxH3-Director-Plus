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


def resolve_smart_1080p_plan(backend, duration, total_vram_gb, free_vram_gb, seedvr2_ready=False):
    """Resolve Smart 1080p generation policy for a backend and VRAM state."""
    if free_vram_gb < LOW_VRAM_MIN_FREE_GB:
        raise RequestError(
            f"低于最低安全预算：当前空闲显存 {float(free_vram_gb):.1f}GB，"
            f"至少需要 {LOW_VRAM_MIN_FREE_GB:.1f}GB；"
            "请关闭其他任务、等待模型卸载或重启 ComfyUI。"
        )
    if backend not in ("fl2va_model", "ref2va_model"):
        raise RequestError(f"不支持的 Smart 1080p backend：{backend}")

    seconds = _duration_seconds(duration)
    low_vram = total_vram_gb <= LOW_VRAM_TOTAL_GB
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
