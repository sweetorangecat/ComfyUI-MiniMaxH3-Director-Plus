"""Pure policy helpers for adaptive video quality; legacy API key is retained."""

from __future__ import annotations

from numbers import Real
import math

from .schema import RequestError
from .vram_budget import plan_two_stage_dimensions


SMART_PRESET = "smart_free_1080p"
SMART_UPSCALE_MODEL = "auto"
SMART_LOW_VRAM_UPSCALE_MODEL = "RealESRGAN_x2plus.pth"
LOW_VRAM_MAX_SECONDS = 6
LOW_VRAM_MIN_FREE_GB = 6.0
LOW_VRAM_TOTAL_GB = 16.0
LOW_VRAM_LONG_MAX_SECONDS = 15
LOW_VRAM_LONG_MAX_PIXELS = 1344 * 768


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
    target_preset=None,
):
    """Resolve Smart generation policy for a backend, VRAM state and target.

    ``target_width``/``target_height`` describe the requested final output.
    Exact QHD uses a trained 2x latent redraw with tiled sampling when VRAM
    permits; smaller grids and UHD finish with SeedVR2. Unmet dependencies
    raise before sampling instead of silently degrading high-resolution output.
    """
    if not (math.isfinite(float(total_vram_gb)) and math.isfinite(float(free_vram_gb))):
        raise RequestError("显存信息不可用，无法规划智能画质链路")
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
    if target_preset == "480p":
        if not 4 <= seconds <= 15:
            raise RequestError("视频时长必须在 4 到 15 秒之间")
        return {
            "performance_preset": "low_vram" if low_vram else "quality_sage",
            "postprocess_mode": "ai_upscale",
            "ai_upscale_model": SMART_LOW_VRAM_UPSCALE_MODEL if low_vram else SMART_UPSCALE_MODEL,
            "seedvr2_ready": bool(seedvr2_ready), "motion_smoothing": "off",
            "use_easycache": False, "low_vram": low_vram, "max_duration": 15,
            "two_stage_route": "bypass", "dimension_plan": None,
            "warning": "480p 单采：保留 480p 输出目标，按时长控制采样网格，不自动执行训练型二采或 SeedVR2。4–15 秒为策略允许范围，8GB 实际峰值和画质仍需实测。",
        }
    low_vram_long_target = (
        low_vram
        and target_width is not None
        and target_height is not None
        and int(target_width) * int(target_height) <= LOW_VRAM_LONG_MAX_PIXELS
    )
    low_vram_detail = (
        low_vram_long_target
        and backend in ("fl2va_model", "ref2va_model")
        and bool(two_stage_ready)
        and voice_mode in ("none", "h3_reference")
    )
    high_res_target = (
        target_width is not None
        and target_height is not None
        and min(int(target_width), int(target_height)) > 1080
    )
    target_label = (
        f"{int(target_width)}×{int(target_height)}" if high_res_target else ""
    )
    dimension_plan = None

    # Reference audio does not require a blanket ban on video redraw. With
    # sufficient budget, use the normal trained route and freeze/reinsert the
    # first-pass audio. Preserve the native fallback for limited machines.
    reference_redraw_ready = (
        backend == "ref2va_model" and two_stage_ready
        and float(total_vram_gb) >= 20.0 and float(free_vram_gb) >= 18.0
    )
    if voice_mode == "h3_reference" and not low_vram and not reference_redraw_ready:
        return {
            "performance_preset": "ref_quality_native",
            # SeedVR2 is another diffusion model.  It is much slower than the
            # H3 pass and can redraw a completed face, mouth shape, fur, or
            # costume, weakening the reference-audio result it is meant to
            # preserve.  Keep H3-reference jobs on the conservative one-pass
            # scaler even if SeedVR2 happens to be installed.
            "postprocess_mode": "ai_upscale",
            "ai_upscale_model": SMART_UPSCALE_MODEL,
            "seedvr2_ready": bool(seedvr2_ready),
            "motion_smoothing": "off",
            "use_easycache": False,
            "low_vram": False,
            "max_duration": 15,
            "two_stage_route": "bypass",
            "warning": (
                "二采依赖或显存预算不足，使用“参考高清（原生 20 步）”单采路线："
                "不拆分音频/视频 latent；1080p 最终使用保守 AI 超分，不自动启用会重绘人物细节的 SeedVR2。"
            ),
            "dimension_plan": None,
        }

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
        dimension_plan = plan_two_stage_dimensions(
            target_width, target_height, seconds, total_vram_gb, free_vram_gb,
            adaptive=True,
        )
        if not dimension_plan["allowed"]:
            raise RequestError(dimension_plan["reason"])
        if not seedvr2_ready and not dimension_plan.get("qhd_direct"):
            raise RequestError(
                f"智能预设的 {target_label} 输出需要 SeedVR2 视频超分完成最后一级扩散重建，"
                "但 SeedVR2 节点或 models/SEEDVR2 权重未就绪；请安装后重试，或把最终目标降为 1080p。"
            )
    if low_vram_detail:
        dimension_plan = plan_two_stage_dimensions(
            int(target_width),
            int(target_height),
            seconds,
            total_vram_gb,
            free_vram_gb,
            profile="low_vram",
        )
        if not dimension_plan["allowed"]:
            low_vram_detail = False
    if low_vram:
        max_duration = (
            LOW_VRAM_LONG_MAX_SECONDS if low_vram_long_target else LOW_VRAM_MAX_SECONDS
        )
        if not 4 <= seconds <= max_duration:
            raise RequestError(
                f"请求 {seconds} 秒超出低显存模式最多支持 {max_duration} 秒（总显存 "
                f"{float(total_vram_gb):.1f}GB，空闲显存 {float(free_vram_gb):.1f}GB），请缩短或拆段"
            )
        if low_vram_detail:
            preset = "low_vram_two_stage"
            route = "trained_latent_ref" if backend == "ref2va_model" else "trained_latent_fl"
            finish_note = (
                "二采达到目标尺寸后直接导出，不追加 AI 放大；8GB 峰值与画质需实测。"
                if dimension_plan["final_scale"] <= 1.0
                else f"最终 AI X2 收尾约 {dimension_plan['final_scale']:.2f} 倍。"
            )
            if voice_mode == "h3_reference":
                warning = (
                    f"已启用低显存 768p REF2VA 二采：{seconds} 秒使用训练型 latent 二采，"
                    "第二阶段锁定音频分支（零噪声、零重绘遮罩并在完成后精确回填），"
                    "仅对视频做时空分块低 sigma 重绘；"
                    + finish_note
                )
            else:
                warning = (
                    f"已启用低显存 768p 二采：{seconds} 秒使用训练型 latent 二采，"
                    "第二阶段按时空分块低 sigma 重绘补足空间细节；"
                    + finish_note
                )
        elif low_vram_long_target:
            preset = "low_vram"
            route = "bypass"
            warning = (
                "已启用 8GB 低显存 768p 长时单采：允许 4–15 秒；系统会按时长降低 "
                "H3 实际采样网格，15 秒约为 0.26MP，再使用单次 AI X2 重建到所选 768p 输出。"
            )
        else:
            preset = "low_vram_two_stage" if backend == "fl2va_model" else "low_vram"
            route = "trained_latent_fl" if backend == "fl2va_model" else "bypass"
            warning = (
                "已启用低显存质量优先 1080p 模式：采样 20 步，速度较慢。当前显存档位最多支持 6 秒；"
                "系统会降低生成阶段分辨率，"
                "并在生成后免费超分到目标 1080p 尺寸。"
            )
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
                f"已启用 {target_label} 智能画质：训练型 latent 二采；"
                + ("2K 网格分块重绘后裁切到目标尺寸。" if dimension_plan.get("qhd_direct")
                   else "分块重绘后由 SeedVR2 完成最终超分。")
                + "保留请求时长；显存不足时缩小二采分块，无法运行则明确报错。"
            )
        # Use trained 8+4 latent redraw when the FHD budget permits.
        # The director exports FHD directly after this redraw.
        elif (
            two_stage_ready
            and float(total_vram_gb) >= 20.0
            and float(free_vram_gb) >= 18.0
        ):
            preset = "quality_two_stage"
            route = "trained_latent_ref" if backend == "ref2va_model" else "trained_latent_fl"
            warning = (
                "已启用训练型 latent 二采 1080p：8 步首采 + 训练型 3D latent 放大 + "
                "4 步低 sigma 重绘；二采后按原有路线等比导出到目标尺寸，"
                "不再自动追加 SeedVR2 3B 重建。"
            )
        else:
            # Keep the full 20-step denoising path and accelerate attention only.
            # Turbo's four-step shortcut is fast, but leaves less native detail for
            # the single local upscale pass to reconstruct.
            preset = "quality_sage"
            route = "bypass"
            warning = ""
        max_duration = 15

    # Low-VRAM FHD runs must stay on the conservative per-frame X2 path.
    # SeedVR2 diffusion reconstruction is the source of unstable artifacts on
    # 8GB-class cards, even when its node and weights happen to be installed.
    if voice_mode == "h3_reference" and route == "trained_latent_ref":
        warning = warning.replace("8 步首采", "完整首采（至 sigma=0）")
        warning += " 完成全部首采步数后再锁定音频（零噪声、零重绘遮罩、最终精确回填），增加首采耗时；音色匹配与口型仍需实际验证。"
    postprocess_mode = (
        "ai_upscale"
        if low_vram
        else ("video_sr" if seedvr2_ready else "ai_upscale")
    )
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
        "dimension_plan": dimension_plan,
        "two_stage_audio_guard": bool(
            route == "trained_latent_ref"
            and voice_mode == "h3_reference"
        ),
    }
