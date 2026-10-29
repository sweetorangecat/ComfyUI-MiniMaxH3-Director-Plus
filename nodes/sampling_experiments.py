"""Progressive sampling contracts for the installed SelfLift H3 sampler.

This module deliberately imports neither Torch nor ComfyUI at module load time.
The upstream node owns the progressive algorithm; this adapter only validates
the approved recipe and preserves its target-resolution AV input contract.
"""

from __future__ import annotations

import inspect
import math
import time


SELFLIFT_NODE_ID = "SelfLiftH3Sampler"
# selflift-Avatar keeps the original H3 node contract but adds the global
# high-resolution stream path and the long-clip reflect-padding fix.  Prefer it
# when installed; the original facok node remains a compatibility fallback so
# saved workflows do not need a new node or a new JSON graph.
SELFLIFT_NODE_IDS = ("SelfLiftAvatarH3Sampler", SELFLIFT_NODE_ID)
SELFLIFT_PARAMETERS = (
    "model", "positive", "negative", "vae", "latent_image", "sampler", "sigmas",
    "seed", "cfg", "transition_step", "lowres_scale", "rho", "w_min", "w_max",
    "upscaler_model", "model_hires", "highres_tiling", "upscaler_unload",
)
MOTION_LORA_CANDIDATES = (
    "Motion_Repair.safetensors", "动作i连续性修复LORA.safetensors",
    "动作连续性修复LORA.safetensors",
)


def validate_sampling_options(request):
    """Return normalized experiment options, rejecting unsupported active routes."""
    method = str(request.get("sampling_method", "standard"))
    profile = str(request.get("community_lora_profile", "off"))
    if method not in {"standard", "selflift"}:
        raise ValueError(f"未知采样方法：{method}")
    if profile not in {"off", "continuity", "combat", "legacy"}:
        raise ValueError(f"未知社区 LoRA 配置：{profile}")
    if (method != "standard" or profile != "off") and request.get("performance_preset") != "quality_two_stage":
        raise ValueError("高级采样与动作 LoRA 配置仅支持质量优先二采样")
    if method == "selflift":
        if request.get("mode") != "REF2VA" or request.get("voice_mode", "none") not in {"none", "h3_reference", "audio_reuse"}:
            raise ValueError("SelfLift 目前支持 REF2VA 的无音色、H3 原生参考或完整复用音频；Fish 克隆音色不支持")
        if request.get("motion_smoothing", "off") not in {"off", "auto"} or float(request.get("fps", 24)) != 24:
            raise ValueError("SelfLift 目前仅支持 24FPS，不能启用 RIFE")
        width = request.get("requested_width", request.get("width"))
        height = request.get("requested_height", request.get("height"))
        if width is not None and height is not None:
            _validate_output_dimensions(width, height)
    return {"sampling_method": method, "community_lora_profile": profile}


def _validate_output_dimensions(width, height):
    width, height = int(width), int(height)
    if min(width, height) <= 0:
        raise ValueError("SelfLift 最终输出宽高必须为正数")
    return width, height


def plan_selflift_dimensions(width, height, duration, total, free):
    """Plan the target template and the exact upstream 0.65 even-latent prefix."""
    width, height = _validate_output_dimensions(width, height)
    if not all(math.isfinite(float(value)) for value in (total, free)) or float(total) < 20 or float(free) < 18:
        raise ValueError("SelfLift 启动至少需要总显存 20GB、空闲显存 18GB")
    max_area = 1344 * 768
    scale = min(1.0, math.sqrt(max_area / (width * height)), 768 / min(width, height))
    target_w, target_h = (max(32, round(side * scale / 32) * 32) for side in (width, height))
    ratio = width / height
    while target_w * target_h > max_area:
        candidates = [(target_w - 32, target_h), (target_w, target_h - 32)]
        target_w, target_h = min(
            ((w, h) for w, h in candidates if min(w, h) >= 32),
            key=lambda pair: abs(math.log(pair[0] / pair[1] / ratio)),
        )
    low_w, low_h = (max(2, round(side / 16 * 0.65 / 2) * 2) * 16 for side in (target_w, target_h))
    scale_x, scale_y = width / target_w, height / target_h
    return {
        "allowed": True, "reason": "保守启动门槛通过；实际峰值显存仍需实测",
        "first_stage_width": low_w, "first_stage_height": low_h,
        "second_stage_width": target_w, "second_stage_height": target_h,
        "second_stage_megapixels": target_w * target_h / 1_000_000,
        "two_stage_tiling_required": False,
        "final_scale_x": scale_x, "final_scale_y": scale_y,
        "final_scale": max(scale_x, scale_y), "max_final_vsr_scale": None,
        "vram_safety_tier": "selflift_shortvideo_experimental",
        "quality_basis": "SelfLift 6+4 步渐进采样（实验）",
    }


def motion_lora_chain(profile):
    """Return only the explicit motion profile; legacy uses the existing loader."""
    if profile in {"off", "legacy"}:
        return ()
    if profile == "continuity":
        return ((MOTION_LORA_CANDIDATES, 0.25),)
    if profile == "combat":
        return ((MOTION_LORA_CANDIDATES, 0.25), (("H3_Combat_V2.safetensors",), 0.5))
    raise ValueError(f"未知社区 LoRA 配置：{profile}")


def _selflift_contract_error(function):
    """Return a readable contract error for either supported upstream node."""
    parameters = inspect.signature(function).parameters
    declared = {
        name: parameter
        for name, parameter in parameters.items()
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    # facok uses ``model``/``model_hires``; selflift-Avatar names the same
    # inputs ``low_res_model``/``high_res_model``.  All other inputs are shared.
    model_name = "low_res_model" if "low_res_model" in declared else "model"
    required = (
        "positive", "negative", "vae", "latent_image", "sampler", "sigmas",
        "seed", "cfg", "transition_step", "lowres_scale", "rho", "w_min",
        "w_max", "upscaler_model", "highres_tiling", model_name,
    )
    aliases = {
        "model", "low_res_model", "model_hires", "high_res_model", "upscaler_unload",
        "highres_tiling_mode", "highres_token_chunk", "highres_weight_cache_gb",
        "tiling_mode", "tiling_tiles", "tiling_axis",
    }
    missing = [
        name for name in required
        if name not in declared
    ]
    unknown_required = [
        name for name, parameter in declared.items()
        if name not in required and name not in aliases
        and parameter.default is inspect.Parameter.empty
    ]
    if missing or unknown_required:
        return ", ".join(missing + unknown_required)
    return None


def _resolve_selflift_callable(registry):
    """Select the newest compatible SelfLift node without changing the graph."""
    errors = []
    for node_id in SELFLIFT_NODE_IDS:
        factory = registry.get(node_id)
        if not callable(factory):
            continue
        try:
            node = factory()
            function = getattr(node, "sample", None)
            if not callable(function):
                raise RuntimeError("缺少当前 sample 方法")
            contract_error = _selflift_contract_error(function)
            if contract_error:
                raise RuntimeError("方法契约不匹配：" + contract_error)
            return function, node_id
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            errors.append(f"{node_id}: {exc}")
    if errors:
        raise RuntimeError("SelfLift 节点不可用：" + "；".join(errors))
    raise RuntimeError(
        "缺少 SelfLiftAvatarH3Sampler 或 SelfLiftH3Sampler 节点，请安装或更新 SelfLift 插件"
    )


def _selflift_callable(registry):
    """Compatibility wrapper retained for callers that only need the method."""
    return _resolve_selflift_callable(registry)[0]


def _selflift_call_kwargs(function, *, node_id, model, positive, negative, vae,
                          latent, sampler, sigmas, seed, upscaler_name):
    """Build the SelfLift call for both legacy and current upstream nodes.

    Recent SelfLift releases add the ``global_stream`` high-resolution path,
    token batching, and a bounded weight cache.  Older releases do not expose
    those keyword arguments, so pass them only when the installed callable
    advertises them.  This keeps old saved workflows loadable while making the
    U22 memory-saving path explicit on the current node.
    """
    parameters = inspect.signature(function).parameters
    # The current Avatar node has a separate highres_tiling_mode input for the
    # full-sequence global stream.  Older Avatar builds only expose the spatial
    # tiler and reject the global mode string in their tiling_mode input.  Keep
    # the call legal on both versions and use the sparse VAE anchor on every
    # pre-global/legacy route; this is internal so the UI stays simple.  The
    # external learned lifter is the detail-preserving baseline on the full
    # sequence path.  The VAE anchor is retained for legacy spatial routes,
    # where the measured hybrid recipe reduced sparse white strands, but it is
    # intentionally not applied to the global stream: the same correction
    # made skin and hair look waxy in 2K clips.
    avatar = node_id == "SelfLiftAvatarH3Sampler"
    supports_global_stream = avatar and "highres_tiling_mode" in parameters
    artifact_guard = (
        dict(name="avatar_global_stream_learned_only", rho=0.0, w_min=0.5, w_max=1.0,
             highres_tiling_mode="global_stream", global_stream_supported=True,
             speed_warning=None)
        if supports_global_stream else
        dict(name="avatar_spatial_compat_guard" if avatar else "legacy_sparse_artifact_guard",
             rho=0.6, w_min=0.5, w_max=1.0, highres_tiling_mode="spatial",
             global_stream_supported=False,
             speed_warning=(
                 "当前 SelfLift 节点不支持 global_stream；高分辨率会按空间块重复执行，"
                 "请更新 selflift-Avatar 后再测长片速度"
             ))
    )
    values = {
        "model": model,
        "low_res_model": model,
        "positive": positive,
        "negative": negative,
        "vae": vae,
        "latent_image": latent,
        "sampler": sampler,
        "sigmas": sigmas,
        "seed": seed,
        "cfg": 1.0,
        "transition_step": 6,
        "lowres_scale": 0.6,
        "rho": artifact_guard["rho"],
        "w_min": artifact_guard["w_min"],
        "w_max": artifact_guard["w_max"],
        "upscaler_model": upscaler_name,
        "model_hires": model,
        "high_res_model": model,
        "highres_tiling": True,
        "upscaler_unload": True,
        # Current Avatar builds use full-sequence context while batching
        # projections. Older builds ignore these optional arguments and use
        # the legal spatial compatibility values below.
        "highres_tiling_mode": artifact_guard["highres_tiling_mode"],
        "highres_token_chunk": 4096,
        "highres_weight_cache_gb": 4.0,
        "tiling_mode": "auto",
        "tiling_tiles": 2,
        "tiling_axis": "auto",
    }
    declared = {
        name for name, parameter in parameters.items()
        if parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                              inspect.Parameter.KEYWORD_ONLY)
    }
    return {
        name: value for name, value in values.items()
        if name in declared
    }, artifact_guard


def selflift_dependency_report(registry, upscaler_name):
    """Check the current upstream method and a resolved external learned lifter."""
    missing = []
    try:
        _, node_id = _resolve_selflift_callable(registry or {})
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        missing.append(str(exc))
    if not str(upscaler_name or "").strip() or str(upscaler_name).lower() == "none":
        missing.append("SelfLift 必须安装并选择 H3 learned latent upscaler")
    return {
        "ready": not missing,
        "missing": missing,
        "upscaler_model": upscaler_name,
        "node_id": node_id if not missing else None,
    }


def _av_shapes(latent):
    samples = latent.get("samples") if isinstance(latent, dict) else None
    if not getattr(samples, "is_nested", False) or not callable(getattr(samples, "unbind", None)):
        raise ValueError("SelfLift 需要 H3 AV nested latent（视频和音频两个分支）")
    streams = samples.unbind()
    shapes = [tuple(stream.shape) for stream in streams]
    if len(shapes) != 2 or len(shapes[0]) != 5 or len(shapes[1]) != 4 or any(
        any(int(size) <= 0 for size in shape) for shape in shapes
    ) or shapes[0][0] != shapes[1][0]:
        raise ValueError("SelfLift AV latent 分支尺寸不匹配")
    return shapes


def _unwrap_sampler_result(result):
    """Normalize ComfyUI node outputs to the returned LATENT value.

    Native node functions may return a tuple/list, while ComfyUI custom nodes
    commonly return ``{"ui": ..., "result": (latent,)}``.  The latter shape
    is used by current selflift-Avatar releases and must be unwrapped before
    validating the AV latent contract.
    """
    if isinstance(result, dict) and "samples" in result:
        return result
    result_value = getattr(result, "result", None)
    if result_value is not None:
        result = result_value
    elif isinstance(result, dict) and "result" in result:
        result = result["result"]
    if isinstance(result, (tuple, list)) and result:
        result = result[0]
    return result


def _validate_euler(sampler):
    function = getattr(sampler, "sampler_function", None)
    try:
        from comfy.samplers import KSAMPLER
        from comfy.k_diffusion.sampling import sample_euler
    except ImportError:
        valid = callable(function) and getattr(function, "__name__", None) == "sample_euler"
    else:
        valid = isinstance(sampler, KSAMPLER) and function is sample_euler
    if not valid or getattr(sampler, "extra_options", {}).get("s_churn", 0.0) != 0.0:
        raise ValueError("SelfLift 必须使用标准 Euler 采样器（s_churn=0）")


def run_selflift(model, positive, vae, sampler, sigmas, latent, seed, guide, registry=None, negative=None):
    """Execute the U22 ten-NFE H3 recipe using the installed upstream node."""
    if vae is None:
        raise ValueError("SelfLift 缺少视频 VAE，请连接 video_vae")
    if model is None or positive is None:
        raise ValueError("SelfLift 缺少模型或 positive conditioning")
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**64:
        raise ValueError("SelfLift 缺少有效的噪音 seed")
    if getattr(sigmas, "ndim", 1) != 1:
        raise ValueError("SelfLift 需要一维 sigma 轨迹")
    values = [float(value) for value in sigmas]
    if len(values) != 11 or not all(math.isfinite(value) for value in values) or any(
        value <= 0 for value in values[:-1]
    ) or values[-1] != 0 or any(b > a for a, b in zip(values, values[1:])) or values[4] >= 1:
        raise ValueError("SelfLift 需要有效的 10 步 sigma 轨迹并以 sigma=0 结束")
    _validate_euler(sampler)
    shapes = _av_shapes(latent)
    expected = (guide.get("second_stage_height"), guide.get("second_stage_width"))
    if all(value is not None for value in expected) and tuple(int(value) // 16 for value in expected) != shapes[0][-2:]:
        raise ValueError("SelfLift 初始 latent 必须使用目标 target 网格，不能传入首采小网格")
    if registry is None:
        from .two_stage_assets import _comfy_node_mappings
        registry = _comfy_node_mappings()
    upscaler_name = guide.get("selflift_upscaler_model")
    report = selflift_dependency_report(registry, upscaler_name)
    if not report["ready"]:
        raise RuntimeError("SelfLift 依赖未就绪：" + "；".join(report["missing"]))
    started = time.perf_counter()
    try:
        function, node_id = _resolve_selflift_callable(registry)
        call_kwargs, artifact_guard = _selflift_call_kwargs(
            function,
            node_id=node_id,
            model=model,
            positive=positive,
            negative=[] if negative is None else negative,
            vae=vae,
            latent=latent,
            sampler=sampler,
            sigmas=sigmas,
            seed=seed,
            upscaler_name=upscaler_name,
        )
        guide["selflift_highres_tiling"] = bool(call_kwargs.get("highres_tiling", False))
        guide["selflift_highres_tiling_mode"] = (
            artifact_guard["highres_tiling_mode"]
            if call_kwargs.get("highres_tiling") else None
        )
        guide["selflift_highres_token_chunk"] = call_kwargs.get("highres_token_chunk")
        guide["selflift_highres_weight_cache_gb"] = call_kwargs.get("highres_weight_cache_gb")
        guide.update(
            selflift_node_id=node_id,
            selflift_node_variant=artifact_guard["name"],
            selflift_global_stream_supported=artifact_guard["global_stream_supported"],
            selflift_speed_warning=artifact_guard["speed_warning"],
            selflift_rho=call_kwargs["rho"],
            selflift_w_min=call_kwargs["w_min"],
            selflift_w_max=call_kwargs["w_max"],
        )
        result = function(**call_kwargs)
    except RuntimeError as exc:
        # Preserve CUDA OOM's exception type so ComfyUI can release its models.
        raise type(exc)(f"SelfLift 执行失败，已停止：{exc}") from exc
    except (AttributeError, ImportError, KeyError, OSError, TypeError, ValueError) as exc:
        raise RuntimeError(f"SelfLift 执行失败，已停止：{exc}") from exc
    result = _unwrap_sampler_result(result)
    try:
        output_shapes = _av_shapes(result)
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError(f"SelfLift 返回无效的 AV latent：{exc}") from exc
    if output_shapes != shapes:
        raise RuntimeError("SelfLift 返回 AV latent 的目标尺寸或时长发生变化")
    guide.update(sampling_actual_method="selflift", sampling_executed_steps=10,
                 sampling_route=node_id,
                 sampling_elapsed_seconds=time.perf_counter() - started,
                 two_stage_first_executed_steps=6, two_stage_second_executed_steps=4,
                 two_stage_first_sigma_count=7, two_stage_second_sigma_count=5,
                 two_stage_status="SelfLift 6+4 步渐进采样完成", two_stage_second_stage_path="selflift")
    return result
