"""Official, backend-aware acceleration for the U11 Director Plus workflow."""

from __future__ import annotations

from contextlib import contextmanager
import importlib
import importlib.util
import logging
import sys
from pathlib import Path

from .h3_reuse_attention import apply_h3_reuse_attention
from .schema import TWO_STAGE_PERFORMANCE_PRESETS, allowed_performance_presets
from .two_stage_assets import (
    FL_STAGE1_LORA,
    FL_STAGE2_LORA,
    REF_STAGE_LORA,
    SIGMA_REFINER_NODE_ID,
    resolve_registered_model_name,
    resolve_two_stage_route,
)


LOGGER = logging.getLogger("MiniMaxH3.DirectorPlus")

PRESETS = {
    "quality": {"steps": 20, "use_sage": False, "use_cache": False, "interpolate": False},
    # Quality-first acceleration: keep native 20-step sampling and patch only
    # SageAttention. No Turbo adapter or EasyCache is involved.
    "quality_sage": {
        "steps": 20,
        "use_sage": True,
        "use_cache": False,
        "interpolate": False,
        "clip_device": "dynamic",
        "vae_device": "dynamic",
        "fish_device": "cpu",
    },
    # Trained H3 latent route: the first pass establishes the composition,
    # then the route-specific learned 3D upscaler feeds a matched tail model.
    "quality_two_stage": {
        "steps": 8,
        "two_stage_split_step": 4,
        "two_stage_scale": 1.5,
        # Keep native attention math, release the large normalized input, use
        # the consumed Q region as scratch, and chunk both attention and MLP.
        "use_head_chunking": True,
        "minimax_head_chunks": 8,
        "use_sage": False,
        "use_cache": False,
        "interpolate": False,
        "clip_device": "dynamic",
        "vae_device": "dynamic",
        "fish_device": "cpu",
    },
    # Community v4 FL/T2V adapter: a single 8-step pass tuned for detail,
    # using the stock simple/Euler contract and no latent second pass.
    "fl_quality_fast_v4": {
        "steps": 8,
        "use_sage": False,
        "use_cache": False,
        "interpolate": False,
        "use_turbo_sampler": False,
        "lora_strength": 1.0,
    },
    # RTX 3070-class route: keep the trained 4+4 contract but shrink the
    # first/second grids in the director and stage both samplers through
    # ComfyUI LOW_VRAM. The deterministic planner limits this to four seconds.
    "low_vram_two_stage": {
        "steps": 8,
        "two_stage_split_step": 4,
        "two_stage_scale": 1.5,
        "use_head_chunking": True,
        "minimax_head_chunks": 16,
        "use_sage": False,
        "use_cache": False,
        "interpolate": False,
        "clip_device": "dynamic",
        "vae_device": "dynamic",
        "fish_device": "cpu",
    },
    "fast_4step": {"steps": 4, "use_sage": True, "use_cache": True, "interpolate": False},
    "reference_fast": {"steps": 6, "use_sage": True, "use_cache": True, "interpolate": False},
    # Reference backend quality/speed choices are intentionally separate from
    # the legacy shared presets above.
    "ref_quality_native": {
        "steps": 20,
        "use_sage": True,
        "use_cache": False,
        "interpolate": False,
    },
    "ref_fast_4step": {
        "steps": 4,
        "use_sage": False,
        "use_cache": False,
        "interpolate": False,
        "use_turbo_sampler": False,
        "lora_strength": 1.0,
    },
    # Keep ComfyUI's native dynamic patcher route.  It stages large H3
    # components in host RAM and moves only the active weights to the GPU.
    # Forcing the wrappers themselves to CPU makes text encoding unusably slow.
    "low_vram": {
        "steps": 8,
        "use_sage": True,
        "use_cache": False,
        "interpolate": False,
        "clip_device": "dynamic",
        "vae_device": "dynamic",
        "fish_device": "cpu",
    },
    "custom": {"steps": 20, "use_sage": False, "use_cache": False, "interpolate": False},
}

# Existing official FL2VA adapter and the separate official Ref2VA adapter.
TURBO_LORA_NAME = "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors"
FL_V4_LORA_NAME = "minimax_h3_turbo_v4_step600_ema.safetensors"
REF2VA_TURBO_LORA_NAME = "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"

PRESET_LABELS = {
    "稳定质量": "quality",
    "质量优先加速": "quality_sage",
    "质量优先二采样": "quality_two_stage",
    "高清快速（v4 8步）": "fl_quality_fast_v4",
    "极速4步": "fast_4step",
    "参考图加速": "reference_fast",
    "参考高清（原生20步）": "ref_quality_native",
    "参考极速（官方4步）": "ref_fast_4step",
    "低显存": "low_vram",
    "低显存二采": "low_vram_two_stage",
    "自定义": "custom",
    # Keep the old mojibake labels loadable for existing saved workflows.
    "绋冲畾璐ㄩ噺": "quality",
    "鏋侀€?姝?": "fast_4step",
    "鍙傝€冨浘鍔犻€?": "reference_fast",
    "浣庢樉瀛?": "low_vram",
    "鑷畾涔?": "custom",
}


class H3LoRAApplicationError(RuntimeError):
    """Raised when an official H3 LoRA cannot be applied safely."""


def _patch_entry_count(model):
    """Return the number of model patch entries exposed by ComfyUI."""
    try:
        patches = model.patches
        if not isinstance(patches, dict):
            raise TypeError("patches 不是 dict")
        return sum(
            len(entries) if isinstance(entries, (list, tuple)) else int(bool(entries))
            for entries in patches.values()
        )
    except Exception as exc:
        raise H3LoRAApplicationError(f"无法读取 H3 LoRA 模型补丁: {exc}") from exc


def _load_lightx2v_lora(model, lora_name=None, strength=1.0, low_vram=False):
    """Apply an official H3 adapter with ComfyUI's core model-only loader."""
    del low_vram  # ComfyUI's current memory policy owns low-VRAM handling.
    requested_lora_name = lora_name or TURBO_LORA_NAME
    resolved_name = resolve_registered_model_name("loras", requested_lora_name)
    if resolved_name is None:
        raise H3LoRAApplicationError(f"缺少 H3 Turbo LoRA: {requested_lora_name}")
    before_count = _patch_entry_count(model)
    try:
        import nodes as comfy_nodes

        loaded_model = comfy_nodes.LoraLoaderModelOnly().load_lora_model_only(
            model,
            resolved_name,
            float(strength),
        )[0]
    except H3LoRAApplicationError:
        raise
    except Exception as exc:
        raise H3LoRAApplicationError(f"官方 H3 Turbo LoRA 加载失败: {resolved_name}: {exc}") from exc
    patch_delta = _patch_entry_count(loaded_model) - before_count
    if patch_delta <= 0:
        raise H3LoRAApplicationError(f"官方 H3 Turbo LoRA 未应用任何模型补丁: {resolved_name}")
    LOGGER.info(
        "[H3 LoRA] 官方加载成功 name=%s strength=%s patch_delta=%s",
        resolved_name,
        float(strength),
        patch_delta,
    )
    return loaded_model


def _turbo_class(name):
    module_name = "_u11_minimax_h3_turbo_runtime"
    loaded = sys.modules.get(module_name)
    if loaded is not None:
        return getattr(loaded, name)
    path = Path(__file__).resolve().parents[2] / "ComfyUI-MiniMax-H3-Turbo" / "__init__.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("缺少 ComfyUI-MiniMax-H3-Turbo 自定义节点")
    module = importlib.util.module_from_spec(spec)
    try:
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return getattr(module, name)
    except (ImportError, AttributeError, OSError) as exc:
        raise RuntimeError("缺少 ComfyUI-MiniMax-H3-Turbo 自定义节点") from exc


def _kj_class(name):
    """Load a KJ optimization node without relying on a hyphenated package import."""
    module_name = "_u11_kjnodes_optimization_runtime"
    loaded = sys.modules.get(module_name)
    if loaded is None:
        path = Path(__file__).resolve().parents[2] / "ComfyUI-KJNodes" / "nodes" / "model_optimization_nodes.py"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError("缺少 ComfyUI-KJNodes 优化节点")
        loaded = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = loaded
        try:
            spec.loader.exec_module(loaded)
        except (AttributeError, ImportError, OSError) as exc:
            sys.modules.pop(module_name, None)
            raise RuntimeError("无法加载 ComfyUI-KJNodes 优化节点") from exc
    try:
        return getattr(loaded, name)
    except AttributeError as exc:
        raise RuntimeError(f"缺少 KJ 优化节点: {name}") from exc


def _kj_ltx_class(name):
    """Load KJNodes' H3-specific memory-efficient attention patch."""
    for loaded_name, loaded in list(sys.modules.items()):
        if loaded_name.endswith("nodes.ltxv_nodes") and hasattr(loaded, name):
            return getattr(loaded, name)
    module_name = "_u11_kjnodes_ltxv_runtime"
    loaded = sys.modules.get(module_name)
    if loaded is None:
        path = Path(__file__).resolve().parents[2] / "ComfyUI-KJNodes" / "nodes" / "ltxv_nodes.py"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError("缺少 KJNodes H3 内存高效 Sage 补丁")
        loaded = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = loaded
        try:
            spec.loader.exec_module(loaded)
        except (AttributeError, ImportError, OSError) as exc:
            sys.modules.pop(module_name, None)
            raise RuntimeError("无法加载 KJNodes H3 内存高效 Sage 补丁") from exc
    try:
        return getattr(loaded, name)
    except AttributeError as exc:
        raise RuntimeError(f"缺少 KJNodes H3 内存高效节点: {name}") from exc


def _kj_minimax_class(name):
    """Load KJNodes' exact head-chunk patch for MiniMax H3."""
    for loaded_name, loaded in list(sys.modules.items()):
        if loaded_name.endswith("nodes.minimax_nodes") and hasattr(loaded, name):
            return getattr(loaded, name)
    module_name = "_u11_kjnodes_minimax_runtime"
    loaded = sys.modules.get(module_name)
    if loaded is None:
        path = Path(__file__).resolve().parents[2] / "ComfyUI-KJNodes" / "nodes" / "minimax_nodes.py"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError("缺少 KJNodes MiniMax H3 分块补丁")
        loaded = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = loaded
        try:
            spec.loader.exec_module(loaded)
        except (AttributeError, ImportError, OSError) as exc:
            sys.modules.pop(module_name, None)
            raise RuntimeError("无法加载 KJNodes MiniMax H3 分块补丁") from exc
    try:
        return getattr(loaded, name)
    except AttributeError as exc:
        raise RuntimeError(f"缺少 KJNodes MiniMax H3 节点: {name}") from exc


def _node_model(result):
    """Unwrap both ComfyUI v3 NodeOutput and legacy tuple node results."""
    result = getattr(result, "result", result)
    return result[0] if isinstance(result, (tuple, list)) else result


def _apply_sage_attention(model, guide):
    """Apply SageAttention with an RTX 30xx-safe kernel."""
    preset = PRESET_LABELS.get(guide.get("performance_preset", "quality"), guide.get("performance_preset", "quality"))
    if preset in {"quality_sage", "low_vram"}:
        # The generic KJ override keeps full Q/K/V tensors alive and can add
        # multiple GiB of temporary memory on long H3 sequences. The H3
        # patch quantizes the packed attention path and splits independent
        # heads, preserving the math while shrinking that working set.
        try:
            sage_node = _kj_ltx_class("MiniMaxH3MemoryEfficientSageAttentionPatch")()
            model = _node_model(sage_node.execute(model))
            default_chunks = 16 if preset == "low_vram" else 8
            chunks = max(1, int(guide.get("minimax_head_chunks", default_chunks)))
            chunk_node = _kj_minimax_class("MiniMaxLowVRAMAttention")()
            return _node_model(chunk_node.execute(model, chunks))
        except (AttributeError, ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
            raise RuntimeError(f"H3 内存高效 SageAttention 不可用: {exc}") from exc

    # H3 is BF16 on this install. The FP8 PV kernel is unreliable on SM86;
    # the FP16 PV kernel supports BF16 and FP16 inputs on RTX 3070.
    mode = guide.get("sage_attention_mode", "sageattn_qk_int8_pv_fp16_cuda")
    node = _kj_class("PathchSageAttentionKJ")()
    return node.patch(model, mode, allow_compile=False)[0]


def _apply_minimax_reuse_attention(model, guide):
    """Reduce H3 attention peak by reusing consumed fused-QKV storage."""
    chunks = max(1, int(guide.get("minimax_head_chunks", 8)))
    return apply_h3_reuse_attention(model, chunks)


def _apply_easy_cache(model, guide):
    """Apply ComfyUI's native EasyCache wrapper to the routed model."""
    node_module = importlib.import_module("comfy_extras.nodes_easycache")
    result = node_module.EasyCacheNode.execute(
        model,
        reuse_threshold=float(guide.get("easycache_threshold", 0.2)),
        start_percent=float(guide.get("easycache_start_percent", 0.2)),
        end_percent=float(guide.get("easycache_end_percent", 0.9)),
        verbose=True,
    )
    return result.result[0]


def preset_values(name, backend=None):
    name = PRESET_LABELS.get(name, name)
    if name == "smart_free_1080p":
        raise ValueError("免费智能 1080p 必须先由导演台解析为具体性能预设")
    try:
        values = dict(PRESETS[name])
    except KeyError as exc:
        raise ValueError(f"未知性能预设：{name}") from exc

    if name == "fast_4step":
        if backend == "ref2va_model":
            # Official Ref2VA Turbo is trained for four native Euler steps.
            values.update(steps=4, use_turbo_sampler=False)
        else:
            values["use_turbo_sampler"] = backend == "fl2va_model"
    else:
        values["use_turbo_sampler"] = False
    return values


def _runtime_preset_values(guide, preset):
    """Apply route-specific quality guards on top of the shared preset."""
    values = preset_values(preset, guide.get("resolved_backend"))
    # EasyCache is not part of the official T2VA Turbo recipe and can soften
    # fast text-only motion. Keep T2VA fast mode to Turbo + Sage only.
    if guide.get("mode") == "T2VA" and preset == "fast_4step":
        values["use_cache"] = False
    return values


@contextmanager
def memory_policy(guide):
    """Temporarily apply the workflow's low-VRAM policy to ComfyUI.

    ComfyUI normally restores its global state only at process start.  A
    context keeps this workflow self-contained: model loading and sampling
    see LOW_VRAM, while other workflows retain the state they had before.
    """
    requested = guide.get("performance_preset", "quality")
    preset = PRESET_LABELS.get(
        requested,
        requested,
    )
    if preset == "smart_free_1080p":
        raise ValueError("免费智能 1080p 必须先由导演台解析为具体性能预设")
    if preset not in {"low_vram", "low_vram_two_stage", "quality_sage"}:
        yield
        return

    try:
        import comfy.model_management as model_management
    except ImportError:
        yield
        return

    previous = model_management.vram_state
    model_management.vram_state = model_management.VRAMState.LOW_VRAM
    try:
        yield
    finally:
        model_management.vram_state = previous


class MiniMaxH3PerformancePreset:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"guide": ("MINIMAX_H3_DIRECTOR_PLUS_GUIDE",)},
            "optional": {"acceleration_ready": ("BOOLEAN",)},
        }

    RETURN_TYPES = ("INT", "BOOLEAN", "BOOLEAN", "STRING")
    RETURN_NAMES = ("采样步数", "启用 Sage", "启用缓存", "预设说明")
    FUNCTION = "apply"
    CATEGORY = "MiniMax H3 导演台 Plus"

    def apply(self, guide, acceleration_ready=None):
        name, downgraded = _safe_guide_preset(guide)
        mode = guide.get("mode")
        voice_mode = guide.get("voice_mode", "none")
        values = _runtime_preset_values(guide, name)
        if name in {"fast_4step", "reference_fast", "ref_fast_4step", "fl_quality_fast_v4"} and (
            guide.get("turbo_lora_applied") is False or acceleration_ready is False
        ):
            # A missing/incompatible adapter must never leave an unsafe 4-step
            # setting behind. The native fallback uses a conservative count.
            values["steps"] = 8
        descriptions = {
            "quality": "稳定质量：20 步，不强制启用缓存",
            "quality_sage": "质量优先加速：20 步 + SageAttention，动态分层加载，关闭 Turbo LoRA 与 EasyCache",
            "quality_two_stage": "质量优先二采样：匹配 LoRA 首采 4 步 + 训练型 3D latent 放大 + 低 sigma 二采；自动保留音频 latent",
            "fl_quality_fast_v4": "高清快速（v4 8 步）：FL/T2V 后端使用社区 v4 LoRA，单采 8 步 + simple/Euler，不启用 latent 二采",
            "fast_4step": "极速 4 步：T2VA/FL2VA/I2VA/L2VA 使用官方 H3 Turbo；REF2VA/音色参考使用官方 Ref2VA Turbo + 原生 Euler",
            "reference_fast": "参考图加速：6 步 + Sage + EasyCache",
            "ref_quality_native": "参考高清（原生 20 步）：REF2VA/H3 音色参考原生 20 步 + SageAttention，不使用 Turbo/二采",
            "ref_fast_4step": "参考极速（官方 4 步）：REF2VA/H3 音色参考使用官方 Ref2VA Turbo，4 步原生 Euler",
            "low_vram": "低显存：8 步 + Sage，使用 ComfyUI 动态分层加载，关闭缓存",
            "low_vram_two_stage": "低显存二采：4–6 秒 FHD，按时长缩小首采网格 + RealESRGAN X2 细节重建",
            "custom": "自定义：保守默认值，可在设置子图中调整",
        }
        if mode == "T2VA" and name == "fast_4step":
            descriptions[name] = "T2VA 极速 4 步：官方 H3 Turbo + Sage，关闭 EasyCache 以减少细节损失"
        if name == "low_vram_two_stage" and acceleration_ready is not True:
            raise RuntimeError(
                "低显存二采训练型 LoRA 或精确分块注意力未就绪，已停止任务；"
                "不会静默回退为模糊单采"
            )
        two_stage_enabled = name in TWO_STAGE_PERFORMANCE_PRESETS and acceleration_ready is True
        if name in TWO_STAGE_PERFORMANCE_PRESETS and not two_stage_enabled:
            values["two_stage_split_step"] = 0
            values["two_stage_scale"] = 1.0
            descriptions[name] = "二采精确低显存注意力不可用，已在采样前回退为 8 步单采，避免第二阶段显存溢出"
            if name == "quality_two_stage":
                _reconcile_two_stage_fallback_geometry(guide)
        guide["two_stage_enabled"] = two_stage_enabled
        guide["two_stage_split_step"] = int(values.get("two_stage_split_step", 0))
        guide["two_stage_scale"] = float(values.get("two_stage_scale", 1.0))
        guide["two_stage_status"] = "待执行" if guide["two_stage_enabled"] else "旁路"
        if downgraded:
            descriptions[name] = f"{mode} / {voice_mode} 不支持所选预设，已回退稳定质量"
        return values["steps"], values["use_sage"], values["use_cache"], descriptions[name]


def _safe_guide_preset(guide):
    requested = guide.get("performance_preset", "quality")
    name = PRESET_LABELS.get(requested, requested)
    if name == "smart_free_1080p":
        raise ValueError("免费智能 1080p 必须先由导演台解析为具体性能预设")
    mode = guide.get("mode")
    voice_mode = guide.get("voice_mode", "none")
    backend = guide.get("resolved_backend")
    if backend == "ref2va_model" and name in TWO_STAGE_PERFORMANCE_PRESETS:
        raise ValueError(
            "REF2VA 不支持训练型二采，请使用 quality_sage、low_vram 或 ref_fast_4step"
        )
    if backend == "ref2va_model" and name == "fl_quality_fast_v4":
        return "quality", True
    if backend == "fl2va_model" and name in {"ref_quality_native", "ref_fast_4step"}:
        return "quality", True
    if mode and name != "custom" and name not in allowed_performance_presets(mode, voice_mode):
        return "quality", True
    return name, False


def acceleration_plan(guide):
    """Return the model/LoRA/sampler contract shared by U11 nodes."""
    preset, _ = _safe_guide_preset(guide)
    backend = guide.get("resolved_backend", "fl2va_model")
    route = resolve_two_stage_route(guide) if preset in TWO_STAGE_PERFORMANCE_PRESETS else "bypass"
    use_two_stage = preset in TWO_STAGE_PERFORMANCE_PRESETS and route != "bypass"
    use_turbo = preset in {"fast_4step", "fl_quality_fast_v4", "ref_fast_4step"} or use_two_stage
    # Official H3 Turbo graphs use ComfyUI's stock Euler sampler for every
    # backend. The legacy custom sampler is intentionally bypassed.
    use_turbo_sampler = False
    if guide.get("turbo_lora_applied") is False:
        use_turbo = False
        use_turbo_sampler = False
    plan = {
        "use_turbo_lora": use_turbo,
        "use_turbo_sampler": use_turbo_sampler,
        "lora_name": REF2VA_TURBO_LORA_NAME if backend == "ref2va_model" else TURBO_LORA_NAME,
        "lora_strength": float(preset_values(preset, backend).get("lora_strength", 1.0)),
        "backend": backend,
        "preset": preset,
        "route": route,
    }
    if preset == "fl_quality_fast_v4":
        plan["lora_name"] = FL_V4_LORA_NAME
    elif preset == "ref_quality_native":
        plan["lora_name"] = None
    if use_two_stage:
        if route == "trained_latent_ref":
            plan.update(
                first_lora_name=REF_STAGE_LORA,
                first_lora_strength=0.75,
                second_lora_name=REF_STAGE_LORA,
                second_lora_strength=0.75,
            )
        else:
            plan.update(
                first_lora_name=FL_STAGE1_LORA,
                first_lora_strength=0.75,
                second_lora_name=FL_STAGE2_LORA,
                second_lora_strength=0.70,
            )
    return plan


def scheduler_plan(guide):
    """Resolve the schedule used by the selected H3 sampling route."""
    preset, _ = _safe_guide_preset(guide)
    if preset in TWO_STAGE_PERFORMANCE_PRESETS and guide.get("two_stage_enabled") is not False:
        route = resolve_two_stage_route(guide)
        if route == "trained_latent_ref":
            return {
                "scheduler": "simple",
                "steps": 8,
                "split_step": 4,
                "refine_reference_tail": True,
            }
        if route == "trained_latent_fl":
            return {
                "scheduler": "beta",
                "steps": 8,
                "split_step": 4,
                "refine_reference_tail": False,
            }
    return {
        "scheduler": "simple",
        "steps": int(_runtime_preset_values(guide, preset)["steps"]),
        "split_step": 0,
        "refine_reference_tail": False,
    }


def sampler_name_for_guide(guide, requested):
    """Use the stock Euler sampler for every official Turbo contract."""
    plan = acceleration_plan(guide)
    if plan["preset"] in TWO_STAGE_PERFORMANCE_PRESETS or (
        plan["preset"] in {"fast_4step", "fl_quality_fast_v4", "ref_fast_4step"}
        and plan["use_turbo_lora"]
    ):
        return "euler"
    return requested


def sampler_route(guide):
    return "h3_turbo" if acceleration_plan(guide)["use_turbo_sampler"] else "native"


def _apply_reference_sigma_refiner(sigmas):
    """Invoke YCNodes' U16-compatible H3 reference tail refiner."""
    import nodes as comfy_nodes

    mappings = getattr(comfy_nodes, "NODE_CLASS_MAPPINGS", {})
    node_class = mappings.get(SIGMA_REFINER_NODE_ID)
    if node_class is None:
        raise RuntimeError("缺少 H3SigmaRefiner 节点")
    node = node_class()
    function_name = getattr(node, "FUNCTION", getattr(node_class, "FUNCTION", "refine_sigmas"))
    function = getattr(node, function_name)
    return _node_model(function(sigmas, 1, 0.7, 0.0, "cosine"))


class MiniMaxH3SchedulerRouter:
    """Create the exact FL or Reference schedule selected by the Director."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "steps": ("INT", {"default": 8, "min": 1, "max": 100}),
            "guide": ("MINIMAX_H3_DIRECTOR_PLUS_GUIDE",),
        }}

    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("匹配Sigma序列",)
    FUNCTION = "route"
    CATEGORY = "MiniMax H3 导演台 Plus"

    def route(self, model, steps, guide):
        from comfy_extras.nodes_custom_sampler import BasicScheduler

        plan = scheduler_plan(guide)
        resolved_steps = int(plan["steps"] if plan["split_step"] else steps)
        sigmas = _node_model(BasicScheduler.execute(
            model,
            plan["scheduler"],
            resolved_steps,
            1.0,
        ))
        if plan["refine_reference_tail"]:
            sigmas = _apply_reference_sigma_refiner(sigmas)
        guide["scheduler_name"] = plan["scheduler"]
        guide["scheduler_sigma_count"] = len(sigmas)
        guide["two_stage_split_step"] = int(plan["split_step"])
        LOGGER.info(
            "[H3 scheduler] route=%s scheduler=%s steps=%s sigmas=%s split=%s refiner=%s",
            resolve_two_stage_route(guide),
            plan["scheduler"],
            resolved_steps,
            len(sigmas),
            plan["split_step"],
            plan["refine_reference_tail"],
        )
        return (sigmas,)


def _reset_h3_acceleration_state(guide, error=None):
    """Clear stale LoRA route results before a new acceleration attempt."""
    guide.pop("turbo_lora_applied", None)
    guide.pop("turbo_lora_error", None)
    guide.pop("first_lora_name", None)
    guide.pop("second_lora_name", None)
    guide["turbo_sampler_applied"] = False
    guide["two_stage_enabled"] = False
    guide["two_stage_status"] = "旁路"
    guide.pop("two_stage_fallback", None)
    guide["two_stage_split_step"] = 0
    guide["two_stage_scale"] = 1.0
    if error is not None:
        guide["turbo_lora_applied"] = False
        guide["turbo_lora_error"] = str(error)


def _reconcile_two_stage_fallback_geometry(guide):
    """Make a failed two-stage route behave like the actual native sampler."""
    if not isinstance(guide, dict):
        return

    native_width = int(
        guide.get("native_width")
        or guide.get("width")
        or guide.get("first_stage_width")
        or 0
    )
    native_height = int(
        guide.get("native_height")
        or guide.get("height")
        or guide.get("first_stage_height")
        or 0
    )
    if native_width <= 0 or native_height <= 0:
        return

    requested_width = int(
        guide.get("requested_width") or guide.get("target_width") or native_width
    )
    requested_height = int(
        guide.get("requested_height") or guide.get("target_height") or native_height
    )
    postprocess_mode = str(guide.get("postprocess_mode") or "native")
    if requested_width == native_width and requested_height == native_height:
        postprocess_path = "native_bypass"
        target_width, target_height = native_width, native_height
    elif requested_width < native_width or requested_height < native_height:
        postprocess_path = "downscale"
        target_width, target_height = requested_width, requested_height
    elif postprocess_mode in {"lanczos", "ai_upscale", "rtx_vsr"}:
        postprocess_path = postprocess_mode
        target_width, target_height = requested_width, requested_height
    else:
        postprocess_path = "native_bypass"
        target_width, target_height = native_width, native_height

    guide["width"] = native_width
    guide["height"] = native_height
    guide["native_width"] = native_width
    guide["native_height"] = native_height
    guide["first_stage_width"] = native_width
    guide["first_stage_height"] = native_height
    guide["second_stage_width"] = native_width
    guide["second_stage_height"] = native_height
    guide["postprocess_source_width"] = native_width
    guide["postprocess_source_height"] = native_height
    guide["target_width"] = target_width
    guide["target_height"] = target_height
    guide["postprocess_path"] = postprocess_path
    guide["upscale_required"] = postprocess_path in {"lanczos", "ai_upscale", "rtx_vsr"}
    guide["upscale_method"] = {
        "rtx_vsr": "rtx_vsr",
        "ai_upscale": "comfy_upscale_model",
        "lanczos": "lanczos",
        "downscale": "cpu_bicubic",
        "native_bypass": "none",
    }[postprocess_path]
    guide["final_upscale_scale_x"] = target_width / native_width
    guide["final_upscale_scale_y"] = target_height / native_height
    guide["final_upscale_scale"] = max(
        guide["final_upscale_scale_x"], guide["final_upscale_scale_y"]
    )
    guide["max_final_vsr_scale"] = None
    guide["vram_safety_tier"] = "not_applicable"
    guide["quality_basis"] = "H3 原生（训练型二采不可用，已回退）"
    guide["required_assets"] = []
    guide["two_stage_fallback"] = True
    guide["resolved_two_stage_route"] = "bypass"


def _apply_two_stage_models(model, guide, plan, values):
    """Build independently patched first/second models for trained sampling."""
    original_model = model
    guide["sage_requested"] = False
    guide["cache_requested"] = False
    guide["head_chunking_requested"] = True
    guide["sage_applied"] = False
    guide["easycache_applied"] = False
    guide["head_chunking_applied"] = False
    guide.pop("head_chunking_error", None)
    guide["minimax_head_chunks"] = int(values.get("minimax_head_chunks", 8))
    try:
        first_model = _load_lightx2v_lora(
            original_model,
            plan["first_lora_name"],
            strength=plan["first_lora_strength"],
        )
        if (
            plan["second_lora_name"] == plan["first_lora_name"]
            and plan["second_lora_strength"] == plan["first_lora_strength"]
        ):
            second_model = first_model
        else:
            second_model = _load_lightx2v_lora(
                original_model,
                plan["second_lora_name"],
                strength=plan["second_lora_strength"],
            )
        shared_second_model = second_model is first_model
        first_model = _apply_minimax_reuse_attention(first_model, guide)
        if shared_second_model:
            second_model = first_model
        else:
            second_model = _apply_minimax_reuse_attention(second_model, guide)
    except H3LoRAApplicationError as exc:
        _reset_h3_acceleration_state(guide, exc)
        guide["head_chunking_applied"] = False
        guide["resolved_two_stage_route"] = "bypass"
        LOGGER.error("[H3 two-stage models] official LoRA application failed: %s", exc)
        # A stale or version-mismatched official adapter must never abort the
        # whole workflow.  Return the untouched model and let the preset node
        # switch to the conservative native step count and single-pass path.
        return original_model, "训练型二采 LoRA 与当前 H3 模型不兼容，已回退原生采样", False, original_model
    except (ImportError, AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        _reset_h3_acceleration_state(guide, exc)
        guide["head_chunking_applied"] = False
        guide["head_chunking_error"] = str(exc)
        LOGGER.warning("[H3 two-stage models] route setup failed: %s", exc)
        return original_model, "训练型二采模型或精确分块补丁不可用，已在采样前安全旁路", False, original_model

    guide["turbo_lora_applied"] = True
    guide["head_chunking_applied"] = True
    guide["two_stage_enabled"] = True
    guide["two_stage_status"] = "待执行"
    guide["two_stage_split_step"] = int(values.get("two_stage_split_step", 0))
    guide["two_stage_scale"] = float(values.get("two_stage_scale", 1.0))
    guide["first_lora_name"] = plan["first_lora_name"]
    guide["second_lora_name"] = plan["second_lora_name"]
    guide["resolved_two_stage_route"] = plan["route"]
    LOGGER.info(
        "[H3 two-stage models] route=%s first=%s@%.2f second=%s@%.2f head_chunks=%s",
        plan["route"],
        plan["first_lora_name"],
        plan["first_lora_strength"],
        plan["second_lora_name"],
        plan["second_lora_strength"],
        guide["minimax_head_chunks"],
    )
    return first_model, "匹配 LoRA 双模型与精确分块补丁已启用", True, second_model


def _apply_acceleration(model, guide):
    """Apply all requested accelerators to the model that will actually sample."""
    _reset_h3_acceleration_state(guide)
    plan = acceleration_plan(guide)
    guide["resolved_two_stage_route"] = plan["route"]
    values = _runtime_preset_values(guide, plan["preset"])
    sage_requested = bool(values.get("use_sage"))
    cache_requested = bool(values.get("use_cache"))
    head_chunking_requested = bool(values.get("use_head_chunking"))
    if head_chunking_requested:
        guide["minimax_head_chunks"] = int(values.get("minimax_head_chunks", 8))
    guide["sage_requested"] = sage_requested
    guide["cache_requested"] = cache_requested
    guide["head_chunking_requested"] = head_chunking_requested
    guide["sage_applied"] = False
    guide["easycache_applied"] = False
    guide["head_chunking_applied"] = False
    guide.pop("sage_error", None)
    guide.pop("easycache_error", None)
    guide.pop("head_chunking_error", None)

    original_model = model
    if plan["preset"] in TWO_STAGE_PERFORMANCE_PRESETS and plan.get("route") != "bypass":
        return _apply_two_stage_models(original_model, guide, plan, values)
    if plan["use_turbo_lora"]:
        try:
            model = _load_lightx2v_lora(
                model,
                plan["lora_name"],
                strength=plan.get("lora_strength", 1.0),
                low_vram=plan["preset"] == "low_vram",
            )
            guide["turbo_lora_applied"] = True
        except H3LoRAApplicationError as exc:
            _reset_h3_acceleration_state(guide, exc)
            LOGGER.error("[H3 acceleration] official LoRA application failed: %s", exc)
            return original_model, "Turbo LoRA 与当前 H3 模型不兼容，已回退原生采样", False, original_model
        except (AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
            _reset_h3_acceleration_state(guide, exc)
            return original_model, "Turbo LoRA 与当前模型不兼容，已回退原生采样", False, original_model
    else:
        guide["turbo_lora_applied"] = False

    if sage_requested:
        try:
            model = _apply_sage_attention(model, guide)
            guide["sage_applied"] = True
        except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            guide["sage_error"] = str(exc)
            LOGGER.warning("[H3 acceleration] SageAttention unavailable: %s", exc)
            return original_model, "SageAttention 加速失败，已回退原生模型", False, original_model

    if head_chunking_requested:
        try:
            model = _apply_minimax_reuse_attention(model, guide)
            guide["head_chunking_applied"] = True
        except (ImportError, AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
            guide["head_chunking_error"] = str(exc)
            LOGGER.warning("[H3 acceleration] exact head chunking unavailable: %s", exc)
            return original_model, "二采精确低显存注意力不可用，将在采样前回退单采", False, original_model

    if cache_requested:
        try:
            model = _apply_easy_cache(model, guide)
            guide["easycache_applied"] = True
        except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            guide["easycache_error"] = str(exc)
            LOGGER.warning("[H3 acceleration] EasyCache unavailable: %s", exc)
            return original_model, "EasyCache 加速失败，已回退原生模型", False, original_model

    requested = plan["use_turbo_lora"] or sage_requested or cache_requested or head_chunking_requested
    ready = (not plan["use_turbo_lora"] or guide["turbo_lora_applied"]) and (
        not sage_requested or guide["sage_applied"]
    ) and (not cache_requested or guide["easycache_applied"]) and (
        not head_chunking_requested or guide["head_chunking_applied"]
    )
    if requested:
        label = "REF2VA Turbo 4 步 LoRA" if plan["backend"] == "ref2va_model" else "H3 Turbo 4 步 LoRA"
        if plan["preset"] == "quality_sage":
            label = "质量优先 SageAttention"
        elif plan["preset"] == "fl_quality_fast_v4":
            label = "FL/T2V v4 高清快速 8 步 LoRA"
        elif plan["preset"] == "ref_quality_native":
            label = "REF2VA 参考高清原生 20 步 + SageAttention"
        elif plan["preset"] == "ref_fast_4step":
            label = "REF2VA 参考极速官方 4 步 LoRA"
        elif plan["preset"] == "quality_two_stage":
            label = "质量优先二采样（H3 专用 latent 二采）"
        elif plan["preset"] == "low_vram_two_stage":
            label = "低显存二采（4–6 秒 FHD 专用）"
        elif not plan["use_turbo_lora"]:
            label = "参考图 Sage/EasyCache"
        LOGGER.info(
            "[H3 acceleration] preset=%s backend=%s turbo=%s sage=%s easycache=%s head_chunks=%s",
            plan["preset"], plan["backend"], guide["turbo_lora_applied"], guide["sage_applied"], guide["easycache_applied"], guide["head_chunking_applied"],
        )
        return model, f"{label} 已启用；加速与低显存状态已写入指南", ready, model
    return model, "当前预设保持原生模型", False, model


class MiniMaxH3SamplerRouter:
    """Select the FL2VA Turbo sampler only when its LoRA is active."""

    @classmethod
    def INPUT_TYPES(cls):
        try:
            import comfy.samplers
            sampler_names = list(comfy.samplers.SAMPLER_NAMES)
        except ImportError:
            sampler_names = ["res_multistep"]
        if "res_multistep" not in sampler_names:
            sampler_names.insert(0, "res_multistep")
        return {"required": {
            "sampler_name": (sampler_names, {"default": "res_multistep"}),
            "guide": ("MINIMAX_H3_DIRECTOR_PLUS_GUIDE",),
        }}

    RETURN_TYPES = ("SAMPLER",)
    RETURN_NAMES = ("实际采样器",)
    FUNCTION = "route"
    CATEGORY = "MiniMax H3 导演台 Plus"

    def route(self, sampler_name, guide):
        if sampler_route(guide) == "h3_turbo":
            try:
                MiniMaxH3TurboSampler = _turbo_class("MiniMaxH3TurboSampler")
                result = MiniMaxH3TurboSampler().get_sampler()
                guide["turbo_sampler_applied"] = True
                return result
            except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
                guide["turbo_sampler_applied"] = False
                guide["turbo_sampler_error"] = str(exc)
        # Official Ref2VA Turbo is specified with the stock Euler sampler.
        # The saved U11 graph keeps its FL2VA-compatible default widget, so
        # override it here instead of requiring a manual node edit.
        sampler_name = sampler_name_for_guide(guide, sampler_name)
        import comfy.samplers
        guide.setdefault("turbo_sampler_applied", False)
        return (comfy.samplers.sampler_object(sampler_name),)


class MiniMaxH3MemoryAwareSampler:
    """Native SamplerCustomAdvanced with a scoped H3 memory policy."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noise": ("NOISE",),
                "guider": ("GUIDER",),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "latent_image": ("LATENT",),
            },
            "optional": {"guide": ("MINIMAX_H3_DIRECTOR_PLUS_GUIDE",)},
        }

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("输出Latent", "去噪Latent")
    FUNCTION = "execute"
    CATEGORY = "MiniMax H3 导演台 Plus"

    def execute(self, noise, guider, sampler, sigmas, latent_image, guide=None):
        from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced

        with memory_policy(guide or {}):
            return SamplerCustomAdvanced.execute(noise, guider, sampler, sigmas, latent_image)


class MiniMaxH3AccelerationRouter:
    """Apply only acceleration assets compatible with the resolved backend."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "guide": ("MINIMAX_H3_DIRECTOR_PLUS_GUIDE",),
        }}

    RETURN_TYPES = ("MODEL", "STRING", "BOOLEAN", "MODEL")
    RETURN_NAMES = ("第一阶段模型", "加速说明", "加速成功", "第二阶段模型")
    FUNCTION = "apply"
    CATEGORY = "MiniMax H3 导演台 Plus"

    def apply(self, model, guide):
        return _apply_acceleration(model, guide)
