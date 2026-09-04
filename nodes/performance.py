"""Official, backend-aware acceleration for the U11 Director Plus workflow."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib
import importlib.util
import logging
import os
import sys
from pathlib import Path

from .h3_reuse_attention import (
    apply_h3_reuse_attention,
    auto_chunk_tiers,
    free_vram_bytes,
)
from .schema import (
    REFERENCE_UNSAFE_FALLBACKS,
    TWO_STAGE_PERFORMANCE_PRESETS,
    allowed_performance_presets,
)
from .two_stage_assets import (
    FL_STAGE1_LORA,
    FL_STAGE2_LORA,
    REF_DETAIL_LORA_CHAIN,
    REF_EXTRA_LORA_CHAIN,
    REF_STAGE_LORA,
    SIGMA_REFINER_NODE_ID,
    V4_TURBO_LORA,
    V4_TURBO_LORA_PRUNED,
    resolve_registered_model_name,
    resolve_two_stage_route,
)


LOGGER = logging.getLogger("MiniMaxH3.DirectorPlus")

_UI_ATTENTION_CHUNK_TIERS = {
    2: (2, 4, 4),
    4: (4, 8, 8),
    8: (8, 16, 16),
    16: (16, 32, 32),
}

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
    # U22 recipe: 12-step budget split 8+4 so the 8-step stage-1 LoRA gets its
    # full denoising schedule and the 4-step stage-2 LoRA does the redraw.
    "quality_two_stage": {
        "steps": 12,
        "two_stage_split_step": 8,
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
    # RTX 3070-class route: keep the trained 8+4 contract but shrink the
    # first/second grids in the director and stage both samplers through
    # ComfyUI LOW_VRAM. The deterministic planner limits this to six seconds.
    "low_vram_two_stage": {
        "steps": 12,
        "two_stage_split_step": 8,
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
FL_V4_LORA_NAME = V4_TURBO_LORA
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


def _lora_effect_count(model):
    """Count regular patches and runtime injections created by an H3 LoRA."""
    injections = getattr(model, "injections", {})
    if not isinstance(injections, dict):
        injections = {}
    injection_count = sum(
        len(entries) if isinstance(entries, (list, tuple)) else int(bool(entries))
        for entries in injections.values()
    )
    return _patch_entry_count(model) + injection_count


def _lora_name_candidates(lora_name):
    """Return acceptable file names for a requested adapter, in load order.

    The turbo v4 adapter circulates under two names.  drbaph's pruned
    ComfyUI conversion is tried first: it loads through ComfyUI's native
    LoRA path with no runtime-injection hooks (the exact file the U22
    recipe validates), while the original file falls back to the slower
    H3-specific bypass loader.
    """
    if lora_name == V4_TURBO_LORA:
        return (V4_TURBO_LORA_PRUNED, V4_TURBO_LORA)
    return (lora_name,)


def _load_lightx2v_lora(model, lora_name=None, strength=1.0, low_vram=False):
    """Apply an official H3 adapter, with a H3-specific fallback for key maps."""
    requested_lora_name = lora_name or TURBO_LORA_NAME
    resolved_name = None
    for candidate in _lora_name_candidates(requested_lora_name):
        resolved_name = resolve_registered_model_name("loras", candidate)
        if resolved_name is not None:
            break
    if resolved_name is None:
        raise H3LoRAApplicationError(
            "缺少 H3 Turbo LoRA: " + " 或 ".join(_lora_name_candidates(requested_lora_name))
        )
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
        core_error = H3LoRAApplicationError(
            f"官方 H3 Turbo LoRA 未应用任何模型补丁: {resolved_name}"
        )
        try:
            # H3 adapters use module names that are not covered by ComfyUI's
            # generic UNet key map. The bundled H3 loader normalizes those keys
            # and supports both bypass and merged low-VRAM application.
            h3_loader = _turbo_class("MiniMaxH3TurboLoRA")()
            fallback_model = h3_loader.apply_lora(
                model,
                resolved_name,
                float(strength),
                low_vram=bool(low_vram),
            )[0]
            fallback_delta = _lora_effect_count(fallback_model) - _lora_effect_count(model)
            if fallback_delta <= 0:
                raise H3LoRAApplicationError(
                    f"H3 专用加载器也未产生补丁或运行时注入（delta={fallback_delta}）"
                )
            LOGGER.warning(
                "[H3 LoRA] 通用加载器未匹配 %s，已切换 H3 专用加载器 effect_delta=%s；"
                "剪枝版 H3 底座建议改用 %s（drbaph/MiniMax-H3-Turbo-Lora-ComfyUI），"
                "原生 LoRA 路径速度更快、画质更稳",
                resolved_name,
                fallback_delta,
                V4_TURBO_LORA_PRUNED,
            )
            return fallback_model
        except Exception as exc:
            raise H3LoRAApplicationError(
                f"{core_error}；H3 专用加载器失败: {exc}"
            ) from exc
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


def _free_vram_bytes():
    return free_vram_bytes()


def _positive_int(value):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _resolve_attention_chunks(guide, forced_tier=None):
    """Resolve UI/env/auto/preset chunk values without mutating the guide."""
    preset = PRESET_LABELS.get(
        guide.get("performance_preset", "quality"),
        guide.get("performance_preset", "quality"),
    )
    if preset == "low_vram_two_stage" and forced_tier is None:
        forced_tier = (16, 32, 32)
    if forced_tier is not None:
        auto = tuple(int(value) for value in forced_tier)
    else:
        free = _free_vram_bytes()
        if free is not None:
            auto = auto_chunk_tiers(int(free))
        else:
            auto = (
                max(1, int(guide.get("minimax_head_chunks", 8))),
                16,
                16,
            )

    head_env = _positive_int(os.environ.get("MMH3_HEAD_CHUNKS"))
    projection_env = _positive_int(os.environ.get("MMH3_PROJ_CHUNKS"))
    mlp_env = _positive_int(os.environ.get("MMH3_MLP_CHUNKS"))
    ui_value = _positive_int(guide.get("minimax_head_chunks_ui"))
    if ui_value not in _UI_ATTENTION_CHUNK_TIERS:
        ui_value = None

    head, projection, mlp = auto
    if ui_value is not None:
        head, projection, mlp = _UI_ATTENTION_CHUNK_TIERS[ui_value]
    if head_env is not None:
        head = head_env
    if projection_env is not None:
        projection = projection_env
    if mlp_env is not None:
        mlp = mlp_env

    if any(value is not None for value in (head_env, projection_env, mlp_env)):
        source = "env"
    elif ui_value is not None:
        source = "ui"
    elif forced_tier is not None or _free_vram_bytes() is None:
        source = "preset"
    else:
        source = "auto"
    return (head, projection, mlp), source


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
            forced_tier = (16, 32, 32) if preset == "low_vram" else None
            chunks, _source = _resolve_attention_chunks(guide, forced_tier=forced_tier)
            chunks = chunks[0]
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
    chunks, source = _resolve_attention_chunks(guide)
    head_chunks, projection_chunks, mlp_chunks = chunks
    guide["minimax_head_chunks"] = head_chunks
    guide["minimax_projection_chunks"] = projection_chunks
    guide["minimax_mlp_chunks"] = mlp_chunks
    guide["attention_chunks_source"] = source
    return apply_h3_reuse_attention(
        model,
        head_chunks=head_chunks,
        projection_chunks=projection_chunks,
        mlp_chunks=mlp_chunks,
        source=source,
    )


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
            "quality_two_stage": "质量优先二采样：匹配 LoRA 首采 8 步（共 12 步）+ 训练型 3D latent 放大 + 4 步低 sigma 重绘；自动保留音频 latent；REF2VA 使用社区验证的 turbo v4 配方",
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
            descriptions[name] = "二采精确低显存注意力不可用，已在采样前回退为 12 步原生单采，避免第二阶段显存溢出"
            if name == "quality_two_stage":
                _reconcile_two_stage_fallback_geometry(guide)
        guide["two_stage_enabled"] = two_stage_enabled
        guide["two_stage_split_step"] = int(values.get("two_stage_split_step", 0))
        if two_stage_enabled:
            # Keep the upscaler multiplier aligned with the VRAM budget plan
            # (e.g. U22 recipe 544->1088 = 2.0x) instead of the preset constant.
            guide["two_stage_scale"] = _planned_two_stage_scale(guide, values)
        else:
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
        # quality_two_stage is allowed on REF2VA (U22 turbo-v4 8+4 recipe);
        # only the unvalidated 8GB low-VRAM variant still falls back.
        if name == "quality_two_stage":
            guide["resolved_performance_preset"] = name
            return name, False
        fallback = REFERENCE_UNSAFE_FALLBACKS.get(name, "quality_sage")
        guide["resolved_performance_preset"] = fallback
        guide["two_stage_enabled"] = False
        guide["two_stage_status"] = "旁路"
        guide["two_stage_fallback"] = True
        guide.setdefault("warnings", []).append(
            f"REF2VA 不支持训练型二采 {name}，已自动切换为兼容路线 {fallback}。"
        )
        return fallback, True
    if backend == "ref2va_model" and name == "fl_quality_fast_v4":
        return "quality", True
    if backend == "fl2va_model" and name in {"ref_quality_native", "ref_fast_4step"}:
        return "quality", True
    if mode and name != "custom" and name not in allowed_performance_presets(mode, voice_mode):
        return "quality", True
    guide["resolved_performance_preset"] = name
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
            # U22 REF2VA recipe: one 12-step turbo v4 adapter drives both the
            # 8-step composition pass and the 4-step low-sigma redraw.
            plan.update(
                first_lora_name=V4_TURBO_LORA,
                first_lora_strength=1.0,
                second_lora_name=V4_TURBO_LORA,
                second_lora_strength=1.0,
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
            # U22-validated REF2VA recipe: beta schedule, 12 turbo-v4 steps,
            # 8 high-sigma composition steps then a 4-step low-sigma redraw on
            # the 1.5x learned latent grid.  The legacy YCNodes tail refiner
            # belonged to the banned 4-step ref route and stays off.
            return {
                "scheduler": "beta",
                "steps": 12,
                "split_step": 8,
                "refine_reference_tail": False,
            }
        if route == "trained_latent_fl":
            # The stage-1 LoRA is trained for 8 steps: give the composition
            # pass the full 8 of 12 and let the 4-step stage-2 LoRA redraw.
            return {
                "scheduler": "beta",
                "steps": 12,
                "split_step": 8,
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
        # A legacy REF2VA guide may carry an 8-step trained-two-stage value.
        # After normalizing it to quality_sage, use the normalized 20-step
        # schedule instead of trusting the stale connected widget value.
        resolved_steps = int(
            plan["steps"]
            if plan["split_step"] or guide.get("two_stage_fallback")
            else steps
        )
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
        resolved_route = (
            resolve_two_stage_route(guide)
            if plan["split_step"]
            else "bypass"
        )
        LOGGER.info(
            "[H3 scheduler] route=%s scheduler=%s steps=%s sigmas=%s split=%s refiner=%s",
            resolved_route,
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
    elif postprocess_mode in {"lanczos", "ai_upscale", "video_sr", "rtx_vsr"}:
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
    guide["upscale_required"] = postprocess_path in {"lanczos", "ai_upscale", "video_sr", "rtx_vsr"}
    guide["upscale_method"] = {
        "rtx_vsr": "rtx_vsr",
        "ai_upscale": "comfy_upscale_model",
        "video_sr": "seedvr2",
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


def _planned_two_stage_scale(guide, values):
    """Prefer the VRAM budget plan's first->second ratio over preset constants.

    The learned latent upscaler multiplier must match the grid the director
    actually planned (U22 recipe: 544x960 -> 1088x1920 = 2.0x).  The preset
    constant only remains as a fallback for guides without a budget plan.
    """
    first_stage_width = int(guide.get("first_stage_width") or 0)
    second_stage_width = int(guide.get("second_stage_width") or 0)
    if first_stage_width > 0 and second_stage_width > first_stage_width:
        return float(second_stage_width) / float(first_stage_width)
    return float(values.get("two_stage_scale", 1.0))


def _env_flag(name, default=False):
    """Parse a boolean environment switch with an explicit default."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "on", "yes"}


def _resolve_first_registered(category, names):
    """Return the first candidate file name registered with ComfyUI."""
    for name in names:
        if resolve_registered_model_name(category, name) is not None:
            return name
    return None


def _apply_ref_detail_loras(model, guide):
    """Stack the optional community LoRAs onto a trained two-stage model clone.

    Two chains: the U22 recipe adapters (MMH3_DETAIL_LORAS=0 disables) and
    the user-requested motion-continuity + cinematic add-ons
    (MMH3_EXTRA_LORAS=0 disables).  Each adapter is skipped with an info
    log when no candidate file is registered.
    """
    applied = []
    chain = []
    mode = str(guide.get("community_lora_mode") or "全部自动叠加")
    detail_on = _env_flag("MMH3_DETAIL_LORAS", True) and mode != "全部关闭"
    extra_on = _env_flag("MMH3_EXTRA_LORAS", True) and mode == "全部自动叠加"
    if mode != "全部自动叠加":
        LOGGER.info("[H3 two-stage models] 社区 LoRA 开关=%s（节点控件设置）", mode)
    if detail_on:
        chain.extend(REF_DETAIL_LORA_CHAIN)
    if extra_on:
        chain.extend(REF_EXTRA_LORA_CHAIN)
    for names, strength in chain:
        candidates = (names,) if isinstance(names, str) else tuple(names)
        resolved = _resolve_first_registered("loras", candidates)
        if resolved is None:
            LOGGER.info(
                "[H3 two-stage models] 可选社区 LoRA 未安装，跳过: %s",
                " 或 ".join(candidates),
            )
            continue
        try:
            model = _load_lightx2v_lora(model, resolved, strength=strength)
        except H3LoRAApplicationError as exc:
            LOGGER.info("[H3 two-stage models] 可选社区 LoRA 加载失败，跳过: %s", exc)
            continue
        applied.append(f"{resolved}@{strength}")
    guide["detail_loras_applied"] = applied
    return model


# SHA-256 of the raw adaln_t_table tensor bytes in the stock H3 diffusion
# checkpoints (measured locally on the safetensors payload).  The b25-49
# hybrid keeps the FL2VA table; the REF2VA pruned checkpoint is the one that
# weaves a cross-hatch texture (and audio impurities) under the dual-stage
# SageAttention kernels.  ComfyUI's "Native ops" log line is only the global
# quant-op registry and cannot identify which checkpoint was loaded — this
# fingerprint can.
_BASE_TABLE_SHA_FL2VA = "ac8727cdec52137c73878d004de5bd2a0e19227e8311e29ab3b68f328310e34e"
_BASE_TABLE_SHA_REF2VA = "c02a6c11888297688c1e6278185ea1f947023acfc69f9003bbcdcec9a229a8e7"


def _fingerprint_base_model(model):
    """Hash the tiny adaln_t_table tensor to identify the loaded checkpoint.

    Returns the hex digest, or None when the model object does not expose a
    readable state dict.  Fingerprinting must never break a run, so every
    failure path is swallowed.
    """
    try:
        inner = getattr(model, "model", None)
        state_dict = inner.state_dict() if inner is not None else None
        if not state_dict:
            return None
        for key, tensor in state_dict.items():
            if "adaln_t_table" not in key:
                continue
            t = tensor.detach() if hasattr(tensor, "detach") else tensor
            dequantize = getattr(t, "dequantize", None)
            if callable(dequantize):
                t = dequantize()
            array = t.cpu().contiguous().numpy()
            return hashlib.sha256(array.tobytes()).hexdigest()
    except Exception:  # noqa: BLE001 - diagnostics must stay side-effect free
        return None
    return None


def _apply_two_stage_models(model, guide, plan, values):
    """Build independently patched first/second models for trained sampling."""
    original_model = model
    # U22 recipe (validated on the user's RTX 4080): the INT8 hybrid base
    # checkpoint runs KJNodes' memory-efficient SageAttention on BOTH stages
    # cleanly — it is the difference between ~48 s/it and single-digit
    # seconds per step at FHD.  The pruned convrot/nvfp4 checkpoints instead
    # produce a woven cross-hatch texture under the INT8/FP8 Sage kernels, so
    # two escape hatches remain: MMH3_SECOND_STAGE_SAGE=0 keeps Sage on the
    # composition pass but runs the low-sigma redraw with exact head-reuse
    # attention, and MMH3_TWO_STAGE_SAGE=0 disables Sage entirely.
    guide["sage_requested"] = True
    guide["cache_requested"] = False
    guide["head_chunking_requested"] = True
    guide["sage_applied"] = False
    guide["second_stage_sage_applied"] = False
    guide["easycache_applied"] = False
    guide["head_chunking_applied"] = False
    guide["detail_loras_applied"] = []
    guide.pop("sage_error", None)
    guide.pop("head_chunking_error", None)
    guide["minimax_head_chunks"] = int(values.get("minimax_head_chunks", 8))
    base_table_sha = _fingerprint_base_model(original_model)
    guide["base_table_sha"] = base_table_sha
    if base_table_sha == _BASE_TABLE_SHA_REF2VA:
        LOGGER.warning(
            "[H3 two-stage models] 底模指纹 %s = REF2VA pruned convrot：该底模在双阶段 SageAttention 下会产生全画面编织纹与音频杂质，请把底模切换为 minimax/minimax_h3_hybrid_fl2va_ref2va_b25-49-int8.safetensors（FL2VA 系指纹 %s）",
            base_table_sha[:12],
            _BASE_TABLE_SHA_FL2VA[:12],
        )
    elif base_table_sha == _BASE_TABLE_SHA_FL2VA:
        LOGGER.info(
            "[H3 two-stage models] 底模指纹 %s = FL2VA 系（hybrid/官方 FL2VA），双阶段 Sage 干净路线",
            base_table_sha[:12],
        )
    elif base_table_sha:
        LOGGER.info(
            "[H3 two-stage models] 底模指纹 %s（未登记的变体，若出现编织纹请对照指纹库）",
            base_table_sha[:12],
        )
    else:
        LOGGER.info("[H3 two-stage models] 未能读取底模指纹（非标准模型对象），跳过底模识别")
    two_stage_sage = _env_flag("MMH3_TWO_STAGE_SAGE", True)
    second_stage_sage = two_stage_sage and _env_flag("MMH3_SECOND_STAGE_SAGE", True)
    try:
        shared_lora = (
            plan["second_lora_name"] == plan["first_lora_name"]
            and plan["second_lora_strength"] == plan["first_lora_strength"]
        )
        # 社区细节链对两条训练型路线都生效：REF 路线（U22 原配方）与 FL 路线
        # （T2VA/I2VA/FL2VA/L2VA）。fl2v turbo 只负责少步加速、不带细节增强，
        # 不挂链时 FL 系成片肉眼可见地比 REF2VA 软。
        detail_chain_route = plan["route"] in ("trained_latent_ref", "trained_latent_fl")
        # Only the mixed attention mode (Sage composition pass + exact
        # low-sigma redraw) needs two separate model clones; identical
        # LoRA+attention chains share a single clone to save VRAM.
        share_model = shared_lora and not (two_stage_sage and not second_stage_sage)
        first_model = _load_lightx2v_lora(
            original_model,
            plan["first_lora_name"],
            strength=plan["first_lora_strength"],
        )
        if detail_chain_route:
            first_model = _apply_ref_detail_loras(first_model, guide)
        if share_model:
            second_model = first_model
        else:
            second_model = _load_lightx2v_lora(
                original_model,
                plan["second_lora_name"],
                strength=plan["second_lora_strength"],
            )
            if detail_chain_route:
                second_model = _apply_ref_detail_loras(second_model, guide)
        shared_second_model = second_model is first_model
        if not two_stage_sage:
            first_model = _apply_minimax_reuse_attention(first_model, guide)
            if shared_second_model:
                second_model = first_model
            else:
                second_model = _apply_minimax_reuse_attention(second_model, guide)
            guide["head_chunking_applied"] = True
        else:
            try:
                sage_node = _kj_ltx_class("MiniMaxH3MemoryEfficientSageAttentionPatch")()
                first_model = _node_model(sage_node.execute(first_model))
                guide["sage_applied"] = True
            except (ImportError, AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
                guide["sage_error"] = str(exc)
                LOGGER.warning(
                    "[H3 two-stage models] SageAttention patch unavailable, falling back to reuse attention: %s",
                    exc,
                )
                first_model = _apply_minimax_reuse_attention(first_model, guide)
                if shared_second_model:
                    second_model = first_model
                else:
                    second_model = _apply_minimax_reuse_attention(second_model, guide)
                guide["head_chunking_applied"] = True
            else:
                if second_stage_sage:
                    if shared_second_model:
                        second_model = first_model
                    else:
                        second_model = _node_model(sage_node.execute(second_model))
                    guide["second_stage_sage_applied"] = True
                else:
                    try:
                        second_model = _apply_minimax_reuse_attention(second_model, guide)
                        guide["head_chunking_applied"] = True
                    except (ImportError, AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
                        guide["head_chunking_error"] = str(exc)
                        LOGGER.warning(
                            "[H3 two-stage models] exact second-stage attention failed, using SageAttention for both stages: %s",
                            exc,
                        )
                        second_model = _node_model(sage_node.execute(second_model))
                        guide["second_stage_sage_applied"] = True
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
    guide["two_stage_enabled"] = True
    guide["two_stage_status"] = "待执行"
    guide["two_stage_split_step"] = int(values.get("two_stage_split_step", 0))
    guide["two_stage_scale"] = _planned_two_stage_scale(guide, values)
    guide["first_lora_name"] = plan["first_lora_name"]
    guide["second_lora_name"] = plan["second_lora_name"]
    guide["resolved_two_stage_route"] = plan["route"]
    LOGGER.info(
        "[H3 two-stage models] route=%s first=%s@%.2f second=%s@%.2f sage=%s second_sage=%s detail_loras=%s head_chunks=%s scale=%.2f",
        plan["route"],
        plan["first_lora_name"],
        plan["first_lora_strength"],
        plan["second_lora_name"],
        plan["second_lora_strength"],
        guide["sage_applied"],
        guide["second_stage_sage_applied"],
        ",".join(guide["detail_loras_applied"]) or "none",
        guide["minimax_head_chunks"],
        guide["two_stage_scale"],
    )
    if guide["sage_applied"] and guide["second_stage_sage_applied"]:
        attention_note = "SageAttention 双阶段加速补丁"
    elif guide["sage_applied"]:
        attention_note = "一采 SageAttention 加速与二采全精度细节注意力"
    else:
        attention_note = "精确分块注意力补丁（SageAttention 已禁用或不可用）"
    detail_note = ""
    if guide["detail_loras_applied"]:
        detail_note = f"，叠加 {len(guide['detail_loras_applied'])} 个社区 LoRA"
    return first_model, f"匹配 LoRA 双模型与{attention_note}已启用{detail_note}", True, second_model


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
    guide["second_stage_sage_applied"] = False
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

    COMMUNITY_LORA_MODES = ("全部自动叠加", "仅 U22 细节链", "全部关闭")
    SECOND_STAGE_NOISE_MODES = ("注入新噪声（U22 同配方）", "不注入（旧行为）")
    ATTENTION_CHUNK_MODES = (
        "自动（按显存）",
        "2（最快）",
        "4",
        "8（均衡）",
        "16（最省显存）",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "guide": ("MINIMAX_H3_DIRECTOR_PLUS_GUIDE",),
            },
            "optional": {
                "community_loras": (
                    list(cls.COMMUNITY_LORA_MODES),
                    {
                        "default": "全部自动叠加",
                        "tooltip": "REF 二采链的社区 LoRA 叠加开关；仅 U22 细节链=关闭动作连续性/电影质感，全部关闭=只保留官方 turbo LoRA",
                    },
                ),
                "second_stage_noise": (
                    list(cls.SECOND_STAGE_NOISE_MODES),
                    {
                        "default": "注入新噪声（U22 同配方）",
                        "tooltip": "二采噪声：U22 在第二阶段按 split sigma 注入新噪声（默认，干净无编织纹）；不注入=旧行为，可能出编织纹/人脸漂移",
                    },
                ),
                "attention_chunks": (
                    list(cls.ATTENTION_CHUNK_MODES),
                    {
                        "default": "自动（按显存）",
                        "tooltip": "二采注意力/MLP 分块档位：数字越小越快但占显存越多；自动=按空闲显存选档",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL", "STRING", "BOOLEAN", "MODEL")
    RETURN_NAMES = ("第一阶段模型", "加速说明", "加速成功", "第二阶段模型")
    FUNCTION = "apply"
    CATEGORY = "MiniMax H3 导演台 Plus"

    def apply(
        self,
        model,
        guide,
        community_loras="全部自动叠加",
        second_stage_noise="注入新噪声（U22 同配方）",
        attention_chunks="自动（按显存）",
    ):
        guide["community_lora_mode"] = community_loras
        guide["second_stage_noise_mode"] = second_stage_noise
        selected_chunks = next((char for char in attention_chunks if char.isdigit()), None)
        guide["minimax_head_chunks_ui"] = int(selected_chunks) if selected_chunks else None
        return _apply_acceleration(model, guide)
