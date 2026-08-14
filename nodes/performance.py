"""Official, backend-aware acceleration for the U11 Director Plus workflow."""

from __future__ import annotations

from contextlib import contextmanager

PRESETS = {
    "quality": {"steps": 20, "use_sage": False, "use_cache": False, "interpolate": False},
    "fast_4step": {"steps": 4, "use_sage": True, "use_cache": True, "interpolate": False},
    "reference_fast": {"steps": 6, "use_sage": True, "use_cache": True, "interpolate": False},
    # The H3 text encoder and video VAE are large enough to exhaust an 8 GB
    # card before sampling starts.  CPU placement is intentional here; Sage
    # and EasyCache affect speed, not the peak model footprint.
    "low_vram": {
        "steps": 8,
        "use_sage": True,
        "use_cache": False,
        "interpolate": False,
        "clip_device": "cpu",
        "vae_device": "cpu",
    },
    "custom": {"steps": 20, "use_sage": False, "use_cache": False, "interpolate": False},
}

# Existing official FL2VA adapter and the separate official Ref2VA adapter.
TURBO_LORA_NAME = "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors"
REF2VA_TURBO_LORA_NAME = "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"

PRESET_LABELS = {
    "稳定质量": "quality",
    "极速4步": "fast_4step",
    "参考图加速": "reference_fast",
    "低显存": "low_vram",
    "自定义": "custom",
    # Keep the old mojibake labels loadable for existing saved workflows.
    "绋冲畾璐ㄩ噺": "quality",
    "鏋侀€?姝?": "fast_4step",
    "鍙傝€冨浘鍔犻€?": "reference_fast",
    "浣庢樉瀛?": "low_vram",
    "鑷畾涔?": "custom",
}


def _load_lightx2v_lora(model, lora_name=None, low_vram=False):
    """Apply an official H3 adapter with the H3-aware loader when available.

    The stock loader assumes ``model.diffusion_model`` and crashes on the
    direct ``MiniMaxH3Model`` wrapper used by pruned/int8 checkpoints. The
    bundled H3 Turbo node handles both layouts and curve-mode adapters.
    """
    import folder_paths

    lora_name = lora_name or TURBO_LORA_NAME
    if not folder_paths.get_full_path("loras", lora_name):
        raise RuntimeError(f"缺少 H3 Turbo LoRA: {lora_name}")
    try:
        turbo = _turbo_class("MiniMaxH3TurboLoRA")
        return turbo().apply_lora(model, lora_name, 1.0, low_vram)[0]
    except (ImportError, AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError):
        # Older installs may not include the companion H3 Turbo node.
        import nodes
        return nodes.LoraLoaderModelOnly().load_lora_model_only(model, lora_name, 1.0)[0]


def _turbo_class(name):
    import importlib.util
    from pathlib import Path
    import sys

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


def preset_values(name, backend=None):
    name = PRESET_LABELS.get(name, name)
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


@contextmanager
def memory_policy(guide):
    """Temporarily apply the workflow's low-VRAM policy to ComfyUI.

    ComfyUI normally restores its global state only at process start.  A
    context keeps this workflow self-contained: model loading and sampling
    see LOW_VRAM, while other workflows retain the state they had before.
    """
    preset = PRESET_LABELS.get(
        guide.get("performance_preset", "quality"),
        guide.get("performance_preset", "quality"),
    )
    if preset != "low_vram":
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
        name = PRESET_LABELS.get(guide.get("performance_preset", "quality"), guide.get("performance_preset", "quality"))
        values = preset_values(name, guide.get("resolved_backend"))
        if name == "fast_4step" and (
            guide.get("turbo_lora_applied") is False or acceleration_ready is False
        ):
            # A missing/incompatible adapter must never leave an unsafe 4-step
            # setting behind. The native fallback uses a conservative count.
            values["steps"] = 8
        descriptions = {
            "quality": "稳定质量：20 步，不强制启用缓存",
            "fast_4step": "极速 4 步：FL2VA 使用官方 H3 Turbo；REF2VA/音色参考使用官方 Ref2VA Turbo + 原生 Euler",
            "reference_fast": "参考图加速：6 步 + Sage + EasyCache",
            "low_vram": "低显存：8 步 + Sage，文本编码器与 VAE 使用 CPU，关闭缓存",
            "custom": "自定义：保守默认值，可在设置子图中调整",
        }
        return values["steps"], values["use_sage"], values["use_cache"], descriptions[name]


def acceleration_plan(guide):
    """Return the model/LoRA/sampler contract shared by U11 nodes."""
    preset = PRESET_LABELS.get(guide.get("performance_preset", "quality"), guide.get("performance_preset", "quality"))
    backend = guide.get("resolved_backend", "fl2va_model")
    use_turbo = preset == "fast_4step"
    # Official H3 Turbo graphs use ComfyUI's stock Euler sampler for every
    # backend. The legacy custom sampler is intentionally bypassed.
    use_turbo_sampler = False
    if guide.get("turbo_lora_applied") is False:
        use_turbo = False
        use_turbo_sampler = False
    return {
        "use_turbo_lora": use_turbo,
        "use_turbo_sampler": use_turbo_sampler,
        "lora_name": REF2VA_TURBO_LORA_NAME if backend == "ref2va_model" else TURBO_LORA_NAME,
        "backend": backend,
        "preset": preset,
    }


def sampler_route(guide):
    return "h3_turbo" if acceleration_plan(guide)["use_turbo_sampler"] else "native"


def _apply_acceleration(model, guide):
    plan = acceleration_plan(guide)
    if plan["use_turbo_lora"]:
        try:
            accelerated = _load_lightx2v_lora(
                model,
                plan["lora_name"],
                low_vram=plan["preset"] == "low_vram",
            )
        except (AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
            guide["turbo_lora_applied"] = False
            guide["turbo_lora_error"] = str(exc)
            return model, "Turbo LoRA 与当前模型封装不兼容，已回退原生采样（工作流不中断）", False
        guide["turbo_lora_applied"] = True
        label = "REF2VA 官方 Turbo 4 步 LoRA" if plan["backend"] == "ref2va_model" else "H3 Turbo 4 步 LoRA"
        return accelerated, f"{label} 已启用", True
    guide["turbo_lora_applied"] = False
    return model, "当前预设保持原生模型；Sage/EasyCache 由设置子图控制", False


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
        plan = acceleration_plan(guide)
        if plan["use_turbo_lora"] and plan["preset"] == "fast_4step":
            sampler_name = "euler"
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

    RETURN_TYPES = ("MODEL", "STRING", "BOOLEAN")
    RETURN_NAMES = ("加速后模型", "加速说明", "加速成功")
    FUNCTION = "apply"
    CATEGORY = "MiniMax H3 导演台 Plus"

    def apply(self, model, guide):
        return _apply_acceleration(model, guide)
