import pytest

from nodes import performance
from nodes.performance import (
    MiniMaxH3AccelerationRouter,
    MiniMaxH3MemoryAwareSampler,
    MiniMaxH3PerformancePreset,
    MiniMaxH3SamplerRouter,
    acceleration_plan,
    memory_policy,
    preset_values,
    sampler_route,
)


def test_fast_preset_exposes_four_step_sampling_contract():
    values = preset_values("fast_4step")
    assert values["steps"] == 4
    assert values["use_cache"] is True
    assert values["use_sage"] is True


def test_turbo_loader_prefers_h3_aware_adapter_for_pruned_models(monkeypatch):
    class FolderPaths:
        @staticmethod
        def get_full_path(category, name):
            assert category == "loras"
            return f"/models/{name}"

    class TurboLoRA:
        def apply_lora(self, model, name, strength, low_vram):
            return (f"turbo:{model}:{name}:{strength}:{low_vram}",)

    import sys
    monkeypatch.setitem(sys.modules, "folder_paths", FolderPaths)
    monkeypatch.setattr(performance, "_turbo_class", lambda name: TurboLoRA)

    result = performance._load_lightx2v_lora("model", "adapter.safetensors")

    assert result == "turbo:model:adapter.safetensors:1.0:False"


def test_turbo_injection_failure_is_not_masked_by_stock_lora_loader(monkeypatch):
    class FolderPaths:
        @staticmethod
        def get_full_path(category, name):
            return f"/models/{name}"

    class BrokenTurbo:
        def apply_lora(self, model, name, strength, low_vram):
            raise AttributeError("MiniMaxH3Model has no diffusion_model")

    class BrokenStockLoader:
        def load_lora_model_only(self, *args):
            raise AssertionError("stock loader must not receive H3 injection failures")

    import sys
    monkeypatch.setitem(sys.modules, "folder_paths", FolderPaths)
    monkeypatch.setitem(sys.modules, "nodes", type("Nodes", (), {"LoraLoaderModelOnly": BrokenStockLoader})())
    monkeypatch.setattr(performance, "_turbo_class", lambda name: BrokenTurbo)

    with pytest.raises(RuntimeError, match="Turbo LoRA 注入失败"):
        performance._load_lightx2v_lora("model", "adapter.safetensors")


def test_quality_preset_keeps_conservative_sampling():
    values = preset_values("quality")
    assert values["steps"] >= 10
    assert values["use_cache"] is False


def test_performance_node_reads_guide_preset():
    result = MiniMaxH3PerformancePreset().apply({"performance_preset": "reference_fast"})
    assert result[0] == 6
    assert result[1] is True


def test_fast_preset_falls_back_to_safe_steps_after_acceleration_failure():
    result = MiniMaxH3PerformancePreset().apply(
        {"performance_preset": "fast_4step"},
        acceleration_ready=False,
    )
    assert result[0] == 8


def test_low_vram_description_matches_disabled_cache():
    result = MiniMaxH3PerformancePreset().apply({"performance_preset": "low_vram"})
    assert "EasyCache" not in result[3]


@pytest.mark.parametrize("mode", ["T2VA", "I2VA", "FL2VA", "L2VA", "REF2VA"])
@pytest.mark.parametrize("preset", ["quality", "fast_4step", "reference_fast", "low_vram", "custom"])
def test_every_mode_has_a_defined_performance_contract(mode, preset):
    backend = "ref2va_model" if mode == "REF2VA" else "fl2va_model"
    values = preset_values(preset, backend=backend)
    plan = acceleration_plan({
        "mode": mode,
        "performance_preset": preset,
        "resolved_backend": backend,
    })

    assert values["steps"] >= 4
    assert values["use_sage"] is (preset in {"fast_4step", "reference_fast", "low_vram"})
    assert values["use_cache"] is (preset in {"fast_4step", "reference_fast"})
    assert plan["backend"] == backend
    assert plan["use_turbo_lora"] is (preset == "fast_4step")
    assert plan["lora_name"] == (
        performance.REF2VA_TURBO_LORA_NAME if backend == "ref2va_model" else performance.TURBO_LORA_NAME
    )


def test_low_vram_preset_uses_dynamic_safe_policy_without_cache():
    values = preset_values("low_vram")
    assert values["clip_device"] == "dynamic"
    assert values["vae_device"] == "dynamic"
    assert values["fish_device"] == "cpu"
    assert values["use_cache"] is False


def test_reference_fast_applies_sage_and_easycache_on_routed_model(monkeypatch):
    calls = []

    def apply_sage(model, guide):
        calls.append("sage")
        return f"{model}:sage"

    def apply_cache(model, guide):
        calls.append("cache")
        return f"{model}:cache"

    monkeypatch.setattr(performance, "_apply_sage_attention", apply_sage, raising=False)
    monkeypatch.setattr(performance, "_apply_easy_cache", apply_cache, raising=False)

    guide = {
        "mode": "REF2VA",
        "performance_preset": "reference_fast",
        "resolved_backend": "ref2va_model",
    }
    result = MiniMaxH3AccelerationRouter().apply("model", guide)

    assert result[0] == "model:sage:cache"
    assert calls == ["sage", "cache"]
    assert guide["sage_applied"] is True
    assert guide["easycache_applied"] is True
    assert result[2] is True


def test_reference_fast_uses_safe_steps_when_requested_acceleration_fails(monkeypatch):
    monkeypatch.setattr(
        performance,
        "_apply_sage_attention",
        lambda model, guide: (_ for _ in ()).throw(RuntimeError("sage unavailable")),
        raising=False,
    )
    guide = {
        "performance_preset": "reference_fast",
        "resolved_backend": "ref2va_model",
    }
    result = MiniMaxH3AccelerationRouter().apply("model", guide)
    steps = MiniMaxH3PerformancePreset().apply(guide, acceleration_ready=result[2])[0]

    assert result[0] == "model"
    assert result[2] is False
    assert guide["sage_applied"] is False
    assert steps == 8


def test_memory_policy_restores_comfy_state(monkeypatch):
    class VramState:
        NORMAL_VRAM = "normal"
        LOW_VRAM = "low"

    mm = type("ModelManagement", (), {"vram_state": VramState.NORMAL_VRAM, "VRAMState": VramState})()
    import sys
    monkeypatch.setitem(sys.modules, "comfy", type("Comfy", (), {"model_management": mm})())
    monkeypatch.setitem(sys.modules, "comfy.model_management", mm)

    with memory_policy({"performance_preset": "low_vram"}):
        assert mm.vram_state == VramState.LOW_VRAM
    assert mm.vram_state == VramState.NORMAL_VRAM


def test_memory_aware_sampler_exposes_guide_input():
    inputs = MiniMaxH3MemoryAwareSampler.INPUT_TYPES()
    assert "guide" in inputs["optional"]


def test_fl2va_fast_loads_existing_official_lora(monkeypatch):
    model = object()
    accelerated = object()
    calls = []
    monkeypatch.setattr(performance, "_load_lightx2v_lora", lambda value, name, **kwargs: calls.append((value, name)) or accelerated)
    monkeypatch.setattr(performance, "_apply_sage_attention", lambda value, guide: value)
    monkeypatch.setattr(performance, "_apply_easy_cache", lambda value, guide: value)

    result = MiniMaxH3AccelerationRouter().apply(
        model,
        {"performance_preset": "fast_4step", "resolved_backend": "fl2va_model"},
    )

    assert result[0] is accelerated
    assert "已启用" in result[1]
    assert calls == [(model, performance.TURBO_LORA_NAME)]


def test_ref2va_fast_loads_official_ref2va_lora(monkeypatch):
    model = object()
    accelerated = object()
    calls = []
    monkeypatch.setattr(performance, "_load_lightx2v_lora", lambda value, name, **kwargs: calls.append((value, name)) or accelerated)
    monkeypatch.setattr(performance, "_apply_sage_attention", lambda value, guide: value)
    monkeypatch.setattr(performance, "_apply_easy_cache", lambda value, guide: value)

    result = MiniMaxH3AccelerationRouter().apply(
        model,
        {"performance_preset": "fast_4step", "resolved_backend": "ref2va_model"},
    )

    assert result[0] is accelerated
    assert "REF2VA" in result[1]
    assert calls == [(model, performance.REF2VA_TURBO_LORA_NAME)]


def test_fast_preset_uses_official_ref2va_four_step_contract():
    values = preset_values("fast_4step", backend="ref2va_model")
    assert values["steps"] == 4
    assert values["use_sage"] is True
    assert values["use_cache"] is True
    assert values["use_turbo_sampler"] is False


def test_ref2va_plan_uses_official_lora_and_native_sampler():
    guide = {"performance_preset": "fast_4step", "resolved_backend": "ref2va_model"}
    plan = acceleration_plan(guide)
    assert plan["use_turbo_lora"] is True
    assert plan["use_turbo_sampler"] is False
    assert plan["lora_name"] == performance.REF2VA_TURBO_LORA_NAME
    assert sampler_route(guide) == "native"


def test_voice_reference_backend_gets_ref2va_acceleration():
    guide = {
        "mode": "FL2VA",
        "voice_mode": "h3_reference",
        "performance_preset": "fast_4step",
        "resolved_backend": "ref2va_model",
    }
    plan = acceleration_plan(guide)
    assert plan["lora_name"] == performance.REF2VA_TURBO_LORA_NAME
    assert plan["use_turbo_lora"] is True


def test_fast_fl2va_routes_to_h3_turbo_sampler():
    assert sampler_route({"performance_preset": "fast_4step", "resolved_backend": "fl2va_model"}) == "native"
    assert sampler_route({"performance_preset": "fast_4step", "resolved_backend": "ref2va_model"}) == "native"


@pytest.mark.parametrize("mode", ["T2VA", "I2VA", "FL2VA", "L2VA"])
def test_fast_fl2va_modes_share_the_same_checked_turbo_contract(mode):
    guide = {"mode": mode, "performance_preset": "fast_4step", "resolved_backend": "fl2va_model"}
    plan = acceleration_plan(guide)
    assert plan["use_turbo_lora"] is True
    assert plan["use_turbo_sampler"] is False
    assert sampler_route(guide) == "native"


def test_turbo_sampler_is_disabled_when_lora_falls_back():
    guide = {
        "mode": "I2VA",
        "performance_preset": "fast_4step",
        "resolved_backend": "fl2va_model",
        "turbo_lora_applied": False,
    }
    assert sampler_route(guide) == "native"


def test_sampler_router_falls_back_if_turbo_node_is_unavailable(monkeypatch):
    guide = {"mode": "FL2VA", "performance_preset": "fast_4step", "resolved_backend": "fl2va_model", "turbo_lora_applied": True}
    monkeypatch.setattr(performance, "_turbo_class", lambda _name: (_ for _ in ()).throw(RuntimeError("node unavailable")))
    import sys
    samplers = type("Samplers", (), {"sampler_object": staticmethod(lambda name: ("native", name))})()
    monkeypatch.setitem(sys.modules, "comfy", type("Comfy", (), {"samplers": samplers})())
    monkeypatch.setitem(sys.modules, "comfy.samplers", samplers)

    result = MiniMaxH3SamplerRouter().route("res_multistep", guide)
    assert result == (("native", "euler"),)
    assert guide["turbo_sampler_applied"] is False


def test_acceleration_router_falls_back_without_leaving_turbo_sampler_enabled(monkeypatch):
    model = object()

    def fail(_model):
        raise AttributeError("incompatible H3 model wrapper")

    monkeypatch.setattr(performance, "_load_lightx2v_lora", lambda value, _name: fail(value))
    guide = {"mode": "I2VA", "performance_preset": "fast_4step", "resolved_backend": "fl2va_model"}

    result = MiniMaxH3AccelerationRouter().apply(model, guide)
    assert result[0] is model
    assert guide["turbo_lora_applied"] is False
    assert sampler_route(guide) == "native"
    assert "回退" in result[1]


def test_sampler_router_exposes_valid_sampler_combo():
    sampler_spec = MiniMaxH3SamplerRouter.INPUT_TYPES()["required"]["sampler_name"]
    assert sampler_spec[0]
    assert sampler_spec[1]["default"] == "res_multistep"


def test_ref2va_fast_sampler_router_forces_official_euler(monkeypatch):
    guide = {"performance_preset": "fast_4step", "resolved_backend": "ref2va_model"}
    import sys
    samplers = type("Samplers", (), {"sampler_object": staticmethod(lambda name: name)})()
    monkeypatch.setitem(sys.modules, "comfy", type("Comfy", (), {"samplers": samplers})())
    monkeypatch.setitem(sys.modules, "comfy.samplers", samplers)
    assert MiniMaxH3SamplerRouter().route("res_multistep", guide) == ("euler",)


def test_fl2va_fast_sampler_router_also_forces_official_euler(monkeypatch):
    guide = {"performance_preset": "fast_4step", "resolved_backend": "fl2va_model"}
    import sys
    samplers = type("Samplers", (), {"sampler_object": staticmethod(lambda name: name)})()
    monkeypatch.setitem(sys.modules, "comfy", type("Comfy", (), {"samplers": samplers})())
    monkeypatch.setitem(sys.modules, "comfy.samplers", samplers)
    assert MiniMaxH3SamplerRouter().route("res_multistep", guide) == ("euler",)
