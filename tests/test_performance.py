import pytest

from nodes import performance
from nodes.performance import (
    MiniMaxH3AccelerationRouter,
    MiniMaxH3PerformancePreset,
    MiniMaxH3SamplerRouter,
    acceleration_plan,
    preset_values,
    sampler_route,
)


def test_fast_preset_exposes_four_step_sampling_contract():
    values = preset_values("fast_4step")
    assert values["steps"] == 4
    assert values["use_cache"] is True
    assert values["use_sage"] is True


def test_quality_preset_keeps_conservative_sampling():
    values = preset_values("quality")
    assert values["steps"] >= 10
    assert values["use_cache"] is False


def test_performance_node_reads_guide_preset():
    result = MiniMaxH3PerformancePreset().apply({"performance_preset": "reference_fast"})
    assert result[0] == 6
    assert result[1] is True


def test_fl2va_fast_loads_existing_official_lora(monkeypatch):
    model = object()
    accelerated = object()
    calls = []
    monkeypatch.setattr(performance, "_load_lightx2v_lora", lambda value, name: calls.append((value, name)) or accelerated)

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
    monkeypatch.setattr(performance, "_load_lightx2v_lora", lambda value, name: calls.append((value, name)) or accelerated)

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
