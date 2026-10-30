import copy
import json

import pytest

from nodes import performance
from nodes.two_stage_assets import SELFLIFT_MYSTIC_LORA, V4_TURBO_LORA


REF_V5 = "minimax/MysticXXX_MMH3-V5-ref2va.safetensors"
FL_V5 = "minimax/MysticXXX_MMH3-V5.safetensors"


def lora_prompt(rows, *, model_type="Basic"):
    return {
        "base": {"class_type": "MiniMaxH3ModelRouter", "inputs": {}},
        "extra": {"class_type": "DaSiWa_LTX2LoraLoader", "inputs": {
            "model": ["base", 0], "stack_data": json.dumps(rows), "model_type": model_type,
        }},
        "router": {"class_type": "MiniMaxH3AccelerationRouter", "inputs": {"model": ["extra", 0]}},
    }


def mock_route(monkeypatch):
    calls = []
    monkeypatch.setenv("MMH3_TWO_STAGE_SAGE", "0")
    monkeypatch.setenv("MMH3_DETAIL_LORAS", "1")
    monkeypatch.setattr(performance, "resolve_registered_model_name", lambda category, name, **kw: name)
    monkeypatch.setattr(performance, "resolve_selflift_lora_recipe", lambda: {
        "kind": "v4_mystic_fallback", "lora_name": V4_TURBO_LORA, "lora_strength": 1.0,
        "detail_name": SELFLIFT_MYSTIC_LORA, "detail_strength": 0.45,
    })
    monkeypatch.setattr(performance, "_resolve_first_registered", lambda category, names:
                        next((name for name in names if name.endswith(SELFLIFT_MYSTIC_LORA)), None))

    def load(model, name, strength=1.0, **kw):
        calls.append(name)
        return (*model, name)

    monkeypatch.setattr(performance, "_load_lightx2v_lora", load)
    monkeypatch.setattr(performance, "_apply_minimax_reuse_attention", lambda model, guide: model)
    return calls


@pytest.mark.parametrize("backend,method,filename", [
    ("ref2va_model", "selflift", REF_V5),
    ("ref2va_model", "standard", REF_V5),
    ("fl2va_model", "standard", FL_V5),
])
@pytest.mark.parametrize("voice_mode", ["none", "h3_reference"])
def test_upstream_v5_reaches_both_models_and_suppresses_only_old_mystic(
    monkeypatch, backend, method, filename, voice_mode
):
    calls = mock_route(monkeypatch)
    guide = dict(mode="REF2VA" if backend == "ref2va_model" else "FL2VA",
                 voice_mode=voice_mode, performance_preset="quality_two_stage",
                 resolved_backend=backend, sampling_method=method, community_lora_profile="off")
    prompt = lora_prompt([{"on": True, "lora": filename, "str": 0.45, "vs": 1}])
    before = copy.deepcopy(prompt)
    first, _, ready, second = performance.MiniMaxH3AccelerationRouter().apply(
        (filename,), guide, community_loras="仅 U22 细节链", prompt=prompt, unique_id="router"
    )
    assert ready
    assert first[0] == second[0] == filename
    assert not any(name.endswith(SELFLIFT_MYSTIC_LORA) for name in calls)
    assert guide["additional_loras_requested"] == [{"name": filename, "strength": 0.45}]
    assert guide["detail_lora_override"] == filename
    assert prompt == before


@pytest.mark.parametrize("row", [
    {"on": False, "lora": REF_V5, "str": 0.45},
    {"on": True, "lora": REF_V5, "str": 0},
    {"on": True, "lora": REF_V5, "str": 0.45, "vs": 0},
    {"on": True, "lora": "None", "str": 1},
])
def test_disabled_additional_lora_restores_core_mystic(monkeypatch, row):
    calls = mock_route(monkeypatch)
    guide = dict(mode="REF2VA", voice_mode="none", performance_preset="quality_two_stage",
                 resolved_backend="ref2va_model", sampling_method="selflift", community_lora_profile="off",
                 detail_lora_override=REF_V5)
    performance.MiniMaxH3AccelerationRouter().apply(
        (), guide, prompt=lora_prompt([row]), unique_id="router"
    )
    assert SELFLIFT_MYSTIC_LORA in calls
    assert guide["additional_loras_requested"] == []
    assert "detail_lora_override" not in guide


def test_disconnected_v5_cannot_suppress_the_active_recipe(monkeypatch):
    calls = mock_route(monkeypatch)
    prompt = lora_prompt([{"on": True, "lora": REF_V5, "str": 0.45}])
    prompt["router"]["inputs"]["model"] = ["base", 0]
    guide = dict(mode="REF2VA", voice_mode="none", performance_preset="quality_two_stage",
                 resolved_backend="ref2va_model", sampling_method="selflift", community_lora_profile="off")
    performance.MiniMaxH3AccelerationRouter().apply((), guide, prompt=prompt, unique_id="router")
    assert SELFLIFT_MYSTIC_LORA in calls
    assert "detail_lora_override" not in guide


@pytest.mark.parametrize("backend,rows,mode,error", [
    ("ref2va_model", [{"on": True, "lora": FL_V5, "str": 0.45}], "Basic", "REF2VA"),
    ("fl2va_model", [{"on": True, "lora": REF_V5, "str": 0.45}], "Basic", "FL2VA"),
    ("ref2va_model", [{"on": True, "lora": REF_V5, "str": 0.45}], "LTX-2.3", "Basic"),
    ("ref2va_model", [{"on": True, "lora": REF_V5, "str": 0.45},
                      {"on": True, "lora": SELFLIFT_MYSTIC_LORA, "str": 0.45}], "Basic", "叠加"),
])
def test_invalid_mystic_combinations_stop_before_acceleration(monkeypatch, backend, rows, mode, error):
    monkeypatch.setattr(performance, "_apply_acceleration", lambda *args: pytest.fail("invalid LoRA must stop first"))
    with pytest.raises(ValueError, match=error):
        performance.MiniMaxH3AccelerationRouter().apply(
            object(), {"resolved_backend": backend}, prompt=lora_prompt(rows, model_type=mode), unique_id="router"
        )


def test_missing_v5_stops_instead_of_suppressing_v1(monkeypatch):
    monkeypatch.setattr(performance, "resolve_registered_model_name", lambda *args, **kw: None)
    with pytest.raises(ValueError, match="未安装|缺少"):
        performance.MiniMaxH3AccelerationRouter().apply(
            object(), {"resolved_backend": "ref2va_model"},
            prompt=lora_prompt([{"on": True, "lora": REF_V5, "str": 0.45}]), unique_id="router"
        )
