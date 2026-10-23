import pytest

from nodes import performance


def test_u22_ref2va_attention_plan_prefers_sol_attn_with_audio_sink():
    guide = {
        "mode": "REF2VA",
        "voice_mode": "h3_reference",
        "resolved_backend": "ref2va_model",
        "performance_preset": "quality_sage",
        "total_vram_gb": 32.0,
        "free_vram_gb": 26.0,
        "sol_attn_available": True,
    }

    plan = performance.u22_attention_plan(guide)

    assert plan["backend"] == "sol_attn"
    assert plan["tau"] == pytest.approx(1.3)
    assert plan["start_percent"] == pytest.approx(0.2)
    assert plan["end_percent"] == pytest.approx(1.0)
    assert plan["min_tokens"] == 12288
    assert plan["sink_conditioning"] == "exact_kv_and_rows"
    assert plan["int8_qk"] is True
    assert plan["int8_pv"] is True


def test_u22_ref2va_attention_plan_uses_dense_conditioning_for_voice_modes():
    for voice_mode in ("none", "h3_reference", "audio_reuse"):
        plan = performance.u22_attention_plan({
            "mode": "REF2VA",
            "voice_mode": voice_mode,
            "resolved_backend": "ref2va_model",
            "performance_preset": "ref_quality_native",
            "total_vram_gb": 32.0,
            "free_vram_gb": 26.0,
            "sol_attn_available": True,
        })
        assert plan["sink_conditioning"] == "exact_kv_and_rows"


def test_u22_attention_plan_falls_back_without_high_vram_or_kernel():
    for guide in (
        {
            "resolved_backend": "ref2va_model",
            "performance_preset": "quality_sage",
            "total_vram_gb": 8.0,
            "free_vram_gb": 7.0,
            "sol_attn_available": True,
        },
        {
            "resolved_backend": "ref2va_model",
            "performance_preset": "quality_sage",
            "total_vram_gb": 32.0,
            "free_vram_gb": 26.0,
            "sol_attn_available": False,
        },
    ):
        plan = performance.u22_attention_plan(guide)
        assert plan["backend"] == "sage"


def test_u22_attention_plan_allows_quality_two_stage_ref2va():
    plan = performance.u22_attention_plan({
        "resolved_backend": "ref2va_model",
        "performance_preset": "quality_two_stage",
        "total_vram_gb": 32.0,
        "free_vram_gb": 26.0,
        "sol_attn_available": True,
    })
    assert plan["backend"] == "sol_attn"


def test_u22_attention_plan_never_changes_non_reference_routes():
    plan = performance.u22_attention_plan({
        "resolved_backend": "fl2va_model",
        "performance_preset": "quality_sage",
        "total_vram_gb": 32.0,
        "free_vram_gb": 26.0,
        "sol_attn_available": True,
    })
    assert plan["backend"] == "sage"


def test_u22_ref2va_route_applies_sol_attn_before_sage(monkeypatch):
    calls = []
    monkeypatch.setattr(performance, "_sol_attn_class", lambda: object)
    monkeypatch.setattr(
        performance,
        "_apply_sol_attention",
        lambda model, guide: calls.append((model, guide["resolved_backend"])) or "sol-model",
    )
    guide = {
        "resolved_backend": "ref2va_model",
        "performance_preset": "quality_sage",
        "total_vram_gb": 32.0,
        "free_vram_gb": 26.0,
        "sol_attn_available": True,
    }
    assert performance._apply_sage_attention("base-model", guide) == "sol-model"
    assert calls == [("base-model", "ref2va_model")]
    assert "sol_attn_error" not in guide


def test_sol_attn_report_distinguishes_optional_extra_tokens(monkeypatch):
    class CurrentSolAttn:
        @staticmethod
        def execute(model, tau, start_percent, end_percent, min_tokens,
                    int8_qk, sink_conditioning, morton, morton_curve,
                    dense_blocks, verbose, tau_profile=None, use_tma=False,
                    int8_pv=True):
            return (model,)

    monkeypatch.setattr(performance, "_sol_attn_class", lambda: CurrentSolAttn)
    monkeypatch.setattr(performance, "_apply_minimax_ffn_chunking", lambda model, guide: model)
    guide = {
        "resolved_backend": "ref2va_model",
        "performance_preset": "quality_sage",
        "total_vram_gb": 32.0,
        "free_vram_gb": 26.0,
        "sol_attn_available": True,
    }
    assert performance._apply_sol_attention("model", guide) == "model"
    assert guide["sol_attn_extra_tokens_requested"] == 256
    assert guide["sol_attn_extra_tokens_applied"] is False
    assert guide["sol_attn_extra_tokens"] is None


def test_u22_ref_two_stage_prefers_sol_attn_for_both_shared_passes(monkeypatch):
    calls = []
    monkeypatch.delenv("MMH3_TWO_STAGE_SAGE", raising=False)
    monkeypatch.delenv("MMH3_SECOND_STAGE_SAGE", raising=False)
    monkeypatch.setattr(performance, "resolve_registered_model_name", lambda *args, **kwargs: None)
    monkeypatch.setattr(performance, "_load_lightx2v_lora", lambda model, name, strength=1.0, low_vram=False: "lora")
    monkeypatch.setattr(performance, "_apply_ref_detail_loras", lambda model, guide: model)
    monkeypatch.setattr(
        performance,
        "_apply_sol_attention",
        lambda model, guide: calls.append(model) or "sol",
    )
    guide = {
        "mode": "REF2VA",
        "performance_preset": "quality_two_stage",
        "resolved_backend": "ref2va_model",
        "voice_mode": "h3_reference",
        "total_vram_gb": 32.0,
        "free_vram_gb": 26.0,
        "sol_attn_available": True,
        "community_lora_mode": "全部关闭",
    }
    result = performance.MiniMaxH3AccelerationRouter().apply("model", guide)
    assert result[0] == result[3] == "sol"
    assert calls == ["lora"]
    assert guide["sol_attn_applied"] is True
    assert guide["attention_backend"] == "sol_attn"


def test_u22_two_stage_sol_attn_failure_falls_back_to_sage(monkeypatch):
    calls = []

    class FakeSageNode:
        def execute(self, model):
            calls.append(model)
            return (f"{model}:sage",)

    monkeypatch.delenv("MMH3_TWO_STAGE_SAGE", raising=False)
    monkeypatch.delenv("MMH3_SECOND_STAGE_SAGE", raising=False)
    monkeypatch.setattr(performance, "resolve_registered_model_name", lambda *args, **kwargs: None)
    monkeypatch.setattr(performance, "_load_lightx2v_lora", lambda model, *args, **kwargs: model)
    monkeypatch.setattr(performance, "_apply_ref_detail_loras", lambda model, guide: model)
    monkeypatch.setattr(performance, "_kj_ltx_class", lambda name: FakeSageNode)
    monkeypatch.setattr(
        performance,
        "_apply_sol_attention",
        lambda model, guide: (_ for _ in ()).throw(RuntimeError("kernel unavailable")),
    )
    guide = {
        "mode": "REF2VA",
        "performance_preset": "quality_two_stage",
        "resolved_backend": "ref2va_model",
        "voice_mode": "h3_reference",
        "total_vram_gb": 32.0,
        "free_vram_gb": 26.0,
        "sol_attn_available": True,
        "community_lora_mode": "全部关闭",
    }
    result = performance.MiniMaxH3AccelerationRouter().apply("model", guide)
    assert result[2] is True
    assert result[0] == result[3] == "model:sage"
    assert calls == ["model"]
    assert guide["sol_attn_applied"] is False
    assert "kernel unavailable" in guide["sol_attn_error"]
    assert guide["sage_applied"] is True
