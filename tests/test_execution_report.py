import json
import sys
import types
from types import SimpleNamespace

import pytest
import torch

from nodes import execution_report as evidence


def test_plan_reports_reference_endpoint_semantics_without_private_payloads():
    guide = dict(mode="FL2VA", resolved_backend="ref2va_model", voice_mode="h3_reference",
                 performance_preset="low_vram", first_frame=torch.zeros(1), last_frame=torch.ones(1),
                 requested_ref_image_size="max", ref_image_size="match",
                 prompt="PRIVATE PROMPT", ref_images={"ref_image_1": torch.zeros(1)},
                 ref_audios={"ref_audio_1": {"waveform": torch.zeros(1)}},
                 sampling_recipe={"steps": 8, "sampler_name": "euler", "scheduler": "simple",
                                  "video_shift": 6.0, "audio_shift": 3.0, "private": "SECRET"})
    plan = evidence.execution_plan(guide)
    assert plan["route"] == "ref2va_model"
    assert plan["endpoints"] == {"first": "reference_only", "last": "reference_only"}
    assert plan["references"]["images"] == 1
    assert plan["references"]["audios"] == 1
    assert plan["requested_ref_image_size"] == "max"
    assert plan["ref_image_size"] == "match"
    assert plan["sampling_recipe"]["steps"] == 8
    serialized = json.dumps(plan, allow_nan=False)
    assert "PRIVATE" not in serialized and "SECRET" not in serialized and "waveform" not in serialized


def test_plan_records_safe_reference_preprocessing_notes():
    plan = evidence.execution_plan({
        "reference_preprocessing": ["五视图已取左侧主身份 panel"],
        "mlp_chunking_applied": True,
        "minimax_mlp_chunks": 2,
    })

    assert plan["reference_preprocessing"] == ["五视图已取左侧主身份 panel"]
    assert plan["mlp_chunking_applied"] is True
    assert plan["minimax_mlp_chunks"] == 2


def test_plan_records_selflift_natural_texture_policy():
    plan = evidence.execution_plan({
        "natural_texture_policy": "selflift_u22_baseline",
        "natural_texture_source": "upstream_mystic_v5_replaced",
    })

    assert plan["natural_texture_policy"] == "selflift_u22_baseline"
    assert plan["natural_texture_source"] == "upstream_mystic_v5_replaced"


def test_plan_records_sol_attn_execution_evidence():
    plan = evidence.execution_plan({
        "sol_attn_applied": True,
        "sol_attn_tau": 1.3,
        "sol_attn_min_tokens": 12288,
        "sol_attn_extra_tokens_requested": 256,
        "sol_attn_extra_tokens_applied": False,
        "sol_attn_extra_tokens": None,
        "sol_attn_sink_conditioning": "exact_kv_and_rows",
        "sol_attn_int8_qk": True,
        "sol_attn_int8_pv": True,
    })
    assert plan["sol_attn_applied"] is True
    assert plan["sol_attn_tau"] == 1.3
    assert plan["sol_attn_min_tokens"] == 12288
    assert plan["sol_attn_extra_tokens_requested"] == 256
    assert plan["sol_attn_extra_tokens_applied"] is False
    assert plan["sol_attn_extra_tokens"] is None
    assert plan["sol_attn_sink_conditioning"] == "exact_kv_and_rows"
    assert plan["sol_attn_int8_qk"] is True
    assert plan["sol_attn_int8_pv"] is True


def test_fl_endpoints_are_hard_constraints():
    plan = evidence.execution_plan(dict(resolved_backend="fl2va_model", first_frame=object()))
    assert plan["endpoints"] == {"first": "hard_anchor", "last": "absent"}


def test_additional_lora_requests_survive_report_handoff_without_claiming_application():
    rows = [{"name": "minimax/MysticXXX_MMH3-V5-ref2va.safetensors", "strength": 0.45,
             "private": "do not serialize", "weights": torch.zeros(1)}]
    guide = {"additional_loras_requested": rows, "detail_lora_override": rows[0]["name"]}
    report = evidence.new_report(guide)
    source = {evidence.REPORT_KEY: report}
    inherited = evidence.new_report({}, source)
    expected = [{"name": rows[0]["name"], "strength": 0.45}]
    assert report["plan"]["additional_loras_requested"] == expected
    assert inherited["plan"]["additional_loras_requested"] == expected
    assert inherited["plan"]["detail_lora_override"] == rows[0]["name"]
    rows[0]["strength"] = 0.9
    assert report["plan"]["additional_loras_requested"] == expected
    serialized = json.dumps(inherited, allow_nan=False)
    assert "additional_loras_applied" not in serialized and "do not serialize" not in serialized


def test_attention_evidence_distinguishes_requested_and_applied_after_fallback():
    guide = dict(sage_requested=True, sage_applied=False,
                 attention_backend='native_fallback', sage_error='Sage unavailable',
                 sampling_recipe={'use_sage': True})
    inherited = evidence.new_report(guide)
    assert inherited['plan']['sage_requested']
    assert inherited['plan']['sage_applied'] is False
    current = evidence.new_report(dict(guide, sage_applied=True,
                                      attention_backend='h3_sage', minimax_head_chunks=2),
                                  {evidence.REPORT_KEY: inherited})
    assert current['current_plan']['attention_backend'] == 'h3_sage'
    assert current['current_plan']['minimax_head_chunks'] == 2


def test_reference_route_records_requested_endpoints_after_director_moves_images():
    plan = evidence.execution_plan(dict(resolved_backend="ref2va_model", first_frame=None,
                                       requested_first_frame=True, requested_last_frame=True))
    assert plan["endpoints"] == {"first": "reference_only", "last": "reference_only"}


def test_failed_stage_is_not_success_and_error_text_is_not_exposed(monkeypatch):
    ticks = iter([10.0, 12.5])
    monkeypatch.setattr(evidence.time, "perf_counter", lambda: next(ticks))
    report = evidence.new_report({})
    with pytest.raises(ValueError, match="private"):
        with evidence.stage_timer(report, "decode", path="safe_vae"):
            raise ValueError("private content")
    stage = report["stages"][0]
    assert stage["status"] == "failed"
    assert stage["seconds"] == 2.5
    assert stage["error_type"] == "ValueError"
    assert "private" not in json.dumps(report)


def test_reports_follow_artifacts_without_mutating_prior_run_or_guide():
    guide = {"sampling_elapsed_seconds": 99.0}
    first = evidence.new_report(guide)
    with evidence.stage_timer(first, "conditioning"):
        pass
    latent = {"samples": torch.zeros(1)}
    evidence.attach_report(latent, first)
    second = evidence.new_report(guide, latent)
    with evidence.stage_timer(second, "sampling", steps=8):
        pass
    assert len(first["stages"]) == 1
    assert len(second["stages"]) == 2
    assert second["stages"][0]["origin"] == "upstream_artifact"
    assert second["stages"][1]["origin"] == "current_execution"
    assert "stages" not in guide and guide["sampling_elapsed_seconds"] == 99.0
    assert evidence.new_report(guide)["stages"] == []


def test_report_survives_native_av_separation_and_decode():
    from nodes.vae_decode import MiniMaxH3SafeVAEDecode

    class Nested:
        is_nested = True
        def __init__(self):
            self.members = (torch.zeros(1, 4, 2, 1, 1), torch.zeros(1))
        def unbind(self):
            return self.members

    class Vae:
        output_device = torch.device("cpu")
        def vae_output_dtype(self):
            return torch.float32
        def decode(self, latent):
            return torch.zeros(2, 4, 4, 3)

    report = evidence.new_report({"mode": "T2VA"})
    with evidence.stage_timer(report, "sampling"):
        pass
    latent = {"samples": Nested()}
    evidence.attach_report(latent, report)
    # Match native SeparateAVLatent: replace samples with the unbound video.
    separated = {"samples": latent["samples"].unbind()[0]}
    images = MiniMaxH3SafeVAEDecode().decode(Vae(), separated)[0]
    result = evidence.report_from(images)
    assert [stage["stage"] for stage in result["stages"]] == ["sampling", "decode"]
    assert result["stages"][-1]["status"] == "completed"


def test_conditioning_timing_attaches_to_native_latent(monkeypatch):
    from nodes import guide as module
    monkeypatch.setattr(module, "native_node", lambda _: SimpleNamespace(
        execute=lambda **kwargs: ([], {"samples": torch.zeros(1)})))
    state = dict(resolved_backend="fl2va_model", prompt="private", width=16, height=16, length=2)
    _, latent = module.MiniMaxH3DirectorPlusGuide().apply(None, None, None, state)
    assert evidence.report_from(latent)["stages"][0]["stage"] == "conditioning"
    assert "execution_report" not in state


def test_fish_conditioning_report_counts_generated_audio_actually_passed(monkeypatch):
    from nodes import guide as module
    monkeypatch.setattr(module, "native_node", lambda _: SimpleNamespace(
        execute=lambda **kwargs: ([], {"samples": torch.zeros(1)})))
    state = dict(resolved_backend="ref2va_model", voice_mode="fish_lock", prompt="private",
                 width=16, height=16, length=2, ref_audios={})
    generated = {"waveform": torch.zeros(1, 2, 8), "sample_rate": 24000}
    _, latent = module.MiniMaxH3DirectorPlusGuide().apply(None, None, None, state, generated)
    assert evidence.report_from(latent)["plan"]["references"]["audios"] == 1
    assert state["ref_audios"] == {}


def test_report_serialization_is_bounded_and_drops_nonfinite_and_unknown_values():
    plan = evidence.execution_plan(dict(mode="X" * 2000, seed=float("nan"),
                                       sampling_recipe={"steps": torch.zeros(1)}, tensor=torch.zeros(1)))
    assert len(plan["mode"]) <= 256
    assert "seed" not in plan
    json.dumps(plan, allow_nan=False)


def test_single_pass_timing_follows_output_and_cached_conditioning(monkeypatch):
    from nodes.two_stage import MiniMaxH3TwoStageSampler
    native = types.ModuleType("comfy_extras.nodes_custom_sampler")
    native.SamplerCustomAdvanced = SimpleNamespace(execute=lambda *args: (
        {"samples": torch.ones(1)}, {"samples": torch.ones(1)}))
    monkeypatch.setitem(sys.modules, "comfy_extras.nodes_custom_sampler", native)
    initial = {"samples": torch.zeros(1)}
    report = evidence.new_report({})
    with evidence.stage_timer(report, "conditioning"):
        pass
    evidence.attach_report(initial, report)
    outputs = MiniMaxH3TwoStageSampler().execute(None, SimpleNamespace(), None, torch.tensor([1., 0.]), initial)
    for output in outputs:
        stages = evidence.report_from(output)["stages"]
        assert [stage["stage"] for stage in stages] == ["conditioning", "sampling"]
        assert stages[0]["origin"] == "upstream_artifact"
        assert stages[1]["steps"] == 1
        assert evidence.report_from(output)["sampling_settings"]["steps"] == 1
        assert evidence.report_from(output)["sampling_settings"]["evidence"] == "sampler_inputs"
    assert len(evidence.report_from(initial)["stages"]) == 1


def test_reconstruction_metadata_only_exports_config_and_measurement_fields():
    source = {"profile": "manual_baseline", "status": "completed", "elapsed_seconds": 2.5,
              "input_shape": [2, 16, 16, 3], "prompt": "PRIVATE",
              "memory_before": {"free_vram_gb": 12.0, "waveform": torch.zeros(1)},
              "requested_config": {"settings": {"frame_batch": 4, "prompt": "PRIVATE"}},
              "error": "PRIVATE", "upstream_settings": {"tile_size": 512}}
    result = evidence.reconstruction_diagnostics(source)
    assert result["requested_config"]["settings"]["frame_batch"] == 4
    assert result["upstream_settings"]["tile_size"] == 512
    assert "PRIVATE" not in json.dumps(result)


def test_unrecognized_metadata_on_an_upstream_artifact_is_not_exported():
    source = {evidence.REPORT_KEY: {"version": 1,
              "plan": {"mode": "T2VA", "prompt": "PRIVATE"},
              "stages": [{"stage": "conditioning", "status": "completed", "seconds": 1.,
                          "waveform": torch.zeros(1), "error": "PRIVATE"}]}}
    result = evidence.new_report({}, source)
    assert result["plan"]["mode"] == "T2VA"
    assert "PRIVATE" not in json.dumps(result, allow_nan=False)


def test_standalone_reconstruction_evidence_survives_downstream_output():
    source = torch.zeros(1)
    report = evidence.new_report({})
    report["reconstruction"] = {"profile": "manual_baseline", "elapsed_seconds": 2.,
                                "requested_config": {"settings": {"frame_batch": 4}}}
    evidence.attach_report(source, report)
    downstream = evidence.new_report({}, source)
    assert downstream["reconstruction"] == report["reconstruction"]


def test_execution_plan_normalizes_actual_legacy_scheduler_and_lora_strengths():
    guide = {"scheduler": "stale", "scheduler_name": "beta", "sampler_name": "euler",
             "scheduler_model_sampling": {"video_shift": 10., "audio_shift": 5., "private": "SECRET"},
             "first_lora_name": "first.safetensors", "first_lora_strength": 0.75,
             "second_lora_name": "second.safetensors", "second_lora_strength": 0.7}
    plan = evidence.execution_plan(guide)
    assert plan["scheduler"] == "beta"
    assert plan["sampler_name"] == "euler"
    assert plan["video_shift"] == 10. and plan["audio_shift"] == 5.
    assert plan["first_lora_strength"] == 0.75 and plan["second_lora_strength"] == 0.7
    assert plan["scheduler_model_sampling"] == {"video_shift": 10., "audio_shift": 5.}
    assert "SECRET" not in json.dumps(plan)
    output = {"samples": torch.zeros(1)}
    evidence.attach_report(output, evidence.new_report(guide))
    assert evidence.new_report(source=output)["plan"] == plan


def test_sampling_settings_use_actual_objects_without_claiming_recipe_execution():
    guide = {"sampler_name": "stale", "scheduler_name": "stale", "video_shift": 1.,
             "scheduler_model_sampling": {"video_shift": 2., "audio_shift": 3.},
             "sampling_recipe": {"steps": 8, "sampler": "euler", "video_shift": 6., "audio_shift": 3.}}
    model = SimpleNamespace(get_model_object=lambda _: SimpleNamespace(shift=10., audio_shift=5.))
    sampler = SimpleNamespace(sampler_function=lambda: None, _h3_sampler_name="res_multistep")
    sigmas = torch.linspace(1., 0., 21)
    sigmas._h3_scheduler_name = "simple"
    sigmas._mmh3_sampling_signature = {"video_shift": 10., "audio_shift": 7.}
    settings = evidence.sampling_settings(guide, model=model, sampler=sampler, sigmas=sigmas)
    assert settings["evidence"] == "sampler_inputs"
    assert settings["sampler"] == "res_multistep" and settings["scheduler"] == "simple"
    assert settings["steps"] == 20
    assert settings["video_shift"] == 10. and settings["audio_shift"] == 5.
    assert settings["scheduler_model_sampling"]["audio_shift"] == 7.
    plan_only = evidence.new_report(guide)
    assert "sampling_settings" not in plan_only and plan_only["stages"] == []


def test_completed_sampling_settings_survive_decode_and_output_snapshots():
    report = evidence.new_report({})
    report["sampling_settings"] = {"evidence": "sampler_inputs", "origin": "current_execution",
                                   "sampler": "euler", "steps": 20, "audio_shift": 5.,
                                   "selflift_rho": 0.6, "selflift_w_min": 0.5,
                                   "selflift_w_max": 1., "selflift_upscaler_model": "h3.safetensors",
                                   "scheduler_model_sampling": {"video_shift": 10., "audio_shift": 5.}}
    source = {"samples": torch.zeros(1)}
    evidence.attach_report(source, report)
    downstream = evidence.new_report({}, source)
    assert downstream["sampling_settings"]["steps"] == 20
    assert downstream["sampling_settings"]["origin"] == "upstream_artifact"
    assert downstream["sampling_settings"]["scheduler_model_sampling"]["video_shift"] == 10.
    for key in ("selflift_rho", "selflift_w_min", "selflift_w_max", "selflift_upscaler_model"):
        assert downstream["sampling_settings"][key] == report["sampling_settings"][key]
