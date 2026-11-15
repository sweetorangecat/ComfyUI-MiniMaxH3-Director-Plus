"""Author selection contracts, exercised without loading H3 weights."""
import importlib
import json
from pathlib import Path
import sys
import types

import pytest

from nodes import performance, sampling_experiments as experiments


LORAS = (
    ("minimax_h3_dmad_full_dareties_v4_step600_comfy_fro.safetensors", .8),
    ("MysticXXX_MMH3-V1.safetensors", .35),
    ("H3_Combat_V2.safetensors", .65),
    ("Motion_Repair_V2.safetensors", .35),
    ("H3_Fight_Flow_V1.safetensors", .35),
    ("H3_speed_slider_1.1.safetensors", 3.0),
)
UPSCALER = "h3_upscaler_lms_v0.1.safetensors"


def recipe():
    assert (Path(__file__).resolve().parents[1] / "nodes/wenwu_recipe.py").is_file(), "Missing author policy"
    return importlib.import_module("nodes.wenwu_recipe")


def guide(**changes):
    value = dict(mode="REF2VA", voice_mode="none", performance_preset="quality_two_stage",
                 resolved_backend="ref2va_model", sampling_method="standard",
                 community_lora_profile="combat", fps=24, motion_smoothing="off",
                 total_vram_gb=24, free_vram_gb=22, sol_attn_available=True)
    value.update(changes)
    return value


@pytest.mark.parametrize("voice", ["none", "h3_reference", "audio_reuse"])
def test_author_choice_normalizes_standard_before_selflift_validation(voice):
    assert experiments.validate_sampling_options(guide(voice_mode=voice))["sampling_method"] == "selflift"


@pytest.mark.parametrize("change", [dict(mode="T2VA"), dict(voice_mode="fish_lock"),
    dict(performance_preset="low_vram_two_stage"), dict(fps=48), dict(motion_smoothing="rife_x2")])
def test_author_choice_rejects_incompatible_standard_requests(change):
    with pytest.raises(ValueError, match="质量优先|REF2VA|24FPS|RIFE"):
        experiments.validate_sampling_options(guide(**change))


def test_author_scheduler_normalizes_direct_api_guide_and_keeps_off_baseline():
    selected = guide()
    assert performance.scheduler_plan(selected) == dict(scheduler="simple", steps=12, split_step=6, refine_reference_tail=False)
    assert selected["sampling_method"] == "selflift"
    off = guide(sampling_method="selflift", community_lora_profile="off")
    assert performance.scheduler_plan(off) == dict(scheduler="beta", steps=10, split_step=6, refine_reference_tail=False)


def test_author_asset_resolution_is_strict_complete_and_preserves_registered_subdirs():
    policy = recipe()
    calls = []
    def resolve(category, name):
        calls.append((category, name))
        return "verified/" + name
    resolved = policy.resolve_author_assets(resolve)
    assert resolved["ready"]
    assert [(row["name"], row["strength"]) for row in resolved["loras"]] == [("verified/" + name, strength) for name, strength in LORAS]
    assert resolved["upscaler_model"] == "verified/" + UPSCALER
    assert calls == [("loras", name) for name, _ in LORAS] + [("latent_upscale_models", UPSCALER)]
    absent = policy.resolve_author_assets(lambda category, name: None)
    assert absent["missing"] == [name for name, _ in LORAS] + [UPSCALER]
    assert not absent["ready"]


def test_author_recipe_parameters_are_an_immutable_versioned_contract():
    from dataclasses import FrozenInstanceError
    policy = recipe()
    assert getattr(policy, "AUTHOR_RECIPE", None) is not None
    with pytest.raises(FrozenInstanceError):
        policy.AUTHOR_RECIPE.steps = 10
    assert policy.AUTHOR_RECIPE.steps == 12


def test_author_low_grid_matches_point_six_without_changing_public_baseline():
    baseline = experiments.plan_selflift_dimensions(1920, 1080, 5, 24, 22)
    author = experiments.plan_selflift_dimensions(1920, 1080, 5, 24, 22, community_lora_profile="combat")
    for key, low_key in (("second_stage_width", "first_stage_width"), ("second_stage_height", "first_stage_height")):
        assert author[low_key] == max(2, round(author[key] / 16 * .6 / 2) * 2) * 16
        assert baseline[low_key] == max(2, round(baseline[key] / 16 * .65 / 2) * 2) * 16
    with pytest.raises(ValueError, match="20GB|18GB"):
        experiments.plan_selflift_dimensions(1920, 1080, 5, 8, 7, community_lora_profile="combat")


def install_router_stubs(monkeypatch, *, fail=None):
    calls = []
    monkeypatch.setattr(performance, "resolve_registered_model_name", lambda category, name: "verified/" + name)
    def load(model, name, strength, **kwargs):
        calls.append((Path(name).name, strength))
        if Path(name).name == fail:
            raise performance.H3LoRAApplicationError("bad weight")
        return model
    monkeypatch.setattr(performance, "_load_lightx2v_lora", load)
    monkeypatch.setattr(performance, "_apply_sol_attention", lambda model, current: model)
    monkeypatch.setattr(performance, "_apply_minimax_reuse_attention", lambda *args: pytest.fail("Author route cannot use spatial/exact fallback"))
    class Shift:
        @staticmethod
        def execute(model, video, audio):
            model.sampling = types.SimpleNamespace(shift=video, audio_shift=audio)
            return (model,)
    monkeypatch.setitem(sys.modules, "comfy_extras.nodes_minimax_h3", types.SimpleNamespace(MiniMaxH3SigmaShift=Shift))
    model = types.SimpleNamespace(model_options={})
    model.get_model_object = lambda key: model.sampling
    model.sampling = types.SimpleNamespace(shift=10., audio_shift=5.)
    return model, calls


def test_actual_router_applies_exact_chain_in_order_despite_old_off_switches(monkeypatch):
    model, calls = install_router_stubs(monkeypatch)
    monkeypatch.setenv("MMH3_DETAIL_LORAS", "0")
    monkeypatch.setenv("MMH3_EXTRA_LORAS", "0")
    selected = guide()
    first, _, ready, second = performance.MiniMaxH3AccelerationRouter().apply(model, selected, community_loras="全部关闭")
    assert ready and first is second
    assert calls == list(LORAS)
    assert (first.sampling.shift, first.sampling.audio_shift) == (12., 3.)
    assert selected["sampling_recipe"]["steps"] == 12
    assert selected["author_adapters_applied"] == [{"name": "verified/" + name, "strength": strength} for name, strength in LORAS]
    assert performance.MiniMaxH3PerformancePreset().apply(selected, True)[0] == 12


@pytest.mark.parametrize("filename", [LORAS[0][0], LORAS[3][0], LORAS[-1][0]])
def test_actual_router_stops_on_any_failed_author_application(monkeypatch, filename):
    model, calls = install_router_stubs(monkeypatch, fail=filename)
    with pytest.raises(RuntimeError, match="SelfLift|作者|文武"):
        performance.MiniMaxH3AccelerationRouter().apply(model, guide())
    assert calls[-1][0] == filename


@pytest.mark.parametrize("name", [LORAS[0][0], LORAS[1][0], LORAS[3][0], "MysticXXX_MMH3-V5-ref2va.safetensors"])
@pytest.mark.parametrize("kind", ["DaSiWa_LTX2LoraLoader", "LoraLoaderModelOnly", "metadata"])
def test_upstream_manual_conflicts_fail_before_reconciliation_or_model_loading(monkeypatch, name, kind):
    model, calls = install_router_stubs(monkeypatch)
    selected = guide()
    prompt = {"router": {"class_type": "MiniMaxH3AccelerationRouter", "inputs": {"model": ["manual", 0]}},
              "manual": {"class_type": kind, "inputs": {"model": ["base", 0], "lora_name": "folder/" + name,
                  "strength_model": .5, "stack_data": json.dumps([{"on": True, "lora": "folder/" + name, "str": .5}])}}}
    if kind == "metadata":
        selected["additional_loras_requested"] = [{"name": name, "strength": .5}]
        prompt = None
    monkeypatch.setattr(performance, "mystic_v5_selection", lambda *args: pytest.fail("conflicts must be checked first"))
    with pytest.raises(ValueError, match="附加 LoRA|手动|冲突"):
        performance.MiniMaxH3AccelerationRouter().apply(model, selected, prompt=prompt, unique_id="router")
    assert calls == []


def test_author_router_preserves_unrelated_manual_adapter(monkeypatch):
    model, calls = install_router_stubs(monkeypatch)
    selected = guide(additional_loras_requested=[{"name": "unrelated_style.safetensors", "strength": .2}])
    performance.MiniMaxH3AccelerationRouter().apply(model, selected)
    assert selected["additional_loras_requested"] == [{"name": "unrelated_style.safetensors", "strength": .2}]
    assert calls == list(LORAS)


def test_direct_author_router_infers_ref_backend_from_validated_mode(monkeypatch):
    model, calls = install_router_stubs(monkeypatch)
    selected = guide()
    selected.pop("resolved_backend")
    assert performance.MiniMaxH3AccelerationRouter().apply(model, selected)[2]
    assert selected["resolved_backend"] == "ref2va_model"
    assert calls == list(LORAS)


def test_author_global_attention_unavailable_cannot_fall_back(monkeypatch):
    model, _ = install_router_stubs(monkeypatch)
    def unavailable():
        raise RuntimeError("Native BlockSparseAttention is unavailable")
    monkeypatch.setattr(performance, "native_sparse_node", unavailable)
    with pytest.raises(RuntimeError, match="Sol-Attn|global_stream"):
        performance.MiniMaxH3AccelerationRouter().apply(model, guide(sol_attn_available=False))


def test_author_run_accepts_thirteen_sigmas_and_reports_six_plus_six(monkeypatch):
    from test_sampling_experiments import AV, AvatarSamplerNode, euler_sampler
    from test_native_attention import native_model, native_runtime
    from nodes import selflift_lifter
    monkeypatch.setattr(selflift_lifter, 'preferred_model',
                        lambda function: pytest.fail('Author recipe must retain its exact LMS lifter'))
    node = AvatarSamplerNode()
    selected = guide(selflift_upscaler_model=UPSCALER, attention_backend="sol_attn", sol_attn_applied=True)
    latent = {"samples": AV(), "history": object()}
    model = native_model(native_runtime(monkeypatch))
    result = experiments.run_selflift(model, [], object(), euler_sampler(),
        [1, .96, .9, .85, .75, .65, .5, .4, .3, .2, .1, .04, 0], latent, 42,
        selected, registry={"SelfLiftAvatarH3Sampler": lambda: node})
    assert result["history"] is latent["history"]
    assert selected["sampling_executed_steps"] == 12
    assert (selected["two_stage_first_executed_steps"], selected["two_stage_second_executed_steps"]) == (6, 6)
    assert (selected["two_stage_first_sigma_count"], selected["two_stage_second_sigma_count"]) == (7, 7)
    actual = node.received
    assert (actual["cfg"], actual["transition_step"], actual["lowres_scale"], actual["rho"]) == (1, 6, .6, 0)
    assert actual["highres_tiling_mode"] == "global_stream"
    assert actual["upscaler_model"] == UPSCALER
    assert selected['selflift_lifter'] == 'upstream_learned'


def test_author_run_requires_global_stream_even_for_a_short_clip():
    from test_sampling_experiments import AV, SamplerNode, euler_sampler
    node = SamplerNode()
    with pytest.raises(RuntimeError, match="global_stream|Sol-Attn"):
        experiments.run_selflift(object(), [], object(), euler_sampler(),
            [1, .96, .9, .85, .75, .65, .5, .4, .3, .2, .1, .04, 0],
            {"samples": AV()}, 42, guide(selflift_upscaler_model=UPSCALER),
            registry={"SelfLiftH3Sampler": lambda: node})
    assert not hasattr(node, "received")


def test_author_run_rejects_wrong_actual_shift_before_sampling():
    from test_sampling_experiments import AV, AvatarSamplerNode, euler_sampler
    node = AvatarSamplerNode()
    model = types.SimpleNamespace(get_model_object=lambda name: types.SimpleNamespace(shift=10., audio_shift=5.))
    with pytest.raises(ValueError, match="shift.*12/3|12/3.*shift"):
        experiments.run_selflift(model, [], object(), euler_sampler(),
            [1, .96, .9, .85, .75, .65, .5, .4, .3, .2, .1, .04, 0],
            {"samples": AV()}, 42, guide(selflift_upscaler_model=UPSCALER,
                attention_backend="sol_attn", sol_attn_applied=True),
            registry={"SelfLiftAvatarH3Sampler": lambda: node})
    assert not hasattr(node, "received")


def test_author_run_rejects_a_known_non_simple_sigma_schedule():
    import torch
    from test_sampling_experiments import AV, AvatarSamplerNode, euler_sampler
    node = AvatarSamplerNode()
    model = types.SimpleNamespace(get_model_object=lambda name: types.SimpleNamespace(shift=12., audio_shift=3.))
    sigmas = torch.linspace(1, 0, 13)
    sigmas._h3_scheduler_name = "beta"
    with pytest.raises(ValueError, match="simple"):
        experiments.run_selflift(model, [], object(), euler_sampler(), sigmas, {"samples": AV()}, 42,
            guide(selflift_upscaler_model=UPSCALER, attention_backend="sol_attn", sol_attn_applied=True),
            registry={"SelfLiftAvatarH3Sampler": lambda: node})
    assert not hasattr(node, "received")


@pytest.mark.parametrize("upscaler", ["minimax_h3_latent_upscaler_3d_fp16.safetensors", "other.safetensors", "", None])
def test_direct_author_run_requires_the_exact_lms_lifter(upscaler):
    from test_sampling_experiments import AV, AvatarSamplerNode, euler_sampler
    node = AvatarSamplerNode()
    model = types.SimpleNamespace(get_model_object=lambda name: types.SimpleNamespace(shift=12., audio_shift=3.))
    with pytest.raises((ValueError, RuntimeError), match="h3_upscaler_lms_v0.1.safetensors"):
        experiments.run_selflift(model, [], object(), euler_sampler(),
            [1, .96, .9, .85, .75, .65, .5, .4, .3, .2, .1, .04, 0],
            {"samples": AV()}, 42, guide(selflift_upscaler_model=upscaler,
                attention_backend="sol_attn", sol_attn_applied=True),
            registry={"SelfLiftAvatarH3Sampler": lambda: node})
    assert not hasattr(node, "received")


@pytest.mark.parametrize("prefix", ["minimax/", "minimax\\"])
def test_direct_author_run_accepts_the_exact_lms_lifter_in_registered_subdirectories(monkeypatch, prefix):
    from test_sampling_experiments import AV, AvatarSamplerNode, euler_sampler
    from test_native_attention import native_model, native_runtime
    node = AvatarSamplerNode()
    model = native_model(native_runtime(monkeypatch))
    latent = {"samples": AV()}
    experiments.run_selflift(model, [], object(), euler_sampler(),
        [1, .96, .9, .85, .75, .65, .5, .4, .3, .2, .1, .04, 0], latent, 42,
        guide(selflift_upscaler_model=prefix + UPSCALER, attention_backend="sol_attn", sol_attn_applied=True),
        registry={"SelfLiftAvatarH3Sampler": lambda: node})
    assert node.received["upscaler_model"] == prefix + UPSCALER


def test_ordinary_selflift_retains_generic_lifter_compatibility():
    from test_sampling_experiments import AV, SamplerNode, euler_sampler
    node = SamplerNode()
    latent = {"samples": AV()}
    result = experiments.run_selflift(object(), [], object(), euler_sampler(),
        [1, .95, .9, .8, .7, .6, .45, .3, .18, .08, 0], latent, 42,
        guide(sampling_method="selflift", community_lora_profile="off", selflift_upscaler_model="generic.safetensors"),
        registry={"SelfLiftH3Sampler": lambda: node})
    assert result["samples"] is latent["samples"]
    assert node.received["upscaler_model"] == "generic.safetensors"


def test_author_report_keeps_versioned_recipe_and_actual_chain_through_inheritance(monkeypatch):
    from nodes.execution_report import execution_plan, new_report, sampling_settings
    model, _ = install_router_stubs(monkeypatch)
    selected = guide()
    performance.MiniMaxH3AccelerationRouter().apply(model, selected)
    plan = execution_plan(selected)
    assert plan["sampling_recipe"]["id"] == recipe().AUTHOR_RECIPE_ID
    assert plan["sampling_recipe"]["version"] == recipe().AUTHOR_RECIPE_VERSION
    assert plan["sampling_recipe"]["canonical_speed_identity"] == "public_canonical_not_verified_byte_identical_to_archive"
    assert plan["author_adapters_applied"] == selected["author_adapters_applied"]
    settings = sampling_settings(selected, model=model, sigmas=[1] * 12 + [0])
    assert settings["author_adapters_applied"] == selected["author_adapters_applied"]
    inherited = new_report(source={"_h3_execution_report": {"version": 1, "plan": plan, "sampling_settings": settings}})
    assert inherited["plan"]["author_adapters_applied"] == plan["author_adapters_applied"]
    assert inherited["sampling_settings"]["author_adapters_applied"] == settings["author_adapters_applied"]


def test_native_author_loader_cannot_select_plain_dmad_when_fro_disappears(monkeypatch):
    import nodes as comfy_nodes
    calls = []
    fro, plain = LORAS[0][0], "minimax_h3_dmad_full_dareties_v4_step600_comfy.safetensors"
    installed = {fro: "author/" + fro, plain: "old/" + plain}
    monkeypatch.setattr(performance, "resolve_registered_model_name", lambda category, name: installed.get(name))
    class Loader:
        def load_lora_model_only(self, model, name, strength):
            calls.append(name)
            return (types.SimpleNamespace(patches={"weight": [strength]}, model_options={}),)
    monkeypatch.setattr(comfy_nodes, "LoraLoaderModelOnly", Loader, raising=False)
    model = types.SimpleNamespace(patches={}, model_options={})
    performance._load_lightx2v_lora(model, fro, .8)
    assert calls == ["author/" + fro]
    installed.pop(fro)
    with pytest.raises(performance.H3LoRAApplicationError, match="fro"):
        performance._load_lightx2v_lora(model, fro, .8)
    assert calls == ["author/" + fro]
