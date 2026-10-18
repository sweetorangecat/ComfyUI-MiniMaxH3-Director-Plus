"""CPU contract checks for the director, model route and SelfLift dispatch."""

import json
from pathlib import Path
import sys
import types

import pytest

from nodes import director
from nodes.sampling_experiments import plan_selflift_dimensions
from nodes.schema import RequestError


def short_guide(**changes):
    guide = dict(mode="REF2VA", voice_mode="none", duration=5,
                 performance_preset="quality_two_stage", resolved_backend="ref2va_model",
                 sampling_method="selflift", community_lora_profile="off")
    guide.update(changes)
    return guide


@pytest.fixture
def director_runtime(monkeypatch):
    monkeypatch.setattr(director, "_probe_vram_after_prefree", lambda: (24, 22))
    monkeypatch.setattr(director, "_cuda_memory_gb", lambda: (24, 22))
    monkeypatch.setattr(director, "_selflift_dependency_report", lambda: {
        "ready": True, "missing": [], "required_assets": ["v4.safetensors", "upscaler.safetensors"],
        "upscaler_model": "upscaler.safetensors"}, raising=False)
    monkeypatch.setattr(director, "_trained_two_stage_dependency_report", lambda route: {
        "ready": True, "missing": [], "required_assets": []})
    monkeypatch.setattr(director, "resolve_split_upscale_callables", lambda: (lambda: None,) * 3)
    monkeypatch.setattr(director, "resolve_upscale_model_name", lambda *args: "RealESRGAN_x2plus.pth")
    monkeypatch.setattr(director, "_seedvr2_dependency_report", lambda: {
        "ready": True, "missing": [], "available_dit": ["seedvr2_ema_3b_fp8_e4m3fn.safetensors"]})


def build_director(**changes):
    args = dict(mode="REF2VA", prompt="角色转身，动作连贯", duration=5, width=1920,
                height=1080, voice_mode="none", ref_image_size="match",
                performance_preset="quality_two_stage", timeline_data="{}", target_dialogue="",
                reference_transcript="", reference_image_1=object(),
                resolution_preset="1080p FHD", postprocess_mode="ai_upscale")
    args.update(changes)
    return director.MiniMaxH3DirectorPlus().build(**args)[0]


def test_director_appends_experiment_widgets_preserving_previous_order(monkeypatch):
    monkeypatch.setattr(director, "_uploaded_files", lambda *args: [""])
    optional = director.MiniMaxH3DirectorPlus.INPUT_TYPES()["optional"]
    assert list(optional)[-3:] == ["sampling_method", "community_lora_profile", "vosr2_profile"]
    assert optional["vosr2_profile"][1]["default"] == "legacy"
    assert optional["sampling_method"][1]["default"] == "standard"
    assert optional["community_lora_profile"][1]["default"] == "off"


@pytest.mark.parametrize("aspect", ["16:9", "9:16"])
def test_director_selflift_conditions_target_canvas(director_runtime, aspect):
    guide = build_director(sampling_method="selflift", aspect_ratio=aspect)
    plan = plan_selflift_dimensions(guide["requested_width"], guide["requested_height"], 5, 24, 22)
    assert (guide["width"], guide["height"]) == (plan["second_stage_width"], plan["second_stage_height"])
    assert (guide["first_stage_width"], guide["first_stage_height"]) == (plan["first_stage_width"], plan["first_stage_height"])
    assert guide["sampling_method"] == "selflift"
    assert not guide["two_stage_tiled"] and not guide["two_stage_tiling_required"]
    assert guide["selflift_upscaler_model"] == "upscaler.safetensors"
    assert guide["community_lora_profile"] == "off"


def test_director_selflift_missing_dependency_stops_before_sampling(director_runtime, monkeypatch):
    monkeypatch.setattr(director, "_selflift_dependency_report", lambda: {"ready": False, "missing": ["SelfLiftH3Sampler"]})
    with pytest.raises(RequestError, match="SelfLiftH3Sampler"):
        build_director(sampling_method="selflift")


@pytest.mark.parametrize("preset,size", [("2K QHD", (2560, 1440)), ("4K UHD", (3840, 2160))])
@pytest.mark.parametrize("method,path", [("ai_upscale", "ai_upscale"),
                                       ("video_sr", "video_sr"), ("vosr2", "video_sr")])
def test_director_selflift_long_uhd_preserves_selected_sr(director_runtime, monkeypatch, preset, size, method, path):
    from nodes import two_stage_assets
    monkeypatch.setattr(two_stage_assets, "_comfy_node_mappings", lambda: {
        name: object() for name in ("TESpeedVOSR2Loader", "TESpeedVOSR2Settings", "TESpeedVOSR2Video")})
    guide = build_director(sampling_method="selflift", duration=15, resolution_preset=preset,
                           postprocess_mode=method)
    assert guide["length"] == director.align_frame_count(15 * 24)
    assert (guide["requested_width"], guide["requested_height"]) == size
    assert (guide["target_width"], guide["target_height"]) == size
    assert guide["postprocess_mode"] == method
    assert guide["postprocess_path"] == path
    assert guide["width"] * guide["height"] <= 1344 * 768
    assert guide["sampling_method"] == "selflift"


@pytest.mark.parametrize("method", ["ai_upscale", "video_sr"])
def test_explicit_sr_wins_over_quality_two_stage_downscale(director_runtime, method):
    guide = build_director(postprocess_mode=method)
    assert guide["postprocess_path"] == method
    assert guide["postprocess_mode"] == method


def test_selected_seedvr2_missing_is_an_error(director_runtime, monkeypatch):
    monkeypatch.setattr(director, "_seedvr2_dependency_report", lambda: {"ready": False, "missing": ["SeedVR2"]})
    with pytest.raises(RequestError, match="SeedVR2"):
        build_director(performance_preset="quality", postprocess_mode="video_sr")


@pytest.mark.parametrize("preset", ["1080p FHD", "2K QHD"])
@pytest.mark.parametrize("method", ["ai_upscale", "video_sr", "vosr2"])
def test_smart_planner_preserves_explicit_sr(preset, method):
    from nodes.smart_1080p import resolve_smart_1080p_plan
    width, height = (1920, 1080) if preset == "1080p FHD" else (2560, 1440)
    plan = resolve_smart_1080p_plan("ref2va_model", 5, 32, 29, two_stage_ready=True,
        seedvr2_ready=True, target_width=width, target_height=height,
        target_preset=preset, postprocess_mode=method)
    assert plan["postprocess_mode"] == method


def test_smart_seedvr2_missing_fails_even_when_qhd_redraw_reaches_target():
    from nodes.smart_1080p import resolve_smart_1080p_plan
    with pytest.raises(RequestError, match="SeedVR2"):
        resolve_smart_1080p_plan("ref2va_model", 5, 32, 29, two_stage_ready=True,
            seedvr2_ready=False, target_width=2560, target_height=1440,
            target_preset="2K QHD", postprocess_mode="video_sr")


@pytest.mark.parametrize("native_size", [(896, 512), (1920, 1080), (2048, 1152)])
@pytest.mark.parametrize("method,path,engine", [
    ("ai_upscale", "ai_upscale", "comfy_upscale_model"),
    ("video_sr", "video_sr", "seedvr2"), ("vosr2", "video_sr", "vosr2"),
])
def test_failed_sampling_setup_preserves_explicit_sr_geometry(native_size, method, path, engine):
    from nodes import performance
    width, height = native_size
    guide = dict(native_width=width, native_height=height, requested_width=1920, requested_height=1080,
                 postprocess_mode=method, target_width=1920, target_height=1080)
    performance._reconcile_two_stage_fallback_geometry(guide)
    assert (guide["target_width"], guide["target_height"]) == (1920, 1080)
    assert (guide["postprocess_source_width"], guide["postprocess_source_height"]) == native_size
    assert guide["postprocess_mode"] == method and guide["postprocess_path"] == path
    assert guide["upscale_method"] == engine and guide["upscale_required"] is True


def test_selflift_performance_uses_simple_eight_step_schedule():
    from nodes import performance
    guide = short_guide()
    assert performance.scheduler_plan(guide) == dict(scheduler="simple", steps=8, split_step=4, refine_reference_tail=False)
    assert performance.sampler_name_for_guide(guide, "dpmpp_2m") == "euler"
    result = performance.MiniMaxH3PerformancePreset().apply(guide, acceleration_ready=True)
    assert result[:3] == (8, False, False)
    assert guide["two_stage_split_step"] == 4


def test_selflift_scheduler_ignores_stale_connected_step_count(monkeypatch):
    from nodes import performance
    calls = []
    class Scheduler:
        @staticmethod
        def execute(model, name, steps, denoise):
            calls.append((name, steps))
            return ([1] * steps + [0],)
    monkeypatch.setitem(sys.modules, "comfy_extras.nodes_custom_sampler", types.SimpleNamespace(BasicScheduler=Scheduler))
    performance.MiniMaxH3SchedulerRouter().route(object(), 12, short_guide())
    assert calls == [("simple", 8)]


def test_selflift_model_route_shares_one_exact_attention_model(monkeypatch):
    from nodes import performance
    calls = []
    monkeypatch.setattr(performance, "_load_lightx2v_lora", lambda model, name, **kw: calls.append(name) or "v4-model")
    monkeypatch.setattr(performance, "_apply_minimax_reuse_attention", lambda model, guide: model + ":exact")
    monkeypatch.setattr(performance, "_kj_ltx_class", lambda *args: pytest.fail("SelfLift must disable Sage"))
    guide = short_guide()
    first, _, ready, second = performance.MiniMaxH3AccelerationRouter().apply("model", guide)
    assert ready and first is second
    assert len(calls) == 1 and calls[0] == performance.V4_TURBO_LORA
    assert not guide["sage_requested"] and not guide["sage_applied"]


@pytest.mark.parametrize("failure", ["lora", "attention", "preset"])
def test_selflift_setup_never_bypasses_to_standard(monkeypatch, failure):
    from nodes import performance
    guide = short_guide()
    if failure == "preset":
        with pytest.raises(RuntimeError, match="SelfLift"):
            performance.MiniMaxH3PerformancePreset().apply(guide, acceleration_ready=False)
        return
    def fail(*args, **kwargs):
        raise performance.H3LoRAApplicationError("missing") if failure == "lora" else RuntimeError("missing")
    monkeypatch.setattr(performance, "_load_lightx2v_lora", fail if failure == "lora" else lambda model, *args, **kwargs: model)
    monkeypatch.setattr(performance, "_apply_minimax_reuse_attention", fail if failure == "attention" else lambda model, guide: model)
    with pytest.raises(RuntimeError, match="SelfLift"):
        performance.MiniMaxH3AccelerationRouter().apply("model", guide)


@pytest.mark.parametrize("profile,expected", [("off", []), ("continuity", [("Motion_Repair.safetensors", .25)]),
    ("combat", [("Motion_Repair.safetensors", .25), ("H3_Combat_V2.safetensors", .5)])])
def test_explicit_profile_ignores_legacy_environment_switches(monkeypatch, profile, expected):
    from nodes import performance
    monkeypatch.setenv("MMH3_DETAIL_LORAS", "1")
    monkeypatch.setenv("MMH3_EXTRA_LORAS", "0")
    monkeypatch.setattr(performance, "_resolve_first_registered", lambda category, names: names[0])
    calls = []
    monkeypatch.setattr(performance, "_load_lightx2v_lora", lambda model, name, strength: calls.append((name, strength)) or model)
    guide = short_guide(community_lora_profile=profile)
    performance._apply_ref_detail_loras("model", guide)
    assert calls == expected
    assert guide["community_loras_applied"] == [f"{name}@{strength}" for name, strength in expected]


@pytest.mark.parametrize("preset", ["quality_two_stage", "low_vram_two_stage"])
def test_explicit_off_suppresses_legacy_chain_on_every_route(monkeypatch, preset):
    from nodes import performance
    monkeypatch.setenv("MMH3_DETAIL_LORAS", "1")
    monkeypatch.setenv("MMH3_EXTRA_LORAS", "1")
    monkeypatch.setattr(performance, "_resolve_first_registered", lambda *args: pytest.fail("off must not resolve optional LoRAs"))
    guide = dict(performance_preset=preset, community_lora_profile="off")
    model = object()
    assert performance._apply_ref_detail_loras(model, guide) is model
    assert guide["community_loras_applied"] == []


def test_selflift_sampler_dispatches_before_standard_bypass(monkeypatch):
    from nodes import two_stage
    import torch
    calls = []
    def run(**kwargs):
        calls.append(kwargs)
        return kwargs["latent"]
    monkeypatch.setattr(two_stage, "run_selflift", run, raising=False)
    monkeypatch.setitem(sys.modules, "comfy_extras.nodes_custom_sampler", types.SimpleNamespace())
    guide = short_guide(two_stage_enabled=True)
    model = object()
    guider = types.SimpleNamespace(model_patcher=model, original_conds={"positive": [["embedding", {}]]})
    latent = {"samples": "target-av"}
    vae = object()
    result = two_stage.MiniMaxH3TwoStageSampler().execute(types.SimpleNamespace(seed=23), guider, object(),
        torch.linspace(1, 0, 9), latent, guide, second_model=model, video_vae=vae)
    assert result == (latent, latent)
    assert calls[0]["vae"] is vae and calls[0]["model"] is model
    assert calls[0]["seed"] == 23 and calls[0]["positive"] == [["embedding", {}]]


def test_selflift_uses_matched_accelerated_model_when_guider_keeps_sigma_shift_clone(monkeypatch):
    from nodes import two_stage
    import torch
    calls = []

    def run(**kwargs):
        calls.append(kwargs)
        return kwargs["latent"]

    monkeypatch.setattr(two_stage, "run_selflift", run, raising=False)
    monkeypatch.setitem(sys.modules, "comfy_extras.nodes_custom_sampler", types.SimpleNamespace())
    guider_model = object()
    accelerated_model = object()
    guider = types.SimpleNamespace(
        model_patcher=guider_model,
        original_conds={"positive": [["embedding", {}]]},
    )
    guide = short_guide(
        two_stage_enabled=True,
        turbo_lora_applied=True,
        head_chunking_applied=True,
        resolved_two_stage_route="trained_latent_ref",
        first_lora_name="minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors",
        second_lora_name="minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors",
    )
    latent = {"samples": "target-av"}

    result = two_stage.MiniMaxH3TwoStageSampler().execute(
        types.SimpleNamespace(seed=23), guider, object(), torch.linspace(1, 0, 9),
        latent, guide, second_model=accelerated_model, video_vae=object(),
    )

    assert result == (latent, latent)
    assert calls[0]["model"] is accelerated_model


def test_standard_bypass_records_actual_steps_route_and_time(monkeypatch):
    from nodes import two_stage
    ticks = iter((10.0, 13.0))
    monkeypatch.setattr(two_stage, "time", types.SimpleNamespace(perf_counter=lambda: next(ticks)))
    first, denoised = {"samples": object()}, {"samples": object()}
    sampler = types.SimpleNamespace(execute=lambda *args: (first, denoised))
    monkeypatch.setitem(sys.modules, "comfy_extras.nodes_custom_sampler", types.SimpleNamespace(SamplerCustomAdvanced=sampler))
    guide = {"two_stage_enabled": False}
    result = two_stage.MiniMaxH3TwoStageSampler().execute(object(), object(), object(),
        [1, .5, .2, 0], {"samples": object()}, guide)
    assert result == (first, denoised)
    assert guide["sampling_actual_method"] == "standard"
    assert guide["sampling_executed_steps"] == 3
    assert guide["sampling_route"] == "native"
    assert guide["sampling_elapsed_seconds"] == 3.0


@pytest.mark.parametrize("audio_guard,expected_steps", [(False, 8), (True, 12)])
def test_standard_two_stage_records_completed_work(monkeypatch, audio_guard, expected_steps):
    import torch
    from nodes import two_stage
    class PackedAV:
        is_nested = True
        def __init__(self, streams):
            self.streams = streams
        def unbind(self):
            return self.streams
    def concat(video, audio):
        return ({"samples": PackedAV((video["samples"], audio["samples"]))},)
    def separate(latent):
        return tuple({"samples": stream} for stream in latent["samples"].unbind())
    monkeypatch.setitem(sys.modules, "comfy.nested_tensor", types.SimpleNamespace(NestedTensor=PackedAV))
    monkeypatch.setitem(sys.modules, "comfy_extras.nodes_lt", types.SimpleNamespace(
        LTXVConcatAVLatent=types.SimpleNamespace(execute=concat),
        LTXVSeparateAVLatent=types.SimpleNamespace(execute=separate)))
    monkeypatch.setitem(sys.modules, "comfy_extras.nodes_custom_sampler", types.SimpleNamespace(
        SamplerCustomAdvanced=types.SimpleNamespace(execute=lambda noise, guider, sampler, sigmas, latent: (latent, latent)),
        Noise_EmptyNoise=lambda: object()))
    monkeypatch.setattr(two_stage, "_release_between_stages", lambda: None)
    monkeypatch.setattr(two_stage, "run_trained_latent_upscaler", lambda latent, scale: {
        "samples": torch.zeros(1, 24, 5, 6, 6)})
    video = {"samples": torch.zeros(1, 24, 5, 4, 4)}
    audio = {"samples": torch.ones(1, 32, 2, 20)}
    model = types.SimpleNamespace(model_options={})
    guider = types.SimpleNamespace(model_patcher=model, original_conds={"positive": []})
    guide = dict(two_stage_enabled=True, two_stage_split_step=4, two_stage_scale=1.5,
        two_stage_tiled=False, two_stage_audio_guard=audio_guard, resolved_two_stage_route="trained_latent_ref")
    two_stage.MiniMaxH3TwoStageSampler().execute(types.SimpleNamespace(seed=3), guider, object(),
        torch.linspace(1, 0, 9), concat(video, audio)[0], guide, second_model=model)
    assert guide["sampling_actual_method"] == "standard"
    assert guide["sampling_route"] == "trained_latent_ref"
    assert guide["sampling_executed_steps"] == expected_steps
    assert guide["two_stage_first_executed_steps"] == (8 if audio_guard else 4)
    assert guide["two_stage_second_executed_steps"] == 4
    assert guide["sampling_elapsed_seconds"] >= 0


def test_official_example_connects_video_vae_to_both_sampler_branches():
    root = Path(__file__).resolve().parents[1]
    path = next((root / "examples").glob("U11-*.json"))
    workflow = json.loads(path.read_text(encoding="utf-8"))
    settings = next(graph for graph in workflow["definitions"]["subgraphs"] if any(node.get("id") == 2584 for node in graph["nodes"]))
    for node_id in (2591, 2668):
        sampler = next(node for node in settings["nodes"] if node["id"] == node_id)
        vae = next(item for item in sampler["inputs"] if item["name"] == "video_vae")
        link = next(item for item in settings["links"] if item["id"] == vae["link"])
        assert link["origin_id"] == 2584 and link["target_id"] == node_id
        assert link["target_slot"] == sampler["inputs"].index(vae)
        loader = next(node for node in settings["nodes"] if node["id"] == 2584)
        assert link["id"] in loader["outputs"][link["origin_slot"]]["links"]
        assert settings["state"]["lastLinkId"] >= link["id"]
