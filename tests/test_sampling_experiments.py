import types

import pytest

from nodes import sampling_experiments as experiments


def euler_sampler():
    try:
        from comfy.samplers import sampler_object
    except ImportError:
        pass
    else:
        return sampler_object("euler")

    def sample_euler():
        pass
    return types.SimpleNamespace(sampler_function=sample_euler, extra_options={})


def request(**changes):
    return dict(mode="REF2VA", voice_mode="none", duration=5,
                performance_preset="quality_two_stage", sampling_method="selflift",
                community_lora_profile="off", **changes)


@pytest.mark.parametrize("size", [(1920, 1080), (1080, 1920), (2560, 1440),
                                 (1440, 2560), (3840, 2160), (2160, 3840)])
@pytest.mark.parametrize("duration", [4, 5, 10, 15])
def test_selflift_uses_target_canvas_and_matches_upstream_low_grid(size, duration):
    plan = experiments.plan_selflift_dimensions(*size, duration, 24, 22)
    assert plan["allowed"]
    w, h = plan["second_stage_width"], plan["second_stage_height"]
    assert w % 32 == h % 32 == 0
    assert w * h <= 1344 * 768
    assert (w > h) == (size[0] > size[1])
    assert plan["first_stage_width"] == max(2, round(w / 16 * 0.65 / 2) * 2) * 16
    assert plan["first_stage_height"] == max(2, round(h / 16 * 0.65 / 2) * 2) * 16
    assert not plan["two_stage_tiling_required"]
    assert plan["final_scale_x"] == size[0] / w
    assert plan["final_scale_y"] == size[1] / h


@pytest.mark.parametrize("preset,size", [("2K QHD", (2560, 1440)), ("4K UHD", (3840, 2160))])
@pytest.mark.parametrize("duration", [4, 5, 10, 15])
def test_selflift_options_accept_normal_duration_and_final_output(preset, size, duration):
    raw = request()
    raw.update(duration=duration, resolution_preset=preset,
               requested_width=size[0], requested_height=size[1])
    assert experiments.validate_sampling_options(raw)["sampling_method"] == "selflift"


@pytest.mark.parametrize("size", [(0, 1080), (1920, 0), (-1920, 1080)])
def test_selflift_rejects_invalid_output_dimensions(size):
    with pytest.raises(ValueError):
        experiments.plan_selflift_dimensions(*size, 15, 24, 22)


@pytest.mark.parametrize("total,free", [(8, 7), (16, 15), (24, 17)])
def test_unqualified_selflift_memory_is_rejected(total, free):
    with pytest.raises(ValueError, match="20GB|18GB"):
        experiments.plan_selflift_dimensions(1920, 1080, 5, total, free)


@pytest.mark.parametrize("key,value", [
    ("mode", "T2VA"), ("voice_mode", "h3_reference"),
    ("performance_preset", "low_vram"), ("sampling_method", "unknown"),
    ("community_lora_profile", "unknown"), ("motion_smoothing", "rife_x2"),
    ("fps", 48),
])
def test_invalid_experiment_does_not_silently_change_route(key, value):
    raw = request()
    raw[key] = value
    with pytest.raises(ValueError):
        experiments.validate_sampling_options(raw)


def test_motion_profiles_separate_general_motion_and_combat():
    assert experiments.motion_lora_chain("off") == ()
    continuity = experiments.motion_lora_chain("continuity")
    assert len(continuity) == 1 and continuity[0][1] == 0.25
    assert "Motion_Repair.safetensors" in continuity[0][0]
    combat = experiments.motion_lora_chain("combat")
    assert len(combat) == 2 and combat[1] == (("H3_Combat_V2.safetensors",), 0.5)
    assert not any("GunFu" in name or "fafafast" in name for names, _ in combat for name in names)


class SamplerNode:
    def sample(self, model, positive, negative, vae, latent_image, sampler, sigmas,
               seed, cfg, transition_step, lowres_scale, rho, w_min, w_max,
               upscaler_model, model_hires=None, highres_tiling=False, upscaler_unload=True):
        self.received = locals()
        # The installed node resizes both branches before sampling, even at CFG 1.
        self.conditioning = [
            [(tensor, options.get("minimax_keyframes")) for tensor, options in branch]
            for branch in (positive, negative)
        ]
        return (latent_image.copy(),)


class AV:
    is_nested = True

    def unbind(self):
        return (types.SimpleNamespace(shape=(1, 24, 8, 48, 84)),
                types.SimpleNamespace(shape=(1, 32, 2, 208)))


def test_selflift_missing_or_obsolete_dependency_fails_preflight():
    assert not experiments.selflift_dependency_report({}, "model.safetensors")["ready"]
    class Obsolete:
        def sample(self, model, sigmas):
            pass
    report = experiments.selflift_dependency_report({"SelfLiftH3Sampler": Obsolete}, "model.safetensors")
    assert not report["ready"] and report["missing"]


def test_adapter_uses_one_eight_step_schedule_and_unloads_lifter(monkeypatch):
    ticks = iter((10.0, 12.5))
    monkeypatch.setattr(experiments, "time", types.SimpleNamespace(perf_counter=lambda: next(ticks)), raising=False)
    node = SamplerNode()
    guide = dict(sampling_method="selflift", selflift_upscaler_model="h3_fp16.safetensors")
    latent = {"samples": AV()}
    result = experiments.run_selflift(
        model=object(), positive=[], vae=object(), sampler=euler_sampler(),
        sigmas=[1, .9, .8, .7, .6, .4, .2, .1, 0],
        latent=latent, seed=42, guide=guide,
        registry={"SelfLiftH3Sampler": lambda: node})
    assert result["samples"] is latent["samples"]
    actual = node.received
    assert actual["transition_step"] == 4 and actual["lowres_scale"] == 0.65
    assert actual["rho"] == 0 and actual["cfg"] == 1
    assert actual["upscaler_unload"] and not actual["highres_tiling"]
    assert actual["upscaler_model"] == "h3_fp16.safetensors"
    assert actual["model_hires"] is actual["model"]
    assert guide["sampling_executed_steps"] == 8
    assert guide["sampling_actual_method"] == "selflift"
    assert guide["sampling_route"] == "SelfLiftH3Sampler"
    assert guide["sampling_elapsed_seconds"] == 2.5
    assert guide["two_stage_first_executed_steps"] == guide["two_stage_second_executed_steps"] == 4


def test_cfg_one_uses_iterable_empty_negative_conditioning():
    node = SamplerNode()
    embedding = object()
    experiments.run_selflift(model=object(), positive=[[embedding, {}]], vae=object(),
        sampler=euler_sampler(), sigmas=[1, .9, .8, .7, .6, .4, .2, .1, 0],
        latent={"samples": AV()}, seed=1, guide={"selflift_upscaler_model": "h3.safetensors"},
        registry={"SelfLiftH3Sampler": lambda: node})
    assert node.received["negative"] == [] and node.received["cfg"] == 1.0
    assert node.conditioning == [[(embedding, None)], []]


@pytest.mark.parametrize("change", ["vae", "seed", "sigmas", "samples", "sampler"])
def test_adapter_rejects_invalid_execution_before_external_sampling(change):
    args = dict(model=object(), positive=[], vae=object(), sampler=euler_sampler(),
                sigmas=[1, .9, .8, .7, .6, .4, .2, .1, 0],
                latent={"samples": AV()}, seed=42,
                guide={"selflift_upscaler_model": "h3.safetensors"},
                registry={"SelfLiftH3Sampler": SamplerNode})
    if change == "samples":
        args["latent"] = {"samples": object()}
    else:
        args[change] = {"vae": None, "seed": None, "sigmas": [1, .5, 0], "sampler": object()}[change]
    with pytest.raises((ValueError, RuntimeError)):
        experiments.run_selflift(**args)


def test_adapter_checks_returned_av_shape():
    class Broken(SamplerNode):
        def sample(self, model, positive, negative, vae, latent_image, sampler, sigmas,
                   seed, cfg, transition_step, lowres_scale, rho, w_min, w_max,
                   upscaler_model, model_hires=None, highres_tiling=False, upscaler_unload=True):
            return ({"samples": object()},)
    with pytest.raises(RuntimeError, match="AV|latent"):
        experiments.run_selflift(
            model=object(), positive=[], vae=object(), sampler=euler_sampler(),
            sigmas=[1, .9, .8, .7, .6, .4, .2, .1, 0], latent={"samples": AV()},
            seed=42, guide={"selflift_upscaler_model": "h3.safetensors"},
            registry={"SelfLiftH3Sampler": Broken})


@pytest.mark.parametrize("sigmas", [
    [1, .9, .8, .7, .6, .4, .2, float("nan"), 0],
    [1, .9, .8, .7, .6, .4, .5, .1, 0],
    [1, .9, .8, .7, .6, .4, .2, .1, .05],
])
def test_adapter_rejects_invalid_eight_step_schedule(sigmas):
    with pytest.raises(ValueError, match="sigma"):
        experiments.run_selflift(model=object(), positive=[], vae=object(),
            sampler=euler_sampler(), sigmas=sigmas, latent={"samples": AV()}, seed=1,
            guide={"selflift_upscaler_model": "h3.safetensors"},
            registry={"SelfLiftH3Sampler": SamplerNode})


def test_no_upscaler_cannot_disable_selflift_correction():
    assert not experiments.selflift_dependency_report({"SelfLiftH3Sampler": SamplerNode}, "none")["ready"]


def test_adapter_rejects_low_grid_as_target_template():
    with pytest.raises(ValueError, match="目标|target"):
        experiments.run_selflift(model=object(), positive=[], vae=object(),
            sampler=euler_sampler(), sigmas=[1, .9, .8, .7, .6, .4, .2, .1, 0],
            latent={"samples": AV()}, seed=1,
            guide={"selflift_upscaler_model": "h3.safetensors", "second_stage_width": 1920, "second_stage_height": 1088},
            registry={"SelfLiftH3Sampler": SamplerNode})


def test_adapter_rejects_changed_audio_shape_even_when_video_matches():
    class ChangedAudio(AV):
        def unbind(self):
            video, _ = super().unbind()
            return video, types.SimpleNamespace(shape=(1, 32, 2, 207))
    class Broken(SamplerNode):
        def sample(self, model, positive, negative, vae, latent_image, sampler, sigmas,
                   seed, cfg, transition_step, lowres_scale, rho, w_min, w_max,
                   upscaler_model, model_hires=None, highres_tiling=False, upscaler_unload=True):
            return ({"samples": ChangedAudio()},)
    with pytest.raises(RuntimeError, match="AV.*尺寸|时长"):
        experiments.run_selflift(model=object(), positive=[], vae=object(),
            sampler=euler_sampler(), sigmas=[1, .9, .8, .7, .6, .4, .2, .1, 0],
            latent={"samples": AV()}, seed=1, guide={"selflift_upscaler_model": "h3.safetensors"},
            registry={"SelfLiftH3Sampler": Broken})


def test_adapter_reports_external_failure_as_selflift_failure():
    class Failing(SamplerNode):
        def sample(self, model, positive, negative, vae, latent_image, sampler, sigmas,
                   seed, cfg, transition_step, lowres_scale, rho, w_min, w_max,
                   upscaler_model, model_hires=None, highres_tiling=False, upscaler_unload=True):
            raise RuntimeError("missing learned weights")
    guide = {"selflift_upscaler_model": "h3.safetensors"}
    with pytest.raises(RuntimeError, match="SelfLift.*missing learned weights"):
        experiments.run_selflift(model=object(), positive=[], vae=object(),
            sampler=euler_sampler(), sigmas=[1, .9, .8, .7, .6, .4, .2, .1, 0],
            latent={"samples": AV()}, seed=1, guide=guide,
            registry={"SelfLiftH3Sampler": Failing})
    assert "sampling_executed_steps" not in guide
