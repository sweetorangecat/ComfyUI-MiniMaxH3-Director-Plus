from contextlib import nullcontext
import sys
import types

import pytest
import torch

from nodes.performance import MiniMaxH3PerformancePreset, preset_values
from nodes.schema import allowed_performance_presets
from nodes.two_stage import (
    MiniMaxH3TwoStageSampler,
    _node_output,
    clone_guider_with_model,
    split_sigmas_at_step,
)


def test_quality_two_stage_is_high_vram_only_route():
    assert "quality_two_stage" in allowed_performance_presets("T2VA", "none")
    assert "quality_two_stage" in allowed_performance_presets("REF2VA", "none")
    assert "quality_two_stage" not in allowed_performance_presets("T2VA", "fish_lock")
    values = preset_values("quality_two_stage")
    assert values["steps"] == 8
    assert values["two_stage_split_step"] == 4
    assert values["two_stage_scale"] == pytest.approx(1.5)


def test_performance_node_marks_two_stage_guide():
    guide = {"mode": "T2VA", "voice_mode": "none", "performance_preset": "quality_two_stage"}
    result = MiniMaxH3PerformancePreset().apply(guide, acceleration_ready=True)
    assert result[0] == 8
    assert guide["two_stage_enabled"] is True
    assert guide["two_stage_split_step"] == 4
    assert "latent" in result[3]


def test_performance_node_marks_low_vram_two_stage_guide():
    guide = {
        "mode": "T2VA",
        "voice_mode": "none",
        "performance_preset": "low_vram_two_stage",
    }

    result = MiniMaxH3PerformancePreset().apply(guide, acceleration_ready=True)

    assert result[0] == 8
    assert guide["two_stage_enabled"] is True
    assert guide["two_stage_split_step"] == 4
    assert guide["two_stage_scale"] == pytest.approx(1.5)


def test_quality_two_stage_has_no_interpolation_call():
    import nodes.two_stage as two_stage

    assert not hasattr(two_stage, "upscale_video_latent")
    assert not hasattr(two_stage, "prepare_h3_two_stage_latent")
    assert not hasattr(two_stage, "upscale_h3_guider")


def test_split_at_step_matches_comfy_split_sigmas():
    sigmas = torch.tensor([10.0, 8.0, 6.0, 4.0, 2.0, 1.0, 0.5, 0.2, 0.0])
    high, low = split_sigmas_at_step(sigmas, 4)
    assert torch.equal(high, sigmas[:5])
    assert torch.equal(low, sigmas[4:])


def test_reference_refined_tail_splits_to_four_plus_five_steps():
    sigmas = torch.tensor([10.0, 8.0, 6.0, 4.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.0])
    high, low = split_sigmas_at_step(sigmas, 4)
    assert len(high) == 5
    assert len(low) == 6


def test_clone_guider_switches_only_model_contract():
    first_model = types.SimpleNamespace(model_options={"route": "first"})
    second_model = types.SimpleNamespace(model_options={"route": "second"})
    guider = types.SimpleNamespace(
        model_patcher=first_model,
        model_options=first_model.model_options,
        original_conds={"positive": [{"prompt": "keep"}]},
    )

    result = clone_guider_with_model(guider, second_model)

    assert result is not guider
    assert result.model_patcher is second_model
    assert result.model_options is second_model.model_options
    assert result.original_conds is guider.original_conds


def test_two_stage_sampler_exposes_second_model_without_video_vae():
    inputs = MiniMaxH3TwoStageSampler.INPUT_TYPES()
    assert "guide" in inputs["optional"]
    assert "second_model" in inputs["optional"]
    assert "video_vae" not in inputs["optional"]
    assert MiniMaxH3TwoStageSampler.RETURN_TYPES == ("LATENT", "LATENT")


def test_enabled_two_stage_fails_before_sampling_without_second_model(monkeypatch):
    class ForbiddenSampler:
        @classmethod
        def execute(cls, *args):
            raise AssertionError("缺失第二模型时不得开始首采")

    monkeypatch.setitem(
        sys.modules,
        "comfy_extras.nodes_custom_sampler",
        types.SimpleNamespace(SamplerCustomAdvanced=ForbiddenSampler),
    )
    with pytest.raises(ValueError, match="第二阶段模型"):
        MiniMaxH3TwoStageSampler().execute(
            object(),
            types.SimpleNamespace(model_patcher=object()),
            object(),
            torch.tensor([1.0, 0.0]),
            {"samples": torch.zeros(1, 24, 1, 2, 2)},
            {"two_stage_enabled": True},
        )


def test_trained_two_stage_preserves_audio_and_uses_fl_four_plus_four(monkeypatch):
    from comfy_extras.nodes_lt import LTXVConcatAVLatent
    import nodes.performance as performance
    import nodes.two_stage as two_stage

    original_video = {"samples": torch.ones(1, 24, 5, 4, 4)}
    original_audio = {"samples": torch.full((1, 32, 2, 20), 7.0)}
    first_denoised = _node_output(LTXVConcatAVLatent.execute(original_video, original_audio))
    calls = []
    stage_events = []

    class FakeNoise:
        def __init__(self, seed):
            self.seed = seed

    class FakeSampler:
        @classmethod
        def execute(cls, noise, guider, sampler, sigmas, latent):
            calls.append((noise, guider, sigmas.clone(), latent))
            if len(calls) == 1:
                return types.SimpleNamespace(result=(first_denoised, first_denoised))
            return types.SimpleNamespace(result=(latent, latent))

    def fake_upscale(video_latent, scale):
        stage_events.append("upscale")
        assert scale == 1.5
        assert video_latent["samples"].shape == (1, 24, 5, 4, 4)
        return {"samples": torch.ones(1, 24, 5, 6, 6)}

    monkeypatch.setattr(performance, "memory_policy", lambda guide: nullcontext())
    monkeypatch.setattr(
        two_stage,
        "_release_between_stages",
        lambda: stage_events.append("release"),
        raising=False,
    )
    monkeypatch.setattr(two_stage, "run_trained_latent_upscaler", fake_upscale)
    monkeypatch.setitem(
        sys.modules,
        "comfy_extras.nodes_custom_sampler",
        types.SimpleNamespace(Noise_RandomNoise=FakeNoise, SamplerCustomAdvanced=FakeSampler),
    )
    first_model = types.SimpleNamespace(model_options={"route": "first"})
    second_model = types.SimpleNamespace(model_options={"route": "second"})
    guider = types.SimpleNamespace(
        model_patcher=first_model,
        model_options=first_model.model_options,
        original_conds={"positive": []},
    )
    sigmas = torch.tensor([10.0, 8.0, 6.0, 4.0, 2.0, 1.0, 0.5, 0.2, 0.0])
    guide = {
        "two_stage_enabled": True,
        "two_stage_split_step": 4,
        "two_stage_scale": 1.5,
        "resolved_two_stage_route": "trained_latent_fl",
    }

    result = MiniMaxH3TwoStageSampler().execute(
        FakeNoise(9),
        guider,
        object(),
        sigmas,
        {"samples": torch.zeros(1, 24, 5, 4, 4)},
        guide,
        second_model=second_model,
    )

    assert result[0] is result[1]
    assert len(calls) == 2
    assert stage_events == ["release", "upscale"]
    assert torch.equal(calls[0][2], sigmas[:5])
    assert torch.equal(calls[1][2], sigmas[4:])
    assert calls[1][0].seed == 9
    assert calls[1][1].model_patcher is second_model
    second_video, second_audio = calls[1][3]["samples"].unbind()
    assert second_video.shape == (1, 24, 5, 6, 6)
    assert torch.equal(second_audio, original_audio["samples"])
    assert guide["two_stage_status"] == "训练型 3D latent 二采完成"


def test_disabled_route_still_runs_one_native_sampler_call(monkeypatch):
    import nodes.performance as performance

    calls = []

    class FakeSampler:
        @classmethod
        def execute(cls, *args):
            calls.append(args)
            return types.SimpleNamespace(result=("sampled", "denoised"))

    monkeypatch.setattr(performance, "memory_policy", lambda guide: nullcontext())
    monkeypatch.setitem(
        sys.modules,
        "comfy_extras.nodes_custom_sampler",
        types.SimpleNamespace(SamplerCustomAdvanced=FakeSampler),
    )
    guide = {"two_stage_enabled": False}
    result = MiniMaxH3TwoStageSampler().execute(
        object(), object(), object(), torch.tensor([1.0, 0.0]), {"samples": torch.zeros(1)}, guide
    )
    assert result == ("sampled", "denoised")
    assert len(calls) == 1
    assert guide["two_stage_status"] == "旁路"
