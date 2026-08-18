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
    split_refinement_sigmas,
    upscale_video_latent,
)


def test_quality_two_stage_is_high_vram_only_route():
    assert "quality_two_stage" in allowed_performance_presets("T2VA", "none")
    assert "quality_two_stage" in allowed_performance_presets("REF2VA", "none")
    assert "quality_two_stage" not in allowed_performance_presets("T2VA", "fish_lock")
    values = preset_values("quality_two_stage")
    assert values["steps"] == 14
    assert values["two_stage_steps"] == 6
    assert values["two_stage_scale"] == pytest.approx(1.5)


def test_performance_node_marks_two_stage_guide():
    guide = {"mode": "T2VA", "voice_mode": "none", "performance_preset": "quality_two_stage"}
    result = MiniMaxH3PerformancePreset().apply(guide)
    assert result[0] == 14
    assert guide["two_stage_enabled"] is True
    assert guide["two_stage_steps"] == 6
    assert "二阶段" in result[3]


def test_split_refinement_sigmas_keeps_continuous_boundary():
    sigmas = torch.tensor([10.0, 7.0, 4.0, 2.0, 1.0, 0.0])
    high, low = split_refinement_sigmas(sigmas, 2)
    assert torch.equal(high, torch.tensor([10.0, 7.0, 4.0, 2.0]))
    assert torch.equal(low, torch.tensor([2.0, 1.0, 0.0]))


def test_upscale_video_latent_changes_spatial_shape_only():
    video = torch.zeros(1, 24, 5, 8, 12)
    result = upscale_video_latent({"samples": video}, 1.5)
    assert result["samples"].shape == (1, 24, 5, 12, 18)


def test_real_av_nodes_keep_audio_shape_during_video_refinement():
    from comfy_extras.nodes_lt import LTXVConcatAVLatent, LTXVSeparateAVLatent

    video = {"samples": torch.zeros(1, 128, 2, 4, 4)}
    audio = {"samples": torch.zeros(1, 8, 2, 1, 1)}
    combined = _node_output(LTXVConcatAVLatent.execute(video, audio))
    video_part, audio_part = LTXVSeparateAVLatent.execute(combined)
    refined_video = upscale_video_latent(video_part, 1.5)
    merged = _node_output(LTXVConcatAVLatent.execute(refined_video, audio_part))
    video_samples, audio_samples = merged["samples"].unbind()

    assert video_samples.shape == (1, 128, 2, 6, 6)
    assert audio_samples.shape == (1, 8, 2, 1, 1)


def test_two_stage_sampler_exposes_guide_and_two_latent_outputs():
    inputs = MiniMaxH3TwoStageSampler.INPUT_TYPES()
    assert "guide" in inputs["optional"]
    assert MiniMaxH3TwoStageSampler.RETURN_TYPES == ("LATENT", "LATENT")


def test_two_stage_refines_denoised_video_and_preserves_sampled_audio(monkeypatch):
    import nodes.performance as performance

    sample_output = {"samples": torch.zeros(1, 4, 2, 4, 4)}
    denoised_preview = {"samples": torch.ones(1, 4, 2, 4, 4)}
    separate_inputs = []
    second_latents = []
    sampled_audio = {"samples": torch.full((1, 2, 2, 1, 1), 7.0)}
    denoised_audio = {"samples": torch.full((1, 2, 2, 1, 1), 9.0)}

    class FakeSampler:
        calls = 0

        @classmethod
        def execute(cls, noise, guider, sampler, sigmas, latent):
            cls.calls += 1
            if cls.calls == 1:
                return types.SimpleNamespace(result=(sample_output, denoised_preview))
            second_latents.append(latent)
            return types.SimpleNamespace(result=(latent, latent))

    class FakeNoise:
        def __init__(self, seed):
            self.seed = seed

    class FakeEmptyNoise:
        seed = 0

    def separate(latent):
        separate_inputs.append(latent)
        audio = sampled_audio if latent is sample_output else denoised_audio
        return latent, audio

    def concat(video, audio):
        assert video is not sample_output
        assert audio is sampled_audio
        return types.SimpleNamespace(result=(video,))

    monkeypatch.setattr(performance, "memory_policy", lambda guide: nullcontext())
    monkeypatch.setitem(sys.modules, "comfy_extras.nodes_custom_sampler", types.SimpleNamespace(
        Noise_RandomNoise=FakeNoise,
        Noise_EmptyNoise=FakeEmptyNoise,
        SamplerCustomAdvanced=FakeSampler,
    ))
    monkeypatch.setitem(sys.modules, "comfy_extras.nodes_lt", types.SimpleNamespace(
        LTXVSeparateAVLatent=types.SimpleNamespace(execute=staticmethod(separate)),
        LTXVConcatAVLatent=types.SimpleNamespace(execute=staticmethod(concat)),
    ))

    MiniMaxH3TwoStageSampler().execute(
        FakeNoise(1), object(), object(), torch.tensor([10.0, 7.0, 4.0, 2.0, 1.0, 0.0]),
        {"samples": torch.zeros(1, 4, 2, 4, 4)},
        {"two_stage_enabled": True, "two_stage_steps": 2, "two_stage_scale": 1.5},
    )

    assert separate_inputs == [sample_output, denoised_preview]
    assert second_latents[0]["samples"].shape[-2:] == (6, 6)


def test_two_stage_returns_final_denoised_latent_for_video_and_audio(monkeypatch):
    import nodes.performance as performance

    first_sampled = {"samples": torch.zeros(1, 4, 2, 4, 4)}
    first_denoised = {"samples": torch.ones(1, 4, 2, 4, 4)}
    final_sampled = {"samples": torch.full((1, 4, 2, 6, 6), 2.0)}
    final_denoised = {"samples": torch.full((1, 4, 2, 6, 6), 3.0)}

    class FakeSampler:
        calls = 0

        @classmethod
        def execute(cls, noise, guider, sampler, sigmas, latent):
            cls.calls += 1
            if cls.calls == 1:
                return types.SimpleNamespace(result=(first_sampled, first_denoised))
            return types.SimpleNamespace(result=(final_sampled, final_denoised))

    class FakeNoise:
        def __init__(self, seed):
            self.seed = seed

    class FakeEmptyNoise:
        seed = 0

    def separate(latent):
        return latent, {"samples": torch.zeros(1, 2, 2, 1, 1)}

    def concat(video, audio):
        return types.SimpleNamespace(result=(video,))

    monkeypatch.setattr(performance, "memory_policy", lambda guide: nullcontext())
    monkeypatch.setitem(sys.modules, "comfy_extras.nodes_custom_sampler", types.SimpleNamespace(
        Noise_RandomNoise=FakeNoise,
        Noise_EmptyNoise=FakeEmptyNoise,
        SamplerCustomAdvanced=FakeSampler,
    ))
    monkeypatch.setitem(sys.modules, "comfy_extras.nodes_lt", types.SimpleNamespace(
        LTXVSeparateAVLatent=types.SimpleNamespace(execute=staticmethod(separate)),
        LTXVConcatAVLatent=types.SimpleNamespace(execute=staticmethod(concat)),
    ))

    result = MiniMaxH3TwoStageSampler().execute(
        FakeNoise(1), object(), object(), torch.tensor([10.0, 7.0, 4.0, 2.0, 1.0, 0.0]),
        {"samples": torch.zeros(1, 4, 2, 4, 4)},
        {"two_stage_enabled": True, "two_stage_steps": 2, "two_stage_scale": 1.5},
    )

    assert result[0] is final_denoised
    assert result[1] is final_denoised


def test_two_stage_final_refinement_uses_empty_noise(monkeypatch):
    import nodes.performance as performance

    calls = []

    class FakeRandomNoise:
        def __init__(self, seed):
            self.seed = seed

    class FakeEmptyNoise:
        seed = 0

    class FakeSampler:
        @classmethod
        def execute(cls, noise, guider, sampler, sigmas, latent):
            calls.append(noise)
            sample = {"samples": torch.zeros(1, 4, 2, 4, 4)}
            return types.SimpleNamespace(result=(sample, sample))

    def separate(latent):
        return latent, {"samples": torch.zeros(1, 2, 2, 1, 1)}

    def concat(video, audio):
        return types.SimpleNamespace(result=(video,))

    monkeypatch.setattr(performance, "memory_policy", lambda guide: nullcontext())
    monkeypatch.setitem(sys.modules, "comfy_extras.nodes_custom_sampler", types.SimpleNamespace(
        Noise_RandomNoise=FakeRandomNoise,
        Noise_EmptyNoise=FakeEmptyNoise,
        SamplerCustomAdvanced=FakeSampler,
    ))
    monkeypatch.setitem(sys.modules, "comfy_extras.nodes_lt", types.SimpleNamespace(
        LTXVSeparateAVLatent=types.SimpleNamespace(execute=staticmethod(separate)),
        LTXVConcatAVLatent=types.SimpleNamespace(execute=staticmethod(concat)),
    ))

    MiniMaxH3TwoStageSampler().execute(
        FakeRandomNoise(1), object(), object(), torch.tensor([10.0, 7.0, 4.0, 2.0, 1.0, 0.0]),
        {"samples": torch.zeros(1, 4, 2, 4, 4)},
        {"two_stage_enabled": True, "two_stage_steps": 2, "two_stage_scale": 1.5},
    )

    assert len(calls) == 3
    assert isinstance(calls[0], FakeRandomNoise)
    assert isinstance(calls[1], FakeRandomNoise)
    assert isinstance(calls[2], FakeEmptyNoise)


def test_two_stage_injects_boundary_noise_before_final_refinement(monkeypatch):
    import nodes.performance as performance

    calls = []
    first_sampled = {"samples": torch.zeros(1, 4, 2, 4, 4)}
    first_denoised = {"samples": torch.ones(1, 4, 2, 4, 4)}
    boundary_sampled = {"samples": torch.full((1, 4, 2, 6, 6), 2.0)}
    final_denoised = {"samples": torch.full((1, 4, 2, 6, 6), 3.0)}

    class FakeRandomNoise:
        def __init__(self, seed):
            self.seed = seed

    class FakeEmptyNoise:
        seed = 0

    class FakeSampler:
        @classmethod
        def execute(cls, noise, guider, sampler, sigmas, latent):
            calls.append((noise, sigmas.clone(), latent))
            if len(calls) == 1:
                return types.SimpleNamespace(result=(first_sampled, first_denoised))
            if len(calls) == 2:
                return types.SimpleNamespace(result=(boundary_sampled, boundary_sampled))
            return types.SimpleNamespace(result=(boundary_sampled, final_denoised))

    def separate(latent):
        if latent is first_sampled:
            return latent, {"samples": torch.zeros(1, 2, 2, 1, 1)}
        return latent, {"samples": torch.zeros(1, 2, 2, 1, 1)}

    def concat(video, audio):
        return types.SimpleNamespace(result=(video,))

    monkeypatch.setattr(performance, "memory_policy", lambda guide: nullcontext())
    monkeypatch.setitem(sys.modules, "comfy_extras.nodes_custom_sampler", types.SimpleNamespace(
        Noise_RandomNoise=FakeRandomNoise,
        Noise_EmptyNoise=FakeEmptyNoise,
        SamplerCustomAdvanced=FakeSampler,
    ))
    monkeypatch.setitem(sys.modules, "comfy_extras.nodes_lt", types.SimpleNamespace(
        LTXVSeparateAVLatent=types.SimpleNamespace(execute=staticmethod(separate)),
        LTXVConcatAVLatent=types.SimpleNamespace(execute=staticmethod(concat)),
    ))

    MiniMaxH3TwoStageSampler().execute(
        FakeRandomNoise(1), object(), object(), torch.tensor([10.0, 7.0, 4.0, 2.0, 1.0, 0.0]),
        {"samples": torch.zeros(1, 4, 2, 4, 4)},
        {"two_stage_enabled": True, "two_stage_steps": 2, "two_stage_scale": 1.5},
    )

    assert len(calls) == 3
    assert isinstance(calls[1][0], FakeRandomNoise)
    assert calls[1][1].shape == (1,)
    assert calls[1][2]["samples"].shape[-2:] == (6, 6)
    assert isinstance(calls[2][0], FakeEmptyNoise)
    assert calls[2][2] is boundary_sampled
