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


def test_two_stage_refines_first_sampler_output_not_denoised_preview(monkeypatch):
    import nodes.performance as performance

    sample_output = {"samples": torch.zeros(1, 4, 2, 4, 4)}
    denoised_preview = {"samples": torch.ones(1, 4, 2, 4, 4)}
    separate_inputs = []
    second_latents = []

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

    def separate(latent):
        separate_inputs.append(latent)
        return latent, {"samples": torch.zeros(1, 2, 2, 1, 1)}

    def concat(video, audio):
        return types.SimpleNamespace(result=(video,))

    monkeypatch.setattr(performance, "memory_policy", lambda guide: nullcontext())
    monkeypatch.setitem(sys.modules, "comfy_extras.nodes_custom_sampler", types.SimpleNamespace(
        Noise_RandomNoise=FakeNoise,
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

    assert len(separate_inputs) == 1
    assert separate_inputs[0] is sample_output
    assert second_latents[0]["samples"].shape[-2:] == (6, 6)
