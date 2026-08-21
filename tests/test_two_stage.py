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
    assert values["steps"] == 8
    assert values["two_stage_steps"] == 5
    assert values["two_stage_scale"] == pytest.approx(1.5)


def test_performance_node_marks_two_stage_guide():
    guide = {"mode": "T2VA", "voice_mode": "none", "performance_preset": "quality_two_stage"}
    result = MiniMaxH3PerformancePreset().apply(guide)
    assert result[0] == 8
    assert guide["two_stage_enabled"] is True
    assert guide["two_stage_steps"] == 5
    assert "latent" in result[3]


def test_split_refinement_sigmas_keeps_continuous_boundary():
    sigmas = torch.tensor([10.0, 7.0, 4.0, 2.0, 1.0, 0.0])
    high, low = split_refinement_sigmas(sigmas, 2)
    assert torch.equal(high, torch.tensor([10.0, 7.0, 4.0, 2.0]))
    assert torch.equal(low, torch.tensor([2.0, 1.0, 0.0]))


def test_upscale_video_latent_changes_spatial_shape_only():
    video = torch.zeros(1, 24, 5, 8, 12)
    result = upscale_video_latent({"samples": video}, 1.5)
    assert result["samples"].shape == (1, 24, 5, 12, 18)


def test_upscale_video_latent_snaps_target_to_h3_spatial_patch_grid():
    video = torch.zeros(1, 24, 5, 34, 60)
    result = upscale_video_latent({"samples": video}, 1.5)
    assert result["samples"].shape == (1, 24, 5, 52, 90)


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
    assert "video_vae" in inputs["optional"]
    assert MiniMaxH3TwoStageSampler.RETURN_TYPES == ("LATENT", "LATENT")


def test_prepare_h3_two_stage_latent_stays_in_latent_space_and_locks_audio():
    import nodes.two_stage as two_stage
    from comfy.nested_tensor import NestedTensor

    prepare = getattr(two_stage, "prepare_h3_two_stage_latent", None)
    assert callable(prepare), "缺少 H3 专用 latent 重加噪准备"

    class FakeNoise:
        def generate_noise(self, latent):
            video, audio = latent["samples"].unbind()
            return NestedTensor([torch.ones_like(video), torch.ones_like(audio)])

    class FakeSampling:
        def noise_scaling(self, sigma, noise, latent):
            return sigma * noise + (1.0 - sigma) * latent

        def inverse_noise_scaling(self, sigma, samples):
            return samples / (1.0 - sigma)

    class FakeModel:
        objects = {
            "model_sampling": FakeSampling(),
            "process_latent_in": lambda value: value,
            "process_latent_out": lambda value: value,
        }

        def get_model_object(self, name):
            return self.objects[name]

    video = torch.ones(1, 24, 5, 34, 60)
    audio = torch.full((1, 32, 2, 20), 7.0)
    latent = {"samples": NestedTensor([video, audio])}
    result = prepare(latent, FakeModel(), FakeNoise(), torch.tensor([0.25, 0.0]), 1.5)
    video_out, audio_out = result["samples"].unbind()
    video_mask, audio_mask = result["noise_mask"].unbind()

    assert video_out.shape == (1, 24, 5, 52, 90)
    assert torch.allclose(audio_out, audio)
    assert torch.count_nonzero(video_mask) == video_mask.numel()
    assert torch.count_nonzero(audio_mask) == 0
    assert video_out.device.type == "cpu"
    assert audio_out.device.type == "cpu"


def test_upscale_h3_guider_scales_reference_metadata_without_mutating_source():
    import nodes.two_stage as two_stage

    upscale_guider = getattr(two_stage, "upscale_h3_guider", None)
    assert callable(upscale_guider), "缺少 REF2VA/I2VA 二采 conditioning 同步"

    ref = {"kind": "image", "latent": torch.ones(1, 24, 1, 3, 5), "latent_h": 3, "latent_w": 5}
    guider = types.SimpleNamespace(
        original_conds={"positive": [{"minimax_refs": [ref]}]},
        model_patcher=object(),
    )
    result = upscale_guider(guider, 1.5)
    scaled = result.original_conds["positive"][0]["minimax_refs"][0]

    assert result is not guider
    assert scaled["latent"].shape[-2:] == (4, 8)
    assert (scaled["latent_h"], scaled["latent_w"]) == (4, 8)
    assert ref["latent"].shape[-2:] == (3, 5)


def test_two_stage_uses_latent_resample_and_two_sampler_calls(monkeypatch):
    import nodes.performance as performance
    import nodes.two_stage as two_stage

    first_sampled = {"samples": torch.zeros(1, 4, 2, 4, 4)}
    first_denoised = {"samples": torch.ones(1, 4, 2, 4, 4)}
    prepared = {"samples": torch.ones(1, 4, 2, 6, 6)}
    final_denoised = {"samples": torch.full((1, 4, 2, 6, 6), 3.0)}
    calls = []

    class FakeRandomNoise:
        def __init__(self, seed):
            self.seed = seed

    class FakeEmptyNoise:
        seed = 0

    class FakeSampler:
        @classmethod
        def execute(cls, noise, guider, sampler, sigmas, latent):
            calls.append((noise, sigmas.clone(), latent, guider))
            if len(calls) == 1:
                return types.SimpleNamespace(result=(first_sampled, first_denoised))
            return types.SimpleNamespace(result=(prepared, final_denoised))

    class ForbiddenVAE:
        def decode(self, _samples):
            raise AssertionError("15 秒整段二采不得进入 VAE 图像空间")

        def encode(self, _pixels):
            raise AssertionError("15 秒整段二采不得进入 VAE 图像空间")

    fake_guider = types.SimpleNamespace(model_patcher=object(), original_conds={"positive": []})
    monkeypatch.setattr(performance, "memory_policy", lambda guide: nullcontext())
    monkeypatch.setattr(two_stage, "prepare_h3_two_stage_latent", lambda *args: prepared, raising=False)
    monkeypatch.setattr(two_stage, "upscale_h3_guider", lambda guider, scale: guider, raising=False)
    monkeypatch.setitem(sys.modules, "comfy_extras.nodes_custom_sampler", types.SimpleNamespace(
        Noise_RandomNoise=FakeRandomNoise,
        Noise_EmptyNoise=FakeEmptyNoise,
        SamplerCustomAdvanced=FakeSampler,
    ))

    result = MiniMaxH3TwoStageSampler().execute(
        FakeRandomNoise(9), fake_guider, object(),
        torch.tensor([10.0, 7.0, 4.0, 2.0, 1.0, 0.0]),
        {"samples": torch.zeros(1, 4, 2, 4, 4)},
        {"two_stage_enabled": True, "two_stage_steps": 2, "two_stage_scale": 1.5},
        video_vae=ForbiddenVAE(),
    )

    assert result == (final_denoised, final_denoised)
    assert len(calls) == 2
    assert isinstance(calls[0][0], FakeRandomNoise)
    assert torch.equal(calls[0][1], torch.tensor([10.0, 7.0, 4.0, 2.0]))
    assert isinstance(calls[1][0], FakeEmptyNoise)
    assert torch.equal(calls[1][1], torch.tensor([2.0, 1.0, 0.0]))
    assert calls[1][2] is prepared
