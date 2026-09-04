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
    anchored_keyframe_noise,
    clone_guider_with_model,
    prepare_second_stage_guider,
    split_sigmas_at_step,
)
from nodes.two_stage_assets import resolve_two_stage_route


def test_quality_two_stage_is_high_vram_only_route():
    assert "quality_two_stage" in allowed_performance_presets("T2VA", "none")
    # U22 recipe: REF2VA quality two-stage is allowed; fish stays excluded.
    assert "quality_two_stage" in allowed_performance_presets("REF2VA", "none")
    assert "quality_two_stage" not in allowed_performance_presets("T2VA", "fish_lock")
    values = preset_values("quality_two_stage")
    assert values["steps"] == 12
    assert values["two_stage_split_step"] == 8
    assert values["two_stage_scale"] == pytest.approx(1.5)


def test_performance_node_marks_two_stage_guide():
    guide = {"mode": "T2VA", "voice_mode": "none", "performance_preset": "quality_two_stage"}
    result = MiniMaxH3PerformancePreset().apply(guide, acceleration_ready=True)
    assert result[0] == 12
    assert guide["two_stage_enabled"] is True
    assert guide["two_stage_split_step"] == 8
    assert "latent" in result[3]


def test_performance_node_marks_low_vram_two_stage_guide():
    guide = {
        "mode": "T2VA",
        "voice_mode": "none",
        "performance_preset": "low_vram_two_stage",
    }

    result = MiniMaxH3PerformancePreset().apply(guide, acceleration_ready=True)

    assert result[0] == 12
    assert guide["two_stage_enabled"] is True
    assert guide["two_stage_split_step"] == 8
    assert guide["two_stage_scale"] == pytest.approx(1.5)


def test_performance_node_keeps_budget_planned_two_stage_scale():
    """The preset node runs after the acceleration router and must not clobber
    the U22-recipe 2.0x ratio the VRAM budget plan derived."""
    guide = {
        "mode": "T2VA",
        "voice_mode": "none",
        "performance_preset": "quality_two_stage",
        "first_stage_width": 544,
        "second_stage_width": 1088,
    }

    result = MiniMaxH3PerformancePreset().apply(guide, acceleration_ready=True)

    assert result[0] == 12
    assert guide["two_stage_enabled"] is True
    assert guide["two_stage_scale"] == pytest.approx(2.0)


def test_two_stage_asset_route_allows_reference_backend():
    assert resolve_two_stage_route({
        "performance_preset": "quality_two_stage",
        "resolved_backend": "ref2va_model",
    }) == "trained_latent_ref"


def test_two_stage_asset_route_bypasses_fish_lock():
    assert resolve_two_stage_route({
        "performance_preset": "quality_two_stage",
        "resolved_backend": "ref2va_model",
        "voice_mode": "fish_lock",
    }) == "bypass"


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


def test_positive_conditioning_rebuilds_converted_conds_for_split_upscale():
    from nodes.two_stage import _positive_conditioning

    cross_attn = torch.zeros(1)
    converted = [
        {
            "cross_attn": cross_attn,
            "prompt": "keep",
            "minimax_refs": [{"kind": "audio"}],
            "model_conds": {},
            "uuid": "generated-uuid",
        }
    ]
    guider = types.SimpleNamespace(original_conds={"positive": converted})

    rebuilt = _positive_conditioning(guider)

    assert len(rebuilt) == 1
    tensor, payload = rebuilt[0]
    assert tensor is cross_attn
    assert payload == {
        "prompt": "keep",
        "minimax_refs": [{"kind": "audio"}],
        "model_conds": {},
    }


def test_positive_conditioning_passes_raw_pairs_through():
    from nodes.two_stage import _positive_conditioning

    raw = [[torch.zeros(1), {"prompt": "keep"}]]
    guider = types.SimpleNamespace(original_conds={"positive": raw})

    result = _positive_conditioning(guider)

    assert len(result) == 1
    assert result[0] is raw[0]


def test_second_stage_guider_resizes_fl_keyframes_without_mutating_first_stage_conditions():
    first_model = types.SimpleNamespace(model_options={"route": "first"})
    second_model = types.SimpleNamespace(model_options={"route": "second"})
    first_latent = torch.zeros(1, 24, 1, 44, 80)
    guider = types.SimpleNamespace(
        model_patcher=first_model,
        model_options=first_model.model_options,
        original_conds={
            "positive": [{
                "minimax_keyframes": [
                    {"resolved_frame_index": 0, "latent": first_latent},
                    {"resolved_frame_index": 119, "latent": first_latent.clone()},
                ],
            }],
        },
    )
    calls = []

    result = prepare_second_stage_guider(
        guider,
        second_model,
        target_video_shape=(1, 24, 107, 66, 120),
    )

    assert result is not guider
    assert result.model_patcher is second_model
    resized = result.original_conds["positive"][0]["minimax_keyframes"]
    assert [kf["latent"].shape for kf in resized] == [
        (1, 24, 1, 66, 120),
        (1, 24, 1, 66, 120),
    ]
    assert guider.original_conds["positive"][0]["minimax_keyframes"][0]["latent"] is first_latent


def test_second_stage_guider_keeps_ref2va_reference_geometry():
    first_model = types.SimpleNamespace(model_options={"route": "first"})
    second_model = types.SimpleNamespace(model_options={"route": "second"})
    ref_latent = torch.zeros(1, 24, 1, 32, 48)
    guider = types.SimpleNamespace(
        model_patcher=first_model,
        model_options=first_model.model_options,
        original_conds={
            "positive": [{
                "minimax_refs": [{
                    "kind": "image",
                    "latent_h": 32,
                    "latent_w": 48,
                    "latent": ref_latent,
                }],
            }],
        },
    )

    result = prepare_second_stage_guider(
        guider,
        second_model,
        target_video_shape=(1, 24, 107, 66, 120),
    )

    ref = result.original_conds["positive"][0]["minimax_refs"][0]
    assert ref["latent_h"] == 32
    assert ref["latent_w"] == 48
    assert ref["latent"] is ref_latent


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

    class EmptyNoise:
        seed = None

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
        types.SimpleNamespace(
            Noise_RandomNoise=FakeNoise,
            Noise_EmptyNoise=EmptyNoise,
            SamplerCustomAdvanced=FakeSampler,
        ),
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
    assert isinstance(calls[1][0], FakeNoise)
    assert calls[1][0].seed == 9, "默认应对齐 U22 配方：第二阶段注入真实随机噪声"
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


def test_resolve_split_upscale_callables_returns_none_without_node():
    from nodes.two_stage_assets import resolve_split_upscale_callables

    assert resolve_split_upscale_callables({}) is None
    assert resolve_split_upscale_callables({"UnrelatedNode": object()}) is None


def test_resolve_split_upscale_callables_resolves_main_node_with_optional_params():
    from nodes.two_stage_assets import resolve_split_upscale_callables

    class FakeSplitUpscale:
        @classmethod
        def execute(cls, **kwargs):
            return kwargs

    upscale, temporal, spatial = resolve_split_upscale_callables(
        {"MMH3SplitUpscale": FakeSplitUpscale}
    )

    assert callable(upscale)
    assert temporal is None
    assert spatial is None


def test_resolve_split_upscale_callables_resolves_param_nodes_when_present():
    from nodes.two_stage_assets import resolve_split_upscale_callables

    class FakeSplitUpscale:
        @classmethod
        def execute(cls, **kwargs):
            return kwargs

    class FakeTemporalParams:
        @classmethod
        def execute(cls, chunk_frames, temporal_overlap_frames, anchor_strength,
                    motion_anchor_frames, identity_anchor_frames):
            return None

    class FakeSpatialParams:
        @classmethod
        def execute(cls, tile_width, tile_height, overlap_ratio, fade_ratio,
                    min_tile_size, seam_denoise):
            return None

    upscale, temporal, spatial = resolve_split_upscale_callables({
        "MMH3SplitUpscale": FakeSplitUpscale,
        "MMH3TemporalSplitParamsV10": FakeTemporalParams,
        "MMH3SpatialSplitParamsV10": FakeSpatialParams,
    })

    assert callable(upscale)
    assert callable(temporal)
    assert callable(spatial)


def _run_two_stage_with_fakes(monkeypatch, *, split_callables, guide_extra=None):
    from comfy_extras.nodes_lt import LTXVConcatAVLatent
    import nodes.performance as performance
    import nodes.two_stage as two_stage

    original_video = {"samples": torch.ones(1, 24, 5, 4, 4)}
    original_audio = {"samples": torch.full((1, 32, 2, 20), 7.0)}
    first_denoised = _node_output(LTXVConcatAVLatent.execute(original_video, original_audio))
    events = {"sampler": [], "tiled": [], "upscale": []}

    class FakeNoise:
        def __init__(self, seed):
            self.seed = seed

    class EmptyNoise:
        seed = 0

    class FakeSampler:
        @classmethod
        def execute(cls, noise, guider, sampler, sigmas, latent):
            events["sampler"].append((noise, guider, sigmas.clone(), latent))
            if len(events["sampler"]) == 1:
                return types.SimpleNamespace(result=(first_denoised, first_denoised))
            return types.SimpleNamespace(result=(latent, latent))

    def fake_upscale(video_latent, scale):
        events["upscale"].append(scale)
        return {"samples": torch.ones(1, 24, 5, 6, 6)}

    monkeypatch.setattr(performance, "memory_policy", lambda guide: nullcontext())
    monkeypatch.setattr(two_stage, "_release_between_stages", lambda: None, raising=False)
    monkeypatch.setattr(two_stage, "run_trained_latent_upscaler", fake_upscale)
    monkeypatch.setattr(
        two_stage, "resolve_split_upscale_callables", lambda: split_callables
    )
    monkeypatch.setitem(
        sys.modules,
        "comfy_extras.nodes_custom_sampler",
        types.SimpleNamespace(
            Noise_RandomNoise=FakeNoise,
            Noise_EmptyNoise=EmptyNoise,
            SamplerCustomAdvanced=FakeSampler,
        ),
    )

    first_model = types.SimpleNamespace(model_options={"route": "first"})
    second_model = types.SimpleNamespace(model_options={"route": "second"})
    cross_attn = torch.zeros(1)
    # Real CFGGuider.original_conds entries are converted dicts (cross_attn
    # tensor inside the dict), not raw [tensor, dict] pairs.
    positive = [
        {
            "cross_attn": cross_attn,
            "prompt": "keep",
            "model_conds": {},
            "uuid": "test-uuid",
        }
    ]
    guider = types.SimpleNamespace(
        model_patcher=first_model,
        model_options=first_model.model_options,
        original_conds={"positive": positive},
    )
    sigmas = torch.tensor([10.0, 8.0, 6.0, 4.0, 2.0, 1.0, 0.5, 0.2, 0.0])
    guide = {
        "two_stage_enabled": True,
        "two_stage_split_step": 4,
        "two_stage_scale": 1.5,
        "resolved_two_stage_route": "trained_latent_fl",
    }
    guide.update(guide_extra or {})

    result = MiniMaxH3TwoStageSampler().execute(
        FakeNoise(9),
        guider,
        object(),
        sigmas,
        {"samples": torch.zeros(1, 24, 5, 4, 4)},
        guide,
        second_model=second_model,
    )
    return events, result, guide, sigmas, positive, second_model, original_audio


def test_tiled_second_stage_routes_through_split_upscale(monkeypatch):
    holder = {}

    def fake_temporal(chunk_frames, temporal_overlap_frames, anchor_strength,
                      motion_anchor_frames, identity_anchor_frames):
        return types.SimpleNamespace(result=({
            "p": (chunk_frames, temporal_overlap_frames, anchor_strength,
                  int(motion_anchor_frames), identity_anchor_frames),
        },))

    def fake_spatial(tile_width, tile_height, overlap_ratio, fade_ratio,
                     min_tile_size, seam_denoise):
        return types.SimpleNamespace(result=({
            "tw": tile_width // 16, "th": tile_height // 16,
            "ol_w": 8, "ol_h": 8, "fw": 4, "fh": 4,
            "mt": min_tile_size // 16, "cap": seam_denoise,
        }, "preview"))

    def fake_split_upscale(**kwargs):
        holder["call"] = kwargs
        samples = kwargs["latent"]["samples"]
        assert getattr(samples, "is_nested", False)
        video, audio = samples.unbind()
        assert video.shape == (1, 24, 5, 6, 6)
        assert torch.equal(audio, torch.full((1, 32, 2, 20), 7.0))
        return types.SimpleNamespace(result=(kwargs["latent"],))

    events, result, guide, sigmas, positive, second_model, original_audio = (
        _run_two_stage_with_fakes(
            monkeypatch,
            split_callables=(fake_split_upscale, fake_temporal, fake_spatial),
        )
    )

    assert len(events["sampler"]) == 1, "整帧采样器只应执行首采"
    assert "call" in holder, "应路由到 MMH3SplitUpscale 分块二采"
    call = holder["call"]
    assert call["model"] is second_model
    conditioning = call["conditioning"]
    assert len(conditioning) == 1, "分块二采只接收重建后的 positive conditioning"
    cond_tensor, cond_payload = conditioning[0]
    assert cond_tensor is positive[0]["cross_attn"]
    assert cond_payload == {"prompt": "keep", "model_conds": {}}
    assert "cross_attn" not in cond_payload and "uuid" not in cond_payload
    assert call["negative"] is None
    assert call["noise"].__class__.__name__ == "FakeNoise"
    assert torch.equal(call["sigmas"], sigmas[4:])
    assert call["cfg"] == 1.0
    assert call["temporal_split_param"] == {"p": (141, 39, 0.999, 39, 24)}
    assert call["spatial_split_param"]["cap"] == 0.65
    assert call["seam_polish"] == "auto"
    assert call["color_match"] is True
    assert result[0] is result[1]
    assert result[0] is call["latent"]
    assert guide["two_stage_second_stage_path"] == "mmh3_split_upscale"
    assert "分块" in guide["two_stage_status"]


def test_full_frame_second_stage_when_split_upscale_missing(monkeypatch):
    events, result, guide, sigmas, positive, second_model, original_audio = (
        _run_two_stage_with_fakes(monkeypatch, split_callables=None)
    )

    assert len(events["sampler"]) == 2, "缺少分块节点时第二阶段必须回退整帧采样"
    assert torch.equal(events["sampler"][1][2], sigmas[4:])
    assert guide["two_stage_second_stage_path"] == "full_frame"
    assert guide["two_stage_status"] == "训练型 3D latent 二采完成"


def test_full_frame_second_stage_when_tiled_disabled_in_guide(monkeypatch):
    def forbidden_split_upscale(**kwargs):
        raise AssertionError("guide 关闭分块时不得调用 MMH3SplitUpscale")

    events, result, guide, sigmas, positive, second_model, original_audio = (
        _run_two_stage_with_fakes(
            monkeypatch,
            split_callables=(forbidden_split_upscale, None, None),
            guide_extra={"two_stage_tiled": False},
        )
    )

    assert len(events["sampler"]) == 2
    assert guide["two_stage_second_stage_path"] == "full_frame"


def test_legacy_empty_noise_mode_still_available(monkeypatch):
    holder = {}

    def fake_split_upscale(**kwargs):
        holder["call"] = kwargs
        return types.SimpleNamespace(result=(kwargs["latent"],))

    events, result, guide, sigmas, positive, second_model, original_audio = (
        _run_two_stage_with_fakes(
            monkeypatch,
            split_callables=(fake_split_upscale, None, None),
            guide_extra={"second_stage_noise_mode": "不注入（旧行为）"},
        )
    )

    assert holder["call"]["noise"].__class__.__name__ == "EmptyNoise"
    assert guide["two_stage_second_stage_path"] == "mmh3_split_upscale"


def test_anchored_keyframe_noise_masks_first_slice_for_i2va():
    class BaseNoise:
        seed = 7

        def generate_noise(self, input_latent):
            video = torch.ones(1, 4, 3, 2, 2)
            audio = torch.ones(1, 4, 2, 2, 2)
            return torch.nested.as_nested_tensor([video, audio], layout=torch.jagged)

    latent = {
        "samples": torch.nested.as_nested_tensor(
            [torch.zeros(1, 4, 3, 2, 2), torch.zeros(1, 4, 2, 2, 2)],
            layout=torch.jagged,
        )
    }
    wrapped, which = anchored_keyframe_noise(
        BaseNoise(), {"mode": "I2VA", "first_frame": object()}
    )
    assert which == "首帧"
    out = wrapped.generate_noise(latent)
    assert out.layout == torch.jagged
    video, audio = out.unbind()
    assert torch.equal(video[:, :, 0], torch.zeros_like(video[:, :, 0]))
    assert torch.equal(video[:, :, 1:], torch.ones_like(video[:, :, 1:]))
    assert torch.equal(audio, torch.ones_like(audio))


def test_anchored_keyframe_noise_masks_both_ends_for_fl2va():
    class BaseNoise:
        def generate_noise(self, input_latent):
            return torch.nested.as_nested_tensor(
                [torch.ones(1, 4, 3, 2, 2), torch.ones(1, 4, 2, 2, 2)],
                layout=torch.jagged,
            )

    latent = {
        "samples": torch.nested.as_nested_tensor(
            [torch.zeros(1, 4, 3, 2, 2), torch.zeros(1, 4, 2, 2, 2)],
            layout=torch.jagged,
        )
    }
    wrapped, which = anchored_keyframe_noise(BaseNoise(), {"mode": "FL2VA"})
    assert which == "首帧+尾帧"
    video, _ = wrapped.generate_noise(latent).unbind()
    assert torch.equal(video[:, :, 0], torch.zeros_like(video[:, :, 0]))
    assert torch.equal(video[:, :, -1], torch.zeros_like(video[:, :, -1]))
    assert torch.equal(video[:, :, 1:-1], torch.ones_like(video[:, :, 1:-1]))


def test_anchored_keyframe_noise_passthrough_for_ref2va():
    base = object()
    wrapped, which = anchored_keyframe_noise(
        base, {"mode": "REF2VA", "first_frame": None, "last_frame": None}
    )
    assert wrapped is base
    assert which == ""
