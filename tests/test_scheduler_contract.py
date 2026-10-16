import sys
import types

import pytest
import torch

from nodes.performance import MiniMaxH3MemoryAwareSampler, MiniMaxH3SchedulerRouter, validate_scheduler_model
from nodes.two_stage import MiniMaxH3TwoStageSampler


class Model:
    def __init__(self, video_shift, audio_shift=5.0):
        self.sampling = types.SimpleNamespace(shift=video_shift, audio_shift=audio_shift)

    def get_model_object(self, name):
        assert name == "model_sampling"
        return self.sampling


def router_schedule(monkeypatch, model, guide=None):
    class Scheduler:
        @staticmethod
        def execute(model, scheduler, steps, denoise):
            return (torch.tensor([1.0, 0.5, 0.0]),)
    monkeypatch.setitem(sys.modules, "comfy_extras.nodes_custom_sampler", types.SimpleNamespace(BasicScheduler=Scheduler))
    return MiniMaxH3SchedulerRouter().route(model, 2, guide if guide is not None else {"performance_preset": "low_vram"})[0]


@pytest.mark.parametrize("sampler_type", [MiniMaxH3TwoStageSampler, MiniMaxH3MemoryAwareSampler])
def test_mismatched_video_shift_stops_before_native_sampling(monkeypatch, sampler_type):
    sigmas = router_schedule(monkeypatch, Model(12.0, 3.0))
    class NativeSampler:
        @staticmethod
        def execute(*args, **kwargs):
            pytest.fail("Mismatched sampling must not start")
    monkeypatch.setitem(sys.modules, "comfy_extras.nodes_custom_sampler", types.SimpleNamespace(SamplerCustomAdvanced=NativeSampler))
    with pytest.raises(ValueError, match="video shift=12.*video shift=10.*尚未开始"):
        sampler_type().execute(None, types.SimpleNamespace(model_patcher=Model(10.0)), None, sigmas, {}, guide={"performance_preset": "low_vram"})


@pytest.mark.parametrize("sampler_type", [MiniMaxH3TwoStageSampler, MiniMaxH3MemoryAwareSampler])
def test_matched_video_shift_preserves_native_sampling(monkeypatch, sampler_type):
    model = Model(10.0)
    sigmas = router_schedule(monkeypatch, model)
    captured = []
    class NativeSampler:
        @staticmethod
        def execute(*args):
            captured.append(args)
            return ("sampled", "denoised")
    monkeypatch.setitem(sys.modules, "comfy_extras.nodes_custom_sampler", types.SimpleNamespace(SamplerCustomAdvanced=NativeSampler))
    result = sampler_type().execute("noise", types.SimpleNamespace(model_patcher=model), "sampler", sigmas, {"samples": "latent"}, guide={"performance_preset": "low_vram"})
    assert result == ("sampled", "denoised")
    assert len(captured) == 1 and captured[0][3] is sigmas


def test_schedule_metadata_is_not_overwritten_by_another_branch(monkeypatch):
    guide = {"performance_preset": "low_vram"}
    first = router_schedule(monkeypatch, Model(10.0), guide)
    second = router_schedule(monkeypatch, Model(12.0, 3.0), guide)
    assert guide["scheduler_model_sampling"]["video_shift"] == 12.0
    validate_scheduler_model(Model(10.0), first)
    with pytest.raises(ValueError, match="video shift=12"):
        validate_scheduler_model(Model(10.0), second)


def test_audio_shift_alone_does_not_change_the_video_sigma_grid(monkeypatch):
    sigmas = router_schedule(monkeypatch, Model(10.0, 3.0))
    validate_scheduler_model(Model(10.0, 5.0), sigmas)


def test_legacy_and_explicitly_transformed_schedules_keep_existing_behavior(monkeypatch):
    validate_scheduler_model(Model(10.0), torch.tensor([1.0, 0.0]))
    sigmas = router_schedule(monkeypatch, Model(12.0))
    validate_scheduler_model(Model(10.0), sigmas.clone())
