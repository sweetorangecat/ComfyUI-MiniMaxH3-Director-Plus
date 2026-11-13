import sys
import types

import pytest
import torch

from nodes.vae_decode import MiniMaxH3SafeVAEDecode, should_use_gpu_output


@pytest.fixture
def model_residency(monkeypatch):
    events = []
    management = types.ModuleType("comfy.model_management")
    management.unload_all_models = lambda: events.append("unload")
    management.soft_empty_cache = lambda: events.append("empty_cache")
    comfy = types.ModuleType("comfy")
    comfy.model_management = management
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", management)
    return events, management


def test_safe_decoder_keeps_video_output_buffer_on_cpu_even_when_budget_allows():
    assert should_use_gpu_output(
        torch.device("cuda"), (1, 3, 48, 1088, 1920), free_memory=12 * 1024**3
    ) is False


def test_safe_decoder_uses_cpu_buffer_for_long_high_resolution_output():
    assert should_use_gpu_output(
        torch.device("cuda"), (1, 3, 362, 1440, 2560), free_memory=12 * 1024**3
    ) is False


def test_safe_decoder_keeps_15_second_2k_output_buffer_on_cpu_with_32gb_headroom():
    assert should_use_gpu_output(
        torch.device("cuda"), (1, 3, 362, 1440, 2560), free_memory=26 * 1024**3
    ) is False


def test_safe_video_vae_decode_uses_cpu_fp16_output_and_restores_vae(model_residency):
    calls = []

    class FakeVAE:
        output_device = torch.device("cuda")
        device = torch.device("cuda")

        def vae_output_dtype(self):
            return torch.float32

        def decode(self, samples):
            calls.append((self.output_device, self.vae_output_dtype()))
            return torch.zeros(1, 2, 4, 4, 3, dtype=self.vae_output_dtype())

    vae = FakeVAE()
    result = MiniMaxH3SafeVAEDecode().decode(vae, {"samples": torch.zeros(1, 4, 2, 1, 1)})[0]

    assert calls == [(torch.device("cpu"), torch.float16)]
    assert result.shape == (2, 4, 4, 3)
    assert result.device.type == "cpu"
    assert result.dtype == torch.float16
    assert vae.output_device == torch.device("cuda")
    assert vae.vae_output_dtype() == torch.float32


def test_safe_decoder_releases_completed_models_before_vae_work_and_preserves_frames(model_residency):
    events, _ = model_residency
    expected = torch.linspace(0.0, 1.0, 2 * 4 * 4 * 3, dtype=torch.float16).reshape(1, 2, 4, 4, 3)

    class FakeVAE:
        output_device = torch.device("cuda")
        device = torch.device("cuda")

        def __init__(self):
            self.first_stage_model = types.SimpleNamespace(decode_output_shape=self.decode_output_shape)

        def decode_output_shape(self, latent_shape):
            events.append("estimate")
            return (1, 3, 2, 4, 4)

        def vae_output_dtype(self):
            return torch.float32

        def decode(self, samples):
            events.append("decode")
            return expected.clone()

    result = MiniMaxH3SafeVAEDecode().decode(FakeVAE(), {"samples": torch.zeros(1, 4, 2, 1, 1)})[0]

    assert events == ["unload", "empty_cache", "estimate", "decode"]
    assert torch.equal(result, expected.reshape(2, 4, 4, 3))


@pytest.mark.parametrize("failure_stage", ["unload", "empty_cache", "decode"])
def test_safe_decoder_preserves_vae_hooks_when_cleanup_or_decode_fails(model_residency, failure_stage):
    events, management = model_residency

    def fail():
        events.append(failure_stage)
        raise RuntimeError(f"{failure_stage} failed")

    class FakeVAE:
        output_device = torch.device("cuda")

        def vae_output_dtype(self):
            return torch.float32

        def decode(self, samples):
            if failure_stage == "decode":
                assert self.output_device == torch.device("cpu")
                assert self.vae_output_dtype() == torch.float16
                fail()
            return torch.zeros(1, 2, 4, 4, 3)

    if failure_stage != "decode":
        setattr(management, "unload_all_models" if failure_stage == "unload" else "soft_empty_cache", fail)
    vae = FakeVAE()
    original_output_device = vae.output_device
    original_dtype_method = vae.vae_output_dtype

    with pytest.raises(RuntimeError, match=f"{failure_stage} failed"):
        MiniMaxH3SafeVAEDecode().decode(vae, {"samples": torch.zeros(1, 4, 2, 1, 1)})

    assert vae.output_device == original_output_device
    assert vae.vae_output_dtype == original_dtype_method
    expected_events = ["unload", "empty_cache", "decode"]
    assert events == expected_events[:expected_events.index(failure_stage) + 1]


def test_unsupported_vae_is_rejected_before_releasing_models(model_residency):
    events, _ = model_residency

    with pytest.raises(RuntimeError, match="缺少 vae_output_dtype"):
        MiniMaxH3SafeVAEDecode().decode(object(), {"samples": torch.zeros(1, 4, 2, 1, 1)})

    assert events == []
