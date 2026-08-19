import types

import torch

from nodes.vae_decode import MiniMaxH3SafeVAEDecode, should_use_gpu_output


def test_safe_decoder_keeps_small_output_on_gpu_when_budget_allows():
    assert should_use_gpu_output(
        torch.device("cuda"), (1, 3, 48, 1088, 1920), free_memory=12 * 1024**3
    ) is True


def test_safe_decoder_uses_cpu_buffer_for_long_high_resolution_output():
    assert should_use_gpu_output(
        torch.device("cuda"), (1, 3, 362, 1440, 2560), free_memory=12 * 1024**3
    ) is False


def test_safe_decoder_keeps_15_second_2k_output_on_gpu_with_32gb_headroom():
    assert should_use_gpu_output(
        torch.device("cuda"), (1, 3, 362, 1440, 2560), free_memory=26 * 1024**3
    ) is True


def test_safe_video_vae_decode_uses_cpu_fp16_output_and_restores_vae(monkeypatch):
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
