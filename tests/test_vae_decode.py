import types

import torch

from nodes.vae_decode import MiniMaxH3SafeVAEDecode


def test_safe_video_vae_decode_uses_cpu_fp16_output_and_restores_vae(monkeypatch):
    calls = []

    class FakeVAE:
        output_device = torch.device("cuda")

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
