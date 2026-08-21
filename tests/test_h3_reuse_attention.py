import types

import pytest
import torch

from nodes import h3_reuse_attention


class _IdentityNorm:
    def __call__(self, value):
        return value


class _FakeAttention:
    heads = 4
    head_dim = 2
    q_norm = _IdentityNorm()
    k_norm = _IdentityNorm()

    def __init__(self):
        self.qkv = None
        self.projection_calls = 0

    def qkv_proj(self, value):
        self.qkv = torch.cat((value, value, value), dim=-1)
        return self.qkv

    def out_proj(self, value):
        self.projection_calls += 1
        return value


def test_attention_reuses_qkv_storage_and_chunks_output_projection(monkeypatch):
    calls = []

    def fake_attention(q, k, v, heads, **kwargs):
        calls.append(heads)
        return v.transpose(1, 2).reshape(1, v.shape[2], heads * v.shape[3])

    monkeypatch.setattr(h3_reuse_attention, "optimized_attention", fake_attention)
    original = torch.arange(40, dtype=torch.float32).reshape(5, 8)
    source = [original.clone()]
    attention = _FakeAttention()

    result = h3_reuse_attention.minimax_attn_reuse_forward(
        attention,
        source,
        transformer_options={
            "minimax_head_chunks": 2,
            "minimax_projection_chunks": 2,
        },
    )

    assert source == []
    assert result.data_ptr() == attention.qkv.data_ptr()
    assert result.shape == (5, 8)
    assert calls == [2, 2]
    assert attention.projection_calls == 2


class _FakePatcher:
    def __init__(self, diffusion_model):
        self.diffusion_model = diffusion_model
        self.model_options = {"transformer_options": {}}
        self.object_patches = {}

    def clone(self):
        return _FakePatcher(self.diffusion_model)

    def get_model_object(self, name):
        assert name == "diffusion_model"
        return self.diffusion_model

    def add_object_patch(self, name, value):
        self.object_patches[name] = value


def _fake_h3_model():
    attention = types.SimpleNamespace(qkv_proj=object())
    block = types.SimpleNamespace(attn=attention)
    return types.SimpleNamespace(blocks=[block])


def test_apply_patch_records_every_required_forward_and_runtime_option():
    result = h3_reuse_attention.apply_h3_reuse_attention(
        _FakePatcher(_fake_h3_model()),
        head_chunks=8,
    )

    assert result.model_options["transformer_options"]["minimax_head_chunks"] == 8
    assert result.model_options["transformer_options"]["minimax_projection_chunks"] == 16
    assert "diffusion_model.blocks.0.forward" in result.object_patches
    assert "diffusion_model.blocks.0.attn.forward" in result.object_patches


def test_apply_patch_rejects_non_h3_model_instead_of_silent_success():
    model = _FakePatcher(types.SimpleNamespace(blocks=[]))

    with pytest.raises(RuntimeError, match="MiniMax H3"):
        h3_reuse_attention.apply_h3_reuse_attention(model, head_chunks=8)
