"""Exact MiniMax H3 attention with a reused full-sequence output buffer.

The enlarged second pass can have more than 200k packed tokens. Allocating an
additional ``[tokens, hidden]`` attention output at that size costs roughly
2.81 GiB for the reported 15-second 2K job. This patch releases the normalized
input immediately, writes independent head groups into the consumed Q region,
chunks the final output projection, and runs the row-independent MLP over
bounded token groups instead of materializing its full 11+ GiB intermediate.
"""

from __future__ import annotations

import types

import comfy.model_management as mm
import comfy.ops
import comfy.quant_ops
from comfy.ldm.minimax.model import _mod_gate, _mod_scale_shift
from comfy.ldm.modules.attention import optimized_attention


def minimax_attn_reuse_forward(self, x, rope_freqs=None, transformer_options=None):
    """Run exact head-chunked attention while reusing the normalized input.

    ``x`` is deliberately passed as a one-item list by the patched block. Once
    Q/K/V have been projected, the normalized input is released. The Q region
    of the fused QKV allocation becomes the attention/output-projection buffer
    after each corresponding head group has been consumed.
    """
    transformer_options = transformer_options or {}
    if not isinstance(x, list) or len(x) != 1:
        raise RuntimeError("MiniMax H3 复用注意力需要由配套 block 补丁调用")

    normalized = x.pop()
    sequence, hidden = normalized.shape
    inner = self.heads * self.head_dim

    device = normalized.device
    qkv = self.qkv_proj(normalized)
    del normalized
    q, k, v = qkv.split(inner, dim=-1)
    v = v.view(sequence, self.heads, self.head_dim)

    if rope_freqs is not None:
        q = q.view(1, sequence, self.heads, self.head_dim)
        k = k.view(1, sequence, self.heads, self.head_dim)
        qw = mm.cast_to(self.q_norm.weight, device=device)
        kw = mm.cast_to(self.k_norm.weight, device=device)
        rotation_dimension = rope_freqs.shape[-3] * 2
        if mm.in_training:
            q, k = comfy.quant_ops.ck.rms_rope_split_half(
                q,
                k,
                rope_freqs,
                qw,
                kw,
                epsilon=self.q_norm.eps,
                rot_dim=rotation_dimension,
            )
        else:
            comfy.quant_ops.ck.rms_rope_split_half_(
                q,
                k,
                rope_freqs,
                qw,
                kw,
                epsilon=self.q_norm.eps,
                rot_dim=rotation_dimension,
            )
        q = q[0]
        k = k[0]
    else:
        q = self.q_norm(q.view(sequence, self.heads, self.head_dim))
        k = self.k_norm(k.view(sequence, self.heads, self.head_dim))

    q = q.transpose(0, 1).unsqueeze(0)
    k = k.transpose(0, 1).unsqueeze(0)
    v = v.transpose(0, 1).unsqueeze(0)
    group_count = min(
        max(2, int(transformer_options.get("minimax_head_chunks", 8))),
        self.heads,
    )
    group_sizes = [
        self.heads // group_count + (1 if index < self.heads % group_count else 0)
        for index in range(group_count)
    ]
    attention_buffer = qkv[:, :inner]

    head_start = 0
    for group_size in group_sizes:
        head_end = head_start + group_size
        chunk = optimized_attention(
            q[:, head_start:head_end],
            k[:, head_start:head_end],
            v[:, head_start:head_end],
            group_size,
            mask=None,
            skip_reshape=True,
            transformer_options=transformer_options,
        ).squeeze(0)
        column_start = head_start * self.head_dim
        column_end = head_end * self.head_dim
        attention_buffer[:, column_start:column_end].copy_(chunk)
        del chunk
        head_start = head_end

    del q, k, v
    projection_chunks = min(
        max(1, int(transformer_options.get("minimax_projection_chunks", 16))),
        sequence,
    )
    rows_per_chunk = (sequence + projection_chunks - 1) // projection_chunks
    for row_start in range(0, sequence, rows_per_chunk):
        row_end = min(sequence, row_start + rows_per_chunk)
        projected = self.out_proj(attention_buffer[row_start:row_end])
        attention_buffer[row_start:row_end, :hidden].copy_(projected)
        del projected
    return attention_buffer[:, :hidden]


minimax_attn_reuse_forward._uses_optimized_attention = True


def minimax_mlp_reuse_forward(self, x, token_chunks=16):
    """Run the row-independent H3 MLP in bounded chunks, reusing its input."""
    sequence = x.shape[0]
    chunk_count = min(max(1, int(token_chunks)), sequence)
    rows_per_chunk = (sequence + chunk_count - 1) // chunk_count
    for row_start in range(0, sequence, rows_per_chunk):
        row_end = min(sequence, row_start + rows_per_chunk)
        projected = comfy.ops.linear_input_act(
            self.fc2,
            self.fc1(x[row_start:row_end]),
            "swiglu",
        )
        x[row_start:row_end].copy_(projected)
        del projected
    return x


def minimax_block_reuse_forward(
    self,
    x,
    t_emb,
    mod_segments,
    rope_freqs,
    transformer_options=None,
):
    transformer_options = transformer_options or {}
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(t_emb)
    normalized = [_mod_scale_shift(self.norm1(x), shift_msa, scale_msa, mod_segments)]
    attention = self.attn(
        normalized,
        rope_freqs=rope_freqs,
        transformer_options=transformer_options,
    )
    x = _mod_gate(x, gate_msa, attention, mod_segments)
    del attention
    normalized = _mod_scale_shift(self.norm2(x), shift_mlp, scale_mlp, mod_segments)
    return _mod_gate(x, gate_mlp, self.mlp(normalized), mod_segments)


def apply_h3_reuse_attention(model, head_chunks=8):
    """Clone and patch a verified MiniMax H3 model; never report a silent no-op."""
    if int(head_chunks) < 2:
        raise ValueError("MiniMax H3 复用注意力至少需要 2 个 head 分组")

    patched = model.clone()
    diffusion_model = patched.get_model_object("diffusion_model")
    blocks = getattr(diffusion_model, "blocks", None)
    if (
        not blocks
        or not hasattr(blocks[0], "attn")
        or not hasattr(blocks[0].attn, "qkv_proj")
    ):
        raise RuntimeError("当前模型不是可识别的 MiniMax H3 blocks/attention 结构")

    transformer_options = patched.model_options.setdefault("transformer_options", {})
    transformer_options["minimax_head_chunks"] = int(head_chunks)
    transformer_options["minimax_projection_chunks"] = 16
    transformer_options["sol_take_forward"] = minimax_attn_reuse_forward

    for index, block in enumerate(blocks):
        if not hasattr(block, "attn") or not hasattr(block.attn, "qkv_proj"):
            raise RuntimeError(f"MiniMax H3 第 {index} 个 block 缺少 attention 结构")
        if (
            not hasattr(block, "mlp")
            or not hasattr(block.mlp, "fc1")
            or not hasattr(block.mlp, "fc2")
        ):
            raise RuntimeError(f"MiniMax H3 第 {index} 个 block 缺少 MLP 结构")
        patched.add_object_patch(
            f"diffusion_model.blocks.{index}.forward",
            types.MethodType(minimax_block_reuse_forward, block),
        )
        patched.add_object_patch(
            f"diffusion_model.blocks.{index}.attn.forward",
            types.MethodType(minimax_attn_reuse_forward, block.attn),
        )
        patched.add_object_patch(
            f"diffusion_model.blocks.{index}.mlp.forward",
            types.MethodType(minimax_mlp_reuse_forward, block.mlp),
        )

    return patched
