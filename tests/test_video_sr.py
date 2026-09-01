import sys
import types

import pytest
import torch

from nodes.schema import POSTPROCESS_MODES, allowed_postprocess_modes
from nodes.smart_1080p import SMART_UPSCALE_MODEL, resolve_smart_1080p_plan
from nodes.video_sr import (
    SEEDVR2_DIT_LOADER_NODE_ID,
    SEEDVR2_UPSCALER_NODE_ID,
    SEEDVR2_VAE_LOADER_NODE_ID,
    SEEDVR2_VAE_MODEL,
    resolve_seedvr2_callables,
    resolve_seedvr2_plan,
    seedvr2_dependency_report,
)


def _seedvr2_classes():
    class FakeUpscaler:
        @classmethod
        def execute(cls, **kwargs):
            return kwargs

    class FakeDiTLoader:
        @classmethod
        def execute(cls, model, device, **kwargs):
            return types.SimpleNamespace(result=({"model": model, **kwargs},))

    class FakeVAELoader:
        @classmethod
        def execute(cls, model, device, **kwargs):
            return types.SimpleNamespace(result=({"model": model, **kwargs},))

    return {
        SEEDVR2_UPSCALER_NODE_ID: FakeUpscaler,
        SEEDVR2_DIT_LOADER_NODE_ID: FakeDiTLoader,
        SEEDVR2_VAE_LOADER_NODE_ID: FakeVAELoader,
    }


def _seedvr2_models_root(tmp_path):
    model_dir = tmp_path / "models" / "SEEDVR2"
    model_dir.mkdir(parents=True)
    (model_dir / "seedvr2_ema_3b_fp8_e4m3fn.safetensors").write_bytes(b"x")
    (model_dir / "seedvr2_ema_3b-Q4_K_M.gguf").write_bytes(b"x")
    (model_dir / SEEDVR2_VAE_MODEL).write_bytes(b"x")
    return tmp_path


def test_video_sr_is_a_public_postprocess_mode():
    assert "video_sr" in POSTPROCESS_MODES
    assert "video_sr" in allowed_postprocess_modes("smart_free_1080p")
    assert "video_sr" in allowed_postprocess_modes("quality")
    assert "video_sr" not in allowed_postprocess_modes("low_vram_two_stage")


def test_smart_plan_prefers_video_sr_when_seedvr2_ready():
    plan = resolve_smart_1080p_plan("fl2va_model", 6, 24, 20, seedvr2_ready=True)
    assert plan["postprocess_mode"] == "video_sr"
    assert plan["ai_upscale_model"] == SMART_UPSCALE_MODEL


def test_smart_plan_keeps_ai_upscale_when_seedvr2_missing():
    plan = resolve_smart_1080p_plan("fl2va_model", 6, 24, 20, seedvr2_ready=False)
    assert plan["postprocess_mode"] == "ai_upscale"
    assert plan["ai_upscale_model"] == SMART_UPSCALE_MODEL


def test_smart_low_vram_uses_gguf_video_sr_when_ready():
    plan = resolve_smart_1080p_plan("ref2va_model", 4, 8, 7, seedvr2_ready=True)
    assert plan["postprocess_mode"] == "video_sr"


def test_seedvr2_plan_scales_with_hardware():
    low = resolve_seedvr2_plan(8.0)
    assert low["dit_model"] == "seedvr2_ema_3b-Q4_K_M.gguf"
    assert low["blocks_to_swap"] == 32
    assert low["swap_io_components"] is True
    assert low["dit_offload_device"] == "cpu"
    assert low["encode_tiled"] is True
    assert low["decode_tiled"] is True
    assert low["batch_size"] == 5

    mid = resolve_seedvr2_plan(16.0)
    assert mid["dit_model"] == "seedvr2_ema_3b_fp8_e4m3fn.safetensors"
    assert mid["blocks_to_swap"] == 6
    assert mid["batch_size"] == 9

    high = resolve_seedvr2_plan(24.0)
    assert high["dit_model"] == "seedvr2_ema_3b_fp8_e4m3fn.safetensors"
    assert high["blocks_to_swap"] == 0
    assert high["dit_offload_device"] == "none"
    assert high["encode_tiled"] is False
    assert high["decode_tiled"] is False
    assert high["batch_size"] == 13


def test_seedvr2_plan_prefers_fp8_when_gguf_missing_on_low_vram():
    plan = resolve_seedvr2_plan(
        8.0,
        available_dit=["seedvr2_ema_3b_fp8_e4m3fn.safetensors"],
    )
    assert plan["dit_model"] == "seedvr2_ema_3b_fp8_e4m3fn.safetensors"


def test_seedvr2_dependency_report_ready(tmp_path):
    root = _seedvr2_models_root(tmp_path)
    report = seedvr2_dependency_report(root, node_mappings=_seedvr2_classes())
    assert report["ready"] is True
    assert report["missing"] == []
    assert report["dit_model"] == "seedvr2_ema_3b_fp8_e4m3fn.safetensors"
    assert report["vae_model"] == SEEDVR2_VAE_MODEL


def test_seedvr2_dependency_report_lists_missing_nodes(tmp_path):
    root = _seedvr2_models_root(tmp_path)
    report = seedvr2_dependency_report(root, node_mappings={})
    assert report["ready"] is False
    assert SEEDVR2_UPSCALER_NODE_ID in report["missing"]


def test_seedvr2_dependency_report_lists_missing_weights(tmp_path):
    (tmp_path / "models" / "SEEDVR2").mkdir(parents=True)
    report = seedvr2_dependency_report(tmp_path, node_mappings=_seedvr2_classes())
    assert report["ready"] is False
    assert any("seedvr2_ema_3b" in name for name in report["missing"])
    assert SEEDVR2_VAE_MODEL in report["missing"]


def test_resolve_seedvr2_callables_returns_none_without_nodes():
    assert resolve_seedvr2_callables({}) is None
    partial = dict(_seedvr2_classes())
    del partial[SEEDVR2_VAE_LOADER_NODE_ID]
    assert resolve_seedvr2_callables(partial) is None


def test_resolve_seedvr2_callables_resolves_three_nodes():
    callables = resolve_seedvr2_callables(_seedvr2_classes())
    assert callables is not None
    upscale, dit_loader, vae_loader = callables
    assert callable(upscale) and callable(dit_loader) and callable(vae_loader)


def test_stream_output_video_sr_routes_through_seedvr2(monkeypatch):
    import nodes.stream_output as stream_output

    calls = {}

    class FakeUpscaler:
        @classmethod
        def execute(cls, image, dit, vae, seed, **kwargs):
            calls["upscale"] = {
                "frames": int(image.shape[0]),
                "dit": dit,
                "vae": vae,
                "seed": seed,
                **kwargs,
            }
            upscaled = torch.nn.functional.interpolate(
                image.movedim(-1, 1), scale_factor=2.0, mode="bilinear", align_corners=False
            ).movedim(1, -1)
            return types.SimpleNamespace(result=(upscaled.clamp(0.0, 1.0),))

    class FakeDiTLoader:
        @classmethod
        def execute(cls, model, device, **kwargs):
            calls["dit"] = {"model": model, "device": device, **kwargs}
            return types.SimpleNamespace(result=(calls["dit"],))

    class FakeVAELoader:
        @classmethod
        def execute(cls, model, device, **kwargs):
            calls["vae"] = {"model": model, "device": device, **kwargs}
            return types.SimpleNamespace(result=(calls["vae"],))

    monkeypatch.setattr(
        stream_output,
        "resolve_seedvr2_callables",
        lambda: (FakeUpscaler.execute, FakeDiTLoader.execute, FakeVAELoader.execute),
    )
    plan = resolve_seedvr2_plan(24.0)
    images = torch.rand(6, 4, 6, 3)
    chunks = list(
        stream_output._iter_video_sr_frame_chunks(images, 12, 8, seed=42, plan=plan)
    )
    output = torch.cat(chunks, dim=0)
    assert output.shape == (6, 8, 12, 3)
    assert calls["upscale"]["resolution"] == 8
    assert calls["upscale"]["max_resolution"] == 12
    assert calls["upscale"]["batch_size"] == 13
    assert calls["upscale"]["color_correction"] == "lab"
    assert calls["dit"]["model"] == "seedvr2_ema_3b_fp8_e4m3fn.safetensors"
    assert calls["vae"]["model"] == SEEDVR2_VAE_MODEL


def test_stream_output_video_sr_requires_seedvr2_nodes(monkeypatch):
    import nodes.stream_output as stream_output

    monkeypatch.setattr(stream_output, "resolve_seedvr2_callables", lambda: None)
    images = torch.rand(2, 4, 6, 3)
    with pytest.raises(RuntimeError, match="SeedVR2"):
        list(
            stream_output._iter_video_sr_frame_chunks(
                images, 12, 8, seed=42, plan=resolve_seedvr2_plan(24.0)
            )
        )


def test_resolve_postprocess_path_accepts_video_sr():
    import nodes.stream_output as stream_output

    guide = {
        "postprocess_path": "video_sr",
        "target_width": 12,
        "target_height": 8,
    }
    assert stream_output._resolve_postprocess_path(guide, 6, 4) == "video_sr"
    same = {**guide, "target_width": 6, "target_height": 4}
    assert stream_output._resolve_postprocess_path(same, 6, 4) == "native_bypass"


def test_prepare_postprocess_releases_h3_before_video_sr(monkeypatch):
    import nodes.stream_output as stream_output

    released = []
    monkeypatch.setattr(
        stream_output, "release_sampling_models", lambda: released.append(True)
    )
    stream_output._prepare_postprocess_runtime(
        {"performance_preset": "low_vram_two_stage"}, "video_sr"
    )
    assert released == [True]
