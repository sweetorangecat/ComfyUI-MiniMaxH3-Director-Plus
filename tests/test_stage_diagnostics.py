import json
import sys
import types

import torch
from safetensors.torch import load_file

from nodes.stage_diagnostics import StageDiagnostics


def test_disabled_diagnostics_do_not_touch_latents_or_output(monkeypatch):
    monkeypatch.delenv("MINIMAX_H3_STAGE_DIAGNOSTICS", raising=False)
    capture = StageDiagnostics({}, 42)
    capture.save("first", object())
    assert capture.directory is None


def test_capture_preserves_video_values_and_does_not_serialize_prompt(tmp_path, monkeypatch):
    monkeypatch.setenv("MINIMAX_H3_STAGE_DIAGNOSTICS", "1")
    monkeypatch.setitem(sys.modules, "folder_paths", types.SimpleNamespace(
        get_output_directory=lambda: str(tmp_path)))
    video = torch.randn(1, 24, 3, 4, 6, dtype=torch.float16)
    before = video.clone()
    capture = StageDiagnostics({"prompt": "private prompt", "two_stage_scale": 1.5}, 42)
    capture.save("first", video)
    capture.save("upscaled", video * 2)
    first = load_file(str(capture.directory / "first.latent"))
    assert torch.equal(first["latent_tensor"], before)
    assert "latent_format_version_0" in first
    assert torch.equal(video, before)
    manifest = (capture.directory / "manifest.json").read_text("utf-8")
    assert "private prompt" not in manifest
    assert json.loads(manifest)["seed"] == 42
    assert set(json.loads(manifest)["stages"]) == {"first", "upscaled"}


def test_capture_failure_does_not_interrupt_generation(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("MINIMAX_H3_STAGE_DIAGNOSTICS", "1")
    monkeypatch.setitem(sys.modules, "folder_paths", types.SimpleNamespace(
        get_output_directory=lambda: str(tmp_path)))
    capture = StageDiagnostics({}, 0)
    import safetensors.torch
    monkeypatch.setattr(safetensors.torch, "save_file", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    capture.save("first", torch.zeros(1, 24, 1, 2, 2))
    assert "disk full" in caplog.text

