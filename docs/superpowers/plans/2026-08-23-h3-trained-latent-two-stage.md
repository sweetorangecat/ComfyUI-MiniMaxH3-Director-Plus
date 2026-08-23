# H3 Trained Latent Two-Stage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace U11's untrained 6+2 bilinear latent path with route-locked U16/U17-style trained 3D latent two-stage sampling, early dependency/VRAM validation, and honest 2K/4K final-output reporting.

**Architecture:** Add focused modules for trained-upscaler dependencies and VRAM planning, then extend the existing model/scheduler/sampler routers without creating duplicate workflow branches. FL requests use matched 8-step/4-step FL LoRAs with beta/Euler 4+4; Reference requests use the Ref2VA LoRA with the U16 tail sigma refiner, producing a 4-step first pass and a 5-step refined tail. The AV sampler preserves pass-one audio, upscales only the clean video latent through the external trained 3D node, and streams a single final MP4 after releasing H3 before RTX VSR.

**Tech Stack:** Python 3.12, PyTorch, ComfyUI node APIs, MiniMax H3 AV NestedTensor, pytest, JavaScript Director UI, JSON workflow builder, FFmpeg/ffprobe, NVIDIA RTX VSR.

---

## File map

- Create `nodes/two_stage_assets.py`: dependency discovery, route constants, external upscaler and sigma-refiner adapters.
- Create `nodes/vram_budget.py`: pure resolution, tier, scale, and conservative workload-budget calculations.
- Modify `nodes/performance.py`: route-specific model pairs, LoRA strengths, scheduler router, acceleration status.
- Modify `nodes/two_stage.py`: delete bilinear/CONST path; preserve AV audio and invoke the trained upscaler between FL 4+4 or Reference 4+5 passes.
- Modify `nodes/director.py`: resolve first/second/final dimensions, perform preflight, and publish honest quality metadata.
- Modify `nodes/schema.py`: compatibility and low-VRAM final-target limits.
- Modify `nodes/status.py`: report exact required nodes/models/LoRAs.
- Modify `nodes/stream_output.py`: release H3 before RTX VSR and log final dimensions/audio.
- Modify `__init__.py`: register the scheduler router and update Chinese display names.
- Modify `js/minimax_h3_director_plus_v9.js`: stable Chinese route/size/safety explanation.
- Modify `tools/build_u11_workflow.py`: wire the second-stage model and scheduler automatically.
- Modify `templates/u11_api.json`: expose the same automatic route to API jobs.
- Modify `docs/使用说明.md`, `docs/API说明.md`, `docs/故障排查.md`: dependency installation and truthful 2K/4K behavior.
- Modify focused tests in `tests/` before each production change.
- Regenerate `D:/ComfyUI_windows_portable-G313/ComfyUI/user/default/workflows/minimaxH3/U11-MiniMaxH3-导演台Plus-中文增强版.json` only after node tests pass.

### Task 1: Add exact dependency contracts and capability reporting

**Files:**
- Create: `nodes/two_stage_assets.py`
- Modify: `nodes/status.py`
- Test: `tests/test_two_stage_assets.py`
- Test: `tests/test_status.py`

- [ ] **Step 1: Write failing dependency tests**

```python
from pathlib import Path

from nodes.two_stage_assets import (
    FL_STAGE1_LORA,
    FL_STAGE2_LORA,
    LATENT_UPSCALER_MODEL,
    REF_STAGE_LORA,
    dependency_report,
    resolve_two_stage_route,
)


def test_route_is_locked_to_backend_and_excludes_fish():
    assert resolve_two_stage_route({"resolved_backend": "fl2va_model", "voice_mode": "none"}) == "trained_latent_fl"
    assert resolve_two_stage_route({"resolved_backend": "ref2va_model", "voice_mode": "h3_reference"}) == "trained_latent_ref"
    assert resolve_two_stage_route({"resolved_backend": "ref2va_model", "voice_mode": "fish_lock"}) == "bypass"


def test_fl_dependency_report_requires_only_fl_assets(tmp_path: Path):
    report = dependency_report(tmp_path, "trained_latent_fl", node_mappings={})
    assert report["ready"] is False
    assert report["missing"] == [
        "MinimaxH3LatentUpscaler3D",
        LATENT_UPSCALER_MODEL,
        FL_STAGE1_LORA,
        FL_STAGE2_LORA,
    ]


def test_reference_dependency_report_requires_refiner_and_ref_lora(tmp_path: Path):
    report = dependency_report(tmp_path, "trained_latent_ref", node_mappings={})
    assert report["missing"] == [
        "MinimaxH3LatentUpscaler3D",
        "H3SigmaRefiner",
        LATENT_UPSCALER_MODEL,
        REF_STAGE_LORA,
    ]
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run:

```powershell
D:\ComfyUI_windows_portable-G313\python_embeded\python.exe -m pytest tests/test_two_stage_assets.py tests/test_status.py -q
```

Expected: collection fails because `nodes.two_stage_assets` does not exist.

- [ ] **Step 3: Implement route constants and dependency discovery**

Create `nodes/two_stage_assets.py` with these public contracts:

```python
from __future__ import annotations

from pathlib import Path


FL_STAGE1_LORA = "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"
FL_STAGE2_LORA = "minimax_h3_fl2v_turbo_4step_v1.1_768p_comfyui_bf16.safetensors"
REF_STAGE_LORA = "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"
LATENT_UPSCALER_MODEL = "minimax_h3_latent_upscaler_3d_bf16.safetensors"
UPSCALE_NODE_IDS = ("MinimaxH3LatentUpscaler3D", "MinimaxH3LatentUpscalerNode3D")
SIGMA_REFINER_NODE_ID = "H3SigmaRefiner"


def resolve_two_stage_route(guide):
    if str((guide or {}).get("voice_mode", "none")) == "fish_lock":
        return "bypass"
    if str((guide or {}).get("resolved_backend", "fl2va_model")) == "ref2va_model":
        return "trained_latent_ref"
    return "trained_latent_fl"


def dependency_report(comfy_root, route, node_mappings=None):
    root = Path(comfy_root)
    mappings = dict(node_mappings or {})
    loras = root / "models" / "loras"
    latent_models = root / "models" / "latent_upscale_models"
    missing = []
    if not any(node_id in mappings for node_id in UPSCALE_NODE_IDS):
        missing.append(UPSCALE_NODE_IDS[0])
    if route == "trained_latent_ref" and SIGMA_REFINER_NODE_ID not in mappings:
        missing.append(SIGMA_REFINER_NODE_ID)
    required = [(LATENT_UPSCALER_MODEL, latent_models)]
    if route == "trained_latent_ref":
        required.append((REF_STAGE_LORA, loras))
    else:
        required.extend(((FL_STAGE1_LORA, loras), (FL_STAGE2_LORA, loras)))
    for name, directory in required:
        if not (directory / name).is_file():
            missing.append(name)
    return {"ready": not missing, "missing": missing}
```

Update `detect_capabilities()` so `status["acceleration"]["trained_latent_two_stage"]` contains separate `fl` and `reference` reports, the two accepted upscaler node IDs, the model directory, and exact model/LoRA filenames. Import ComfyUI's `NODE_CLASS_MAPPINGS` only inside `detect_capabilities()` and fall back to `{}` during unit tests.

- [ ] **Step 4: Run tests and verify pass**

Run the Step 2 command. Expected: all selected tests pass.

- [ ] **Step 5: Commit and push**

```powershell
git add nodes/two_stage_assets.py nodes/status.py tests/test_two_stage_assets.py tests/test_status.py
git commit -m "feat: detect trained H3 two-stage assets"
git push origin main
```

### Task 2: Resolve conservative VRAM and true-detail dimensions

**Files:**
- Create: `nodes/vram_budget.py`
- Modify: `nodes/director.py`
- Modify: `nodes/schema.py`
- Test: `tests/test_vram_budget.py`
- Test: `tests/test_director.py`
- Test: `tests/test_schema.py`

- [ ] **Step 1: Write failing pure-budget tests**

```python
import pytest

from nodes.vram_budget import plan_two_stage_dimensions


def test_32gb_15s_2k_keeps_neural_basis_near_1080p():
    plan = plan_two_stage_dimensions(2560, 1440, 15, total_vram_gb=32, free_vram_gb=29)
    assert plan["allowed"] is True
    assert 0.80 <= plan["first_stage_megapixels"] <= 0.95
    assert 1.80 <= plan["second_stage_megapixels"] <= 2.15
    assert 1.25 <= plan["final_scale"] <= 1.45


def test_32gb_15s_4k_is_streaming_2x_not_native_4k_latent():
    plan = plan_two_stage_dimensions(3840, 2160, 15, total_vram_gb=32, free_vram_gb=29)
    assert plan["allowed"] is True
    assert plan["second_stage_width"] <= 1920
    assert plan["second_stage_height"] <= 1088
    assert 1.95 <= plan["final_scale"] <= 2.10


def test_8gb_long_video_rejects_two_stage_and_4k():
    plan = plan_two_stage_dimensions(3840, 2160, 15, total_vram_gb=8, free_vram_gb=7)
    assert plan["allowed"] is False
    assert plan["max_final_width"] == 1920
    assert "低显存" in plan["reason"]


def test_busy_32gb_gpu_fails_before_sampling():
    plan = plan_two_stage_dimensions(2560, 1440, 15, total_vram_gb=32, free_vram_gb=8)
    assert plan["allowed"] is False
    assert "当前可用显存" in plan["reason"]
```

- [ ] **Step 2: Run tests and confirm missing planner failure**

Run:

```powershell
D:\ComfyUI_windows_portable-G313\python_embeded\python.exe -m pytest tests/test_vram_budget.py tests/test_director.py tests/test_schema.py -q
```

Expected: collection fails on `nodes.vram_budget`.

- [ ] **Step 3: Implement the pure planner**

Create `nodes/vram_budget.py` with:

```python
from __future__ import annotations

import math


def _aligned_size(width, height, target_mp, alignment=32):
    ratio = float(width) / float(height)
    area = float(target_mp) * 1_000_000.0
    resolved_height = math.sqrt(area / ratio)
    resolved_width = resolved_height * ratio
    resolved_width = max(alignment, int(round(resolved_width / alignment)) * alignment)
    resolved_height = max(alignment, int(round(resolved_height / alignment)) * alignment)
    return resolved_width, resolved_height


def plan_two_stage_dimensions(final_width, final_height, duration, total_vram_gb, free_vram_gb):
    final_width, final_height, duration = int(final_width), int(final_height), int(duration)
    total, free = float(total_vram_gb), float(free_vram_gb)
    if total < 16:
        return {"allowed": False, "reason": "低显存档位不执行长视频训练型二采", "max_final_width": 1920, "max_final_height": 1080}
    if total < 28 and (duration > 8 or final_width > 2560 or final_height > 1440):
        return {"allowed": False, "reason": "当前显存档位只开放短视频2K训练型二采", "max_final_width": 2560, "max_final_height": 1440}
    required_free = 24.0 if duration >= 12 else 18.0
    if free < required_free:
        return {"allowed": False, "reason": f"当前可用显存 {free:.1f}GB 低于安全余量 {required_free:.1f}GB", "max_final_width": 2560, "max_final_height": 1440}
    first_mp = 0.90 if total >= 28 else 0.50
    first_width, first_height = _aligned_size(final_width, final_height, first_mp)
    second_width = max(32, int(round(first_width * 1.5 / 32)) * 32)
    second_height = max(32, int(round(first_height * 1.5 / 32)) * 32)
    final_scale = max(final_width / second_width, final_height / second_height)
    return {
        "allowed": True,
        "reason": "显存与帧数通过训练型二采安全预算",
        "first_stage_width": first_width,
        "first_stage_height": first_height,
        "second_stage_width": second_width,
        "second_stage_height": second_height,
        "first_stage_megapixels": first_width * first_height / 1_000_000.0,
        "second_stage_megapixels": second_width * second_height / 1_000_000.0,
        "final_scale": final_scale,
        "max_final_width": 3840 if total >= 28 else 2560,
        "max_final_height": 2160 if total >= 28 else 1440,
    }
```

Keep the first implementation deliberately conservative. Do not infer actual server support solely from these constants; Task 8 adjusts them only after measured server logs.

- [ ] **Step 4: Replace director's hard-coded 6+2 sizing**

Delete `_two_stage_pixel_size()` and `TWO_STAGE_MAX_VSR_SCALE`. Add a small runtime probe:

```python
def _cuda_memory_gb():
    try:
        import torch
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        return total_bytes / 2**30, free_bytes / 2**30
    except Exception:
        return 0.0, 0.0
```

When `performance_preset == "quality_two_stage"`, call `plan_two_stage_dimensions()`, reject an unsafe result with `RequestError`, and write these exact guide fields:

```python
guide.update({
    "resolved_two_stage_route": resolved_route,
    "first_stage_width": plan["first_stage_width"],
    "first_stage_height": plan["first_stage_height"],
    "second_stage_width": plan["second_stage_width"],
    "second_stage_height": plan["second_stage_height"],
    "final_upscale_scale_x": requested_width / plan["second_stage_width"],
    "final_upscale_scale_y": requested_height / plan["second_stage_height"],
    "vram_safety_tier": "28gb_plus" if total_vram_gb >= 28 else "16_24gb",
    "quality_basis": "H3 神经 latent 二采",
})
```

Change `low_vram_target_limit()` so every 8–12GB low-VRAM request is capped at 1920×1080 and 4K is rejected before model loading.

- [ ] **Step 5: Run tests and verify pass**

Run the Step 2 command. Expected: all selected tests pass.

- [ ] **Step 6: Commit and push**

```powershell
git add nodes/vram_budget.py nodes/director.py nodes/schema.py tests/test_vram_budget.py tests/test_director.py tests/test_schema.py
git commit -m "feat: plan safe H3 two-stage dimensions"
git push origin main
```

### Task 3: Build matched first/second model and scheduler routes

**Files:**
- Modify: `nodes/performance.py`
- Modify: `__init__.py`
- Test: `tests/test_performance.py`
- Test: `tests/test_two_stage_assets.py`

- [ ] **Step 1: Write failing route-contract tests**

```python
from nodes.performance import acceleration_plan, scheduler_plan


def test_fl_two_stage_uses_u17_model_lora_contract():
    guide = {"performance_preset": "quality_two_stage", "resolved_backend": "fl2va_model", "voice_mode": "none"}
    plan = acceleration_plan(guide)
    assert plan["first_lora_name"].endswith("8step_v1.0_comfyui_bf16.safetensors")
    assert plan["first_lora_strength"] == 0.75
    assert plan["second_lora_name"].endswith("4step_v1.1_768p_comfyui_bf16.safetensors")
    assert plan["second_lora_strength"] == 0.70
    assert scheduler_plan(guide) == {"scheduler": "beta", "steps": 8, "split_step": 4, "refine_reference_tail": False}


def test_reference_two_stage_uses_u16_contract():
    guide = {"performance_preset": "quality_two_stage", "resolved_backend": "ref2va_model", "voice_mode": "h3_reference"}
    plan = acceleration_plan(guide)
    assert plan["first_lora_name"].endswith("ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors")
    assert plan["second_lora_name"] == plan["first_lora_name"]
    assert scheduler_plan(guide) == {"scheduler": "simple", "steps": 8, "split_step": 4, "refine_reference_tail": True}


def test_every_trained_two_stage_route_forces_euler():
    assert sampler_name_for_guide({"performance_preset": "quality_two_stage"}, "res_multistep") == "euler"
```

- [ ] **Step 2: Run tests and verify failure on missing model-pair fields**

Run:

```powershell
D:\ComfyUI_windows_portable-G313\python_embeded\python.exe -m pytest tests/test_performance.py tests/test_two_stage_assets.py -q
```

Expected: the new assertions fail because current `quality_two_stage` has no Turbo LoRA and uses 6+2.

- [ ] **Step 3: Extend the acceleration router without breaking existing output indexes**

Change `_load_lightx2v_lora(model, lora_name, strength=1.0, low_vram=False)` and pass `strength` to both H3 Turbo and stock loaders. Extend `acceleration_plan()` with the route-locked fields from Step 1.

Append a fourth output to `MiniMaxH3AccelerationRouter`, preserving indexes 0–2:

```python
RETURN_TYPES = ("MODEL", "STRING", "BOOLEAN", "MODEL")
RETURN_NAMES = ("第一阶段模型", "加速说明", "加速成功", "第二阶段模型")
```

For `quality_two_stage`:

```python
first_model = _load_lightx2v_lora(base_model, plan["first_lora_name"], plan["first_lora_strength"])
if plan["second_lora_name"] == plan["first_lora_name"]:
    second_model = first_model
else:
    second_model = _load_lightx2v_lora(base_model, plan["second_lora_name"], plan["second_lora_strength"])
return first_model, status, True, second_model
```

For every other preset return the existing resolved model in both model slots.

Apply the existing exact MiniMax attention/MLP memory patch separately to both model patchers after their LoRAs are attached. Do not apply SolAttn, generic Sage, EasyCache or the same object patch twice to one patcher.

- [ ] **Step 4: Add `MiniMaxH3SchedulerRouter`**

Implement a node that accepts `model`, `steps`, and `guide`, calls ComfyUI `BasicScheduler` with `beta` or `simple`, and invokes `H3SigmaRefiner.refine_sigmas(sigmas, 1, 0.7, 0.0, "cosine")` only for `trained_latent_ref`. Register it in `__init__.py` as `H3 匹配调度器（自动路由）`.

Update `MiniMaxH3SamplerRouter.route()` so `quality_two_stage` always returns ComfyUI's stock `euler` sampler for both backends. Keep existing non-two-stage behavior unchanged.

- [ ] **Step 5: Run tests and verify pass**

Run the Step 2 command. Expected: all selected tests pass.

- [ ] **Step 6: Commit and push**

```powershell
git add nodes/performance.py __init__.py tests/test_performance.py tests/test_two_stage_assets.py
git commit -m "feat: route matched H3 two-stage models"
git push origin main
```

### Task 4: Replace bilinear refinement with trained 3D latent AV sampling

**Files:**
- Modify: `nodes/two_stage_assets.py`
- Modify: `nodes/two_stage.py`
- Test: `tests/test_two_stage.py`
- Test: `tests/test_two_stage_assets.py`

- [ ] **Step 1: Write failing trained-upscale and AV tests**

```python
def test_quality_two_stage_has_no_interpolation_call():
    import nodes.two_stage as two_stage
    assert not hasattr(two_stage, "upscale_video_latent")


def test_split_at_step_matches_comfy_split_sigmas():
    sigmas = torch.tensor([10.0, 8.0, 6.0, 4.0, 2.0, 1.0, 0.5, 0.2, 0.0])
    high, low = split_sigmas_at_step(sigmas, 4)
    assert torch.equal(high, sigmas[:5])
    assert torch.equal(low, sigmas[4:])


def test_trained_two_stage_preserves_audio_and_uses_fl_4_plus_4(monkeypatch):
    import sys
    import types
    from contextlib import nullcontext
    from comfy_extras.nodes_lt import LTXVConcatAVLatent, LTXVSeparateAVLatent
    import nodes.performance as performance
    import nodes.two_stage as two_stage

    original_video = {"samples": torch.ones(1, 24, 5, 4, 4)}
    original_audio = {"samples": torch.full((1, 32, 2, 20), 7.0)}
    first_denoised = _node_output(LTXVConcatAVLatent.execute(original_video, original_audio))
    calls = []

    class FakeNoise:
        def __init__(self, seed):
            self.seed = seed

    class FakeSampler:
        @classmethod
        def execute(cls, noise, guider, sampler, sigmas, latent):
            calls.append((sigmas.clone(), latent))
            if len(calls) == 1:
                return types.SimpleNamespace(result=(first_denoised, first_denoised))
            return types.SimpleNamespace(result=(latent, latent))

    def fake_upscale(video_latent, scale):
        assert scale == 1.5
        return {"samples": torch.ones(1, 24, 5, 6, 6)}

    monkeypatch.setattr(performance, "memory_policy", lambda guide: nullcontext())
    monkeypatch.setattr(two_stage, "run_trained_latent_upscaler", fake_upscale)
    monkeypatch.setitem(sys.modules, "comfy_extras.nodes_custom_sampler", types.SimpleNamespace(
        Noise_RandomNoise=FakeNoise,
        SamplerCustomAdvanced=FakeSampler,
    ))
    first_model = types.SimpleNamespace(model_options={})
    second_model = types.SimpleNamespace(model_options={})
    guider = types.SimpleNamespace(model_patcher=first_model, model_options={}, original_conds={})
    sigmas = torch.tensor([10.0, 8.0, 6.0, 4.0, 2.0, 1.0, 0.5, 0.2, 0.0])

    MiniMaxH3TwoStageSampler().execute(
        FakeNoise(9), guider, object(), sigmas, first_denoised,
        {"two_stage_enabled": True, "two_stage_split_step": 4, "two_stage_scale": 1.5},
        second_model=second_model,
    )

    assert torch.equal(calls[0][0], sigmas[:5])
    assert torch.equal(calls[1][0], sigmas[4:])
    _, second_audio = LTXVSeparateAVLatent.execute(calls[1][1])
    assert torch.equal(second_audio["samples"], original_audio["samples"])
```

Implement the AV test with the existing fake NestedTensor helpers in `tests/test_two_stage.py`; capture the argument passed to the second `SamplerCustomAdvanced.execute()` call rather than decoding video.

- [ ] **Step 2: Run focused tests and verify failure**

Run:

```powershell
D:\ComfyUI_windows_portable-G313\python_embeded\python.exe -m pytest tests/test_two_stage.py tests/test_two_stage_assets.py -q
```

Expected: failures show that current code still contains bilinear interpolation and CONST re-noising.

- [ ] **Step 3: Add the external trained-upscaler adapter**

In `nodes/two_stage_assets.py`, implement:

```python
def run_trained_latent_upscaler(video_latent, scale, node_mappings=None):
    mappings = node_mappings
    if mappings is None:
        import nodes as comfy_nodes
        mappings = comfy_nodes.NODE_CLASS_MAPPINGS
    node_id = next((name for name in UPSCALE_NODE_IDS if name in mappings), None)
    if node_id is None:
        raise RuntimeError("缺少 Minimax H3 训练型3D latent放大节点")
    node_cls = mappings[node_id]
    if node_id == "MinimaxH3LatentUpscaler3D":
        result = node_cls.execute(
            video_latent,
            LATENT_UPSCALER_MODEL,
            {"mode": "scale by multiplier", "scale": float(scale)},
            32,
            True,
            "cuda",
            "bf16",
        )
    else:
        node = node_cls()
        function = getattr(node, node_cls.FUNCTION)
        result = function(video_latent, LATENT_UPSCALER_MODEL, float(scale), "cuda", "bf16")
    value = getattr(result, "result", result)
    if isinstance(value, (tuple, list)):
        return value[0]
    return value
```

Add signature-based filtering for the legacy node so updated older releases that expose `enable_chunking` or `align` receive those fields without positional mismatch.

- [ ] **Step 4: Rewrite `MiniMaxH3TwoStageSampler.execute()`**

Add optional input `second_model` and replace the current CONST/bilinear path with:

```python
first_sigmas, second_sigmas = split_sigmas_at_step(sigmas, int(guide["two_stage_split_step"]))
first = SamplerCustomAdvanced.execute(noise, guider, sampler, first_sigmas, latent_image)
denoised = _node_output(first, 1)
video_latent, audio_latent = LTXVSeparateAVLatent.execute(denoised)
video_latent = run_trained_latent_upscaler(video_latent, 1.5)
merged = _node_output(LTXVConcatAVLatent.execute(video_latent, audio_latent), 0)
second_guider = clone_guider_with_model(guider, second_model)
second = SamplerCustomAdvanced.execute(
    Noise_RandomNoise(int(getattr(noise, "seed", 0))),
    second_guider,
    sampler,
    second_sigmas,
    merged,
)
final_denoised = _node_output(second, 1)
return final_denoised, final_denoised
```

`clone_guider_with_model()` must shallow-copy the guider and set both `model_patcher` and `model_options` from `second_model`. Delete `upscale_video_latent()`, `prepare_h3_two_stage_latent()`, `upscale_h3_guider()`, CONST inverse-noise logic, bilinear interpolation imports, and the unused `video_vae` input.

Log the resolved route, both LoRA names, sigma counts, split index, AV member shapes, trained-upscaler input/output shapes and final denoised shape. The Reference test must assert a refined 10-value sigma tensor splits to 5 values for the 4-step first pass and 6 values for the 5-step second pass.

- [ ] **Step 5: Run focused tests and verify pass**

Run the Step 2 command. Expected: all selected tests pass and no test references interpolation or VideoVAE.

- [ ] **Step 6: Commit and push**

```powershell
git add nodes/two_stage_assets.py nodes/two_stage.py tests/test_two_stage.py tests/test_two_stage_assets.py
git commit -m "feat: use trained H3 latent two-stage sampling"
git push origin main
```

### Task 5: Make the API and single U11 graph fully automatic

**Files:**
- Modify: `tools/build_u11_workflow.py`
- Modify: `templates/u11_api.json`
- Modify: `tests/test_workflow_tools.py`
- Modify: `tests/test_api.py`

- [ ] **Step 1: Add failing graph-contract tests**

Add assertions that the generated workflow and API graph:

```python
def _link_fields(link):
    if isinstance(link, dict):
        return link["origin_id"], link["origin_slot"], link["target_id"], link["target_slot"]
    return link[1], link[2], link[3], link[4]


types = [node["type"] for node in workflow["nodes"]]
assert types.count("MiniMaxH3AccelerationRouter") == 1
assert types.count("MiniMaxH3SchedulerRouter") == 1
assert types.count("MiniMaxH3TwoStageSampler") == 1
assert types.count("MiniMaxH3StreamingVideoCombine") == 1
acceleration = next(node for node in workflow["nodes"] if node["type"] == "MiniMaxH3AccelerationRouter")
two_stage = next(node for node in workflow["nodes"] if node["type"] == "MiniMaxH3TwoStageSampler")
second_model_slot = next(index for index, item in enumerate(two_stage["inputs"]) if item["name"] == "second_model")
assert any(
    origin == acceleration["id"] and origin_slot == 3 and target == two_stage["id"] and target_slot == second_model_slot
    for origin, origin_slot, target, target_slot in map(_link_fields, workflow["links"])
)
assert not set(types) & {"ImageScale", "LatentUpscale", "VAEDecodeTiled", "VAEEncode"}
```

Also assert the API graph connects scheduler-router output to the two-stage sampler and contains no `video_vae` input on that sampler.

- [ ] **Step 2: Run graph tests and verify failure**

Run:

```powershell
D:\ComfyUI_windows_portable-G313\python_embeded\python.exe -m pytest tests/test_workflow_tools.py tests/test_api.py -q
```

Expected: failures on the missing scheduler router and second-stage model link.

- [ ] **Step 3: Update builder and API template**

Replace the saved `BasicScheduler` with `MiniMaxH3SchedulerRouter`; connect model output 0, performance steps, and guide. Connect acceleration-router output 3 to `MiniMaxH3TwoStageSampler.second_model`. Remove the obsolete `video_vae` link. Preserve all node positions, group bounds, the single Director node and the single final output node.

- [ ] **Step 4: Run graph tests and verify pass**

Run the Step 2 command. Expected: all selected tests pass.

- [ ] **Step 5: Commit and push**

```powershell
git add tools/build_u11_workflow.py templates/u11_api.json tests/test_workflow_tools.py tests/test_api.py
git commit -m "feat: wire automatic trained H3 two-stage route"
git push origin main
```

### Task 6: Stabilize Director UI, API metadata, and final GPU handoff

**Files:**
- Modify: `js/minimax_h3_director_plus_v9.js`
- Modify: `nodes/schema.py`
- Modify: `nodes/stream_output.py`
- Test: `tests/test_frontend_source.py`
- Test: `tests/test_schema.py`
- Test: `tests/test_stream_output.py`

- [ ] **Step 1: Add failing UI and output tests**

```python
def test_two_stage_ui_names_trained_route_and_real_sizes():
    text = source()
    assert "训练型3D latent二采" in text
    assert "H3首采" in text
    assert "神经latent二采" in text
    assert "最终输出" in text
    assert "双线性" not in text


def test_prepare_postprocess_releases_h3_before_vsr(monkeypatch):
    from nodes import stream_output
    calls = []
    monkeypatch.setattr(stream_output, "release_sampling_models", lambda: calls.append("released"))
    stream_output._prepare_postprocess_runtime(
        {"performance_preset": "quality_two_stage"},
        "rtx_vsr",
    )
    assert calls == ["released"]
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
D:\ComfyUI_windows_portable-G313\python_embeded\python.exe -m pytest tests/test_frontend_source.py tests/test_schema.py tests/test_stream_output.py -q
```

Expected: UI still describes 6+2 bilinear and output does not explicitly release H3.

- [ ] **Step 3: Update UI and schema metadata**

Keep the Director DOM layout and responsive sizing unchanged. Replace only text/data mappings so the selected route displays:

```text
FL训练型二采：8步LoRA首采4步 + 3D latent 1.5x + 768p LoRA低sigma 4步
Reference训练型二采：Ref LoRA + Sigma尾段强化 + 3D latent 1.5x + 低sigma二采
最终2560×1440；H3首采1280×704；神经latent二采1920×1056；RTX VSR约1.34x
```

Expose `resolved_two_stage_route`, the six dimension fields, `vram_safety_tier`, `quality_basis`, and `required_assets` in `public_schema()` output descriptions, not as additional editable widgets.

- [ ] **Step 4: Release H3 before RTX VSR**

Add:

```python
def release_sampling_models():
    import comfy.model_management as model_management
    model_management.unload_all_models()
    model_management.soft_empty_cache()


def _prepare_postprocess_runtime(guide, postprocess_path):
    if guide.get("performance_preset") == "quality_two_stage" and postprocess_path == "rtx_vsr":
        release_sampling_models()
```

Call it exactly once before creating the RTX VSR effect when `performance_preset == "quality_two_stage"`. Keep VSR execution on CUDA and keep frame-at-a-time encoding; do not move VSR to CPU.

Log H3 release completion, source dimensions, final dimensions, VSR scale, encoded frame count, audio codec and output path.

- [ ] **Step 5: Run tests and verify pass**

Run the Step 2 command. Expected: all selected tests pass.

- [ ] **Step 6: Commit and push**

```powershell
git add js/minimax_h3_director_plus_v9.js nodes/schema.py nodes/stream_output.py tests/test_frontend_source.py tests/test_schema.py tests/test_stream_output.py
git commit -m "feat: expose safe H3 two-stage execution details"
git push origin main
```

### Task 7: Document installation and regenerate the only workflow

**Files:**
- Modify: `docs/使用说明.md`
- Modify: `docs/API说明.md`
- Modify: `docs/故障排查.md`
- Modify: `tools/build_u11_workflow.py`
- Regenerate: `D:/ComfyUI_windows_portable-G313/ComfyUI/user/default/workflows/minimaxH3/U11-MiniMaxH3-导演台Plus-中文增强版.json`
- Test: `tests/test_workflow_tools.py`
- Test: `tests/test_docs.py`

- [ ] **Step 1: Add failing documentation assertions**

```python
def test_docs_list_exact_trained_two_stage_dependencies():
    from pathlib import Path
    text = Path("docs/使用说明.md").read_text(encoding="utf-8")
    assert "Comfyui_Minimax_h3_latent_Upscaler" in text
    assert "minimax_h3_latent_upscaler_3d_bf16.safetensors" in text
    assert "minimax_h3_fl2v_turbo_8step_v1.0" in text
    assert "minimax_h3_fl2v_turbo_4step_v1.1_768p" in text
    assert "minimax_h3_ref2v_turbo_4step_v0.1" in text
    assert "ComfyUI-YCNodes-MiniMax-H3" in text
    assert "低显存不开放4K" in text
```

- [ ] **Step 2: Run docs/workflow tests and verify failure**

Run:

```powershell
D:\ComfyUI_windows_portable-G313\python_embeded\python.exe -m pytest tests/test_docs.py tests/test_workflow_tools.py -q
```

Expected: documentation assertions fail before the new install guide exists.

- [ ] **Step 3: Write exact installation and behavior docs**

Document the two external repositories, exact Hugging Face files, target model directories, restart requirement, status-node checks, 8GB/24GB/32GB limits, and the distinction between H3 second-stage detail basis and final MP4 pixel size. Include AutoDL download commands using `huggingface-cli download` with `--local-dir` and no Git dependency.

- [ ] **Step 4: Regenerate and structurally validate U11**

Run the repository's workflow builder. Parse the resulting JSON and assert:

- exactly one Director node;
- exactly one acceleration router, scheduler router, two-stage sampler and output node;
- no visible node rectangles overlap;
- all mode changes remain automatic;
- the workflow contains no duplicate branch or obsolete VideoVAE redraw link;
- final format is H.264 MP4 with optional AAC audio.

- [ ] **Step 5: Run docs/workflow tests and verify pass**

Run the Step 2 command. Expected: all selected tests pass.

- [ ] **Step 6: Commit and push**

```powershell
git add docs/使用说明.md docs/API说明.md docs/故障排查.md tools/build_u11_workflow.py tests/test_docs.py tests/test_workflow_tools.py D:/ComfyUI_windows_portable-G313/ComfyUI/user/default/workflows/minimaxH3/U11-MiniMaxH3-导演台Plus-中文增强版.json
git commit -m "docs: publish trained H3 two-stage workflow"
git push origin main
```

### Task 8: Full local verification and server acceptance gate

**Files:**
- Modify only if a failing test identifies a root cause; do not bundle speculative parameter changes.

- [ ] **Step 1: Run the complete local test suite**

Run:

```powershell
D:\ComfyUI_windows_portable-G313\python_embeded\python.exe -m pytest -q
```

Expected: all tests pass. Record the exact pass count and warnings.

- [ ] **Step 2: Run static workflow checks**

Run the workflow validator and JSON parser used by `tests/test_workflow_tools.py`. Expected: one workflow, one output, all links valid, no visible overlaps, and no missing registered Director Plus node type.

- [ ] **Step 3: Verify Git state and push**

```powershell
git diff --check
git status --short
git rev-parse HEAD
git rev-parse origin/main
```

Expected: `git diff --check` is empty, the worktree is clean, and local/remote hashes match.

- [ ] **Step 4: Run the server acceptance matrix without starting local ComfyUI**

The user runs these jobs on the server and returns the complete task log plus MP4:

1. RTX 3070 8GB, T2VA, low-VRAM, 4 seconds, 1080p.
2. RTX 3070 8GB, T2VA, low-VRAM, 15 seconds, 1080p.
3. RTX 4080/5090 32GB, T2VA, trained two-stage, 15 seconds, 2K.
4. RTX 5090 32GB, T2VA, trained two-stage, 15 seconds, 4K.
5. REF2VA with one reference image and one H3 voice reference, trained two-stage, 4 seconds, 2K.

For each artifact run:

```bash
ffprobe -v error -show_entries stream=index,codec_type,codec_name,width,height,r_frame_rate,nb_frames -show_entries format=duration -of json OUTPUT.mp4
```

Acceptance requires correct 2560×1440 or 3840×2160 dimensions, H.264 video, AAC audio where H3 generated audio, expected duration/frame count, no grey frames, no static noise field, no new grain burst, and no process crash.

- [ ] **Step 5: Adjust only measured safety thresholds**

If a server job is rejected or OOMs, use the logged first/second latent shapes, free VRAM and failing allocation to change only `nodes/vram_budget.py`, first adding a failing regression test for that exact GPU/duration/target combination. Do not change LoRA, sigma, interpolation, RIFE or audio in the same commit.

- [ ] **Step 6: Declare support only after evidence**

Mark 2K or 4K supported in the final handoff only for acceptance rows that produced a verified MP4 and complete log. Report the actual H3 first-stage, trained second-stage and final dimensions separately.
