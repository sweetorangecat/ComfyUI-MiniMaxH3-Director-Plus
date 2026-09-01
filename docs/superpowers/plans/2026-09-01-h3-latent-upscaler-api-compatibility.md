# H3 Latent Upscaler API Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make U11 trained two-stage sampling call current and legacy `MinimaxH3LatentUpscaler3D` contracts reliably, with temporal chunking, forced VRAM release, and early structural validation.

**Architecture:** Keep `run_trained_latent_upscaler()` as the only external-node boundary. Add two focused helpers in `nodes/two_stage_assets.py`: one resolves the real callable behind ComfyUI v3's normalized wrapper, and one builds and validates version-specific keyword arguments from the callable signature. Preserve the existing AV separation and second-stage sampler behavior.

**Tech Stack:** Python 3.12, PyTorch, ComfyUI v3 node API, pytest, Git.

---

## File Structure

- Modify `nodes/two_stage_assets.py`: resolve the real upscaler implementation, adapt current/intermediate/legacy contracts, and validate output growth.
- Modify `tests/test_two_stage_assets.py`: add regression coverage for the installed current API, normalized wrappers, legacy contracts, unsupported signatures, and malformed results.
- Reference `docs/superpowers/specs/2026-09-01-h3-latent-upscaler-api-compatibility-design.md`: accepted behavior and scope.

### Task 1: Reproduce the current ComfyUI v3 contract failure

**Files:**
- Modify: `tests/test_two_stage_assets.py:165`
- Test: `tests/test_two_stage_assets.py`

- [ ] **Step 1: Add the installed current API contract without removing the existing intermediate-contract test**

```python
def test_trained_upscaler_adapter_uses_current_temporal_chunking_contract(monkeypatch):
    import types
    import torch
    import nodes.two_stage_assets as assets

    calls = []

    class CurrentUpscaler:
        @classmethod
        def execute(
            cls,
            latent,
            model_name,
            mode,
            align,
            enable_temporal_chunking,
            force_unload,
            device,
            precision,
        ):
            calls.append({
                "latent": latent,
                "model_name": model_name,
                "mode": mode,
                "align": align,
                "enable_temporal_chunking": enable_temporal_chunking,
                "force_unload": force_unload,
                "device": device,
                "precision": precision,
            })
            return types.SimpleNamespace(
                result=({"samples": torch.ones(1, 24, 2, 6, 6)},)
            )

    monkeypatch.setattr(
        assets,
        "_comfy_node_mappings",
        lambda: {"MinimaxH3LatentUpscaler3D": CurrentUpscaler},
    )
    latent = {"samples": torch.zeros(1, 24, 2, 4, 4)}

    result = assets.run_trained_latent_upscaler(latent, 1.5)

    assert result["samples"].shape == (1, 24, 2, 6, 6)
    assert calls == [{
        "latent": latent,
        "model_name": assets.LATENT_UPSCALER_MODEL,
        "mode": {"mode": "scale by multiplier", "scale": 1.5},
        "align": 32,
        "enable_temporal_chunking": True,
        "force_unload": True,
        "device": "cuda",
        "precision": "bf16",
    }]
```

- [ ] **Step 2: Run the focused test and verify the current implementation fails**

Run:

```powershell
python -m pytest tests/test_two_stage_assets.py::test_trained_upscaler_adapter_uses_current_temporal_chunking_contract -q
```

Expected: FAIL because the adapter does not supply `enable_temporal_chunking` and `force_unload`.

- [ ] **Step 3: Add a normalized-wrapper regression test**

```python
def test_trained_upscaler_adapter_calls_real_execute_behind_v3_wrapper(monkeypatch):
    import types
    import torch
    import nodes.two_stage_assets as assets

    calls = []

    class WrappedCurrentUpscaler:
        FUNCTION = "EXECUTE_NORMALIZED"

        @classmethod
        def execute(
            cls,
            latent,
            model_name,
            mode,
            align,
            enable_temporal_chunking,
            force_unload,
            device,
            precision,
        ):
            calls.append((enable_temporal_chunking, force_unload, mode))
            return types.SimpleNamespace(
                result=({"samples": torch.ones(1, 24, 2, 6, 6)},)
            )

        @classmethod
        def EXECUTE_NORMALIZED(cls, *args, **kwargs):
            raise AssertionError("adapter must not call the normalized wrapper directly")

    monkeypatch.setattr(
        assets,
        "_comfy_node_mappings",
        lambda: {"MinimaxH3LatentUpscaler3D": WrappedCurrentUpscaler},
    )

    result = assets.run_trained_latent_upscaler(
        {"samples": torch.zeros(1, 24, 2, 4, 4)}, 1.5
    )

    assert result["samples"].shape == (1, 24, 2, 6, 6)
    assert calls == [(True, True, {"mode": "scale by multiplier", "scale": 1.5})]
```

- [ ] **Step 4: Run both tests and verify the wrapper test fails**

Run:

```powershell
python -m pytest tests/test_two_stage_assets.py -k "current_temporal or real_execute" -q
```

Expected: 2 failed tests caused by missing current fields or direct normalized-wrapper invocation.

### Task 2: Implement signature-driven callable and keyword adaptation

**Files:**
- Modify: `nodes/two_stage_assets.py:122-170`
- Test: `tests/test_two_stage_assets.py`

- [ ] **Step 1: Add a real-callable resolver**

Insert after `_unwrap_node_output()`:

```python
def _resolve_upscaler_callable(node_class):
    function_name = str(getattr(node_class, "FUNCTION", "execute"))
    if function_name.startswith("EXECUTE_NORMALIZED"):
        candidate = getattr(node_class, "execute", None)
        if callable(candidate):
            parameters = inspect.signature(candidate).parameters
            if "self" not in parameters:
                return candidate

    node = node_class()
    candidate = getattr(node, function_name, None)
    if not callable(candidate):
        raise RuntimeError(
            f"H3 latent 放大节点没有可调用实现：{node_class.__name__}.{function_name}"
        )
    return candidate
```

- [ ] **Step 2: Add signature-based keyword construction and required-field validation**

```python
def _upscaler_kwargs(function, video_latent, scale):
    parameters = inspect.signature(function).parameters
    kwargs = {
        "latent": video_latent,
        "model_name": LATENT_UPSCALER_MODEL,
        "device": "cuda",
        "precision": "bf16",
    }
    if "mode" in parameters:
        kwargs["mode"] = {
            "mode": "scale by multiplier",
            "scale": float(scale),
        }
    elif "scale" in parameters:
        kwargs["scale"] = float(scale)

    optional_values = {
        "align": 32,
        "enable_temporal_chunking": True,
        "force_unload": True,
        "enable_chunking": True,
        "keep_proportion": False,
    }
    kwargs.update({
        name: value
        for name, value in optional_values.items()
        if name in parameters
    })
    kwargs = {name: value for name, value in kwargs.items() if name in parameters}

    missing = [
        name
        for name, parameter in parameters.items()
        if name not in {"self", "cls"}
        and parameter.kind
        not in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
        and parameter.default is inspect.Parameter.empty
        and name not in kwargs
    ]
    if "mode" not in parameters and "scale" not in parameters:
        missing.append("mode|scale")
    if missing:
        raise RuntimeError(
            "不支持当前 H3 latent 放大节点接口，缺少参数适配："
            + ", ".join(dict.fromkeys(missing))
        )
    return kwargs
```

- [ ] **Step 3: Route `run_trained_latent_upscaler()` through both helpers**

Replace direct `FUNCTION` resolution and invocation with:

```python
    node_class = mappings[node_id]
    function = _resolve_upscaler_callable(node_class)
    kwargs = _upscaler_kwargs(function, video_latent, scale)
    result = _unwrap_node_output(function(**kwargs))
```

Keep the existing result tensor and channel validation after this call.

- [ ] **Step 4: Run current and legacy adapter tests**

Run:

```powershell
python -m pytest tests/test_two_stage_assets.py -k "trained_upscaler_adapter" -q
```

Expected: all current, normalized-wrapper, `enable_chunking`, and `keep_proportion` adapter tests PASS.

- [ ] **Step 5: Commit the API compatibility change**

```powershell
git add -- nodes/two_stage_assets.py tests/test_two_stage_assets.py
git commit -m "fix: support current H3 latent upscaler API"
```

### Task 3: Fail early for unsupported contracts and invalid spatial output

**Files:**
- Modify: `tests/test_two_stage_assets.py`
- Modify: `nodes/two_stage_assets.py:135-190`

- [ ] **Step 1: Write an unsupported-contract test**

```python
def test_trained_upscaler_adapter_rejects_unknown_required_fields(monkeypatch):
    import torch
    import nodes.two_stage_assets as assets

    class UnknownUpscaler:
        @classmethod
        def execute(cls, latent, model_name, mode, required_future_flag):
            raise AssertionError("unsupported node must not execute")

    monkeypatch.setattr(
        assets,
        "_comfy_node_mappings",
        lambda: {"MinimaxH3LatentUpscaler3D": UnknownUpscaler},
    )

    with pytest.raises(RuntimeError, match="required_future_flag"):
        assets.run_trained_latent_upscaler(
            {"samples": torch.zeros(1, 24, 2, 4, 4)}, 1.5
        )
```

- [ ] **Step 2: Write a non-growing-output test**

```python
def test_trained_upscaler_rejects_non_growing_spatial_output(monkeypatch):
    import types
    import torch
    import nodes.two_stage_assets as assets

    class BrokenUpscaler:
        @classmethod
        def execute(cls, latent, model_name, mode, device, precision):
            return types.SimpleNamespace(
                result=({"samples": torch.ones(1, 24, 2, 4, 4)},)
            )

    monkeypatch.setattr(
        assets,
        "_comfy_node_mappings",
        lambda: {"MinimaxH3LatentUpscaler3D": BrokenUpscaler},
    )

    with pytest.raises(RuntimeError, match="空间尺寸没有增长"):
        assets.run_trained_latent_upscaler(
            {"samples": torch.zeros(1, 24, 2, 4, 4)}, 1.5
        )
```

- [ ] **Step 3: Run the new validation tests and verify the spatial test fails**

Run:

```powershell
python -m pytest tests/test_two_stage_assets.py -k "unknown_required or non_growing" -q
```

Expected: unknown-contract test PASS after Task 2; non-growing-output test FAIL because spatial growth is not yet checked.

- [ ] **Step 4: Add input and output spatial validation**

Add before and after the external call:

```python
    source_samples = video_latent.get("samples") if isinstance(video_latent, dict) else None
    if not isinstance(source_samples, torch.Tensor) or source_samples.ndim not in (4, 5):
        raise RuntimeError("训练型 H3 latent 放大输入无效")

    samples = result.get("samples") if isinstance(result, dict) else None
    if samples is None or getattr(samples, "ndim", 0) not in (4, 5):
        raise RuntimeError("训练型 H3 latent 放大节点返回了无效结果")
    if int(samples.shape[1]) != 24:
        raise RuntimeError(
            f"训练型 H3 latent 放大结果通道数错误：{samples.shape[1]}，应为 24"
        )
    if float(scale) > 1.0 and (
        int(samples.shape[-2]) <= int(source_samples.shape[-2])
        or int(samples.shape[-1]) <= int(source_samples.shape[-1])
    ):
        raise RuntimeError(
            "训练型 H3 latent 放大结果空间尺寸没有增长："
            f"{tuple(source_samples.shape[-2:])} -> {tuple(samples.shape[-2:])}"
        )
```

- [ ] **Step 5: Run all upscaler asset tests**

Run:

```powershell
python -m pytest tests/test_two_stage_assets.py -q
```

Expected: all tests in `test_two_stage_assets.py` PASS.

- [ ] **Step 6: Commit validation behavior**

```powershell
git add -- nodes/two_stage_assets.py tests/test_two_stage_assets.py
git commit -m "test: validate H3 latent upscaler contracts"
```

### Task 4: Run regression verification and push main

**Files:**
- Verify: `nodes/two_stage_assets.py`
- Verify: `tests/test_two_stage_assets.py`
- Verify: complete repository test suite

- [ ] **Step 1: Run focused two-stage and workflow tests**

Run:

```powershell
python -m pytest tests/test_two_stage_assets.py tests/test_two_stage.py tests/test_performance.py tests/test_workflow_tools.py -q
```

Expected: all focused tests PASS with zero failures.

- [ ] **Step 2: Run the complete plugin test suite**

Run:

```powershell
python -m pytest -q
```

Expected: complete suite PASS with zero failures. Existing non-fatal warnings may remain and must be reported exactly.

- [ ] **Step 3: Verify clean diff and commit history**

Run:

```powershell
git status --short
git log -3 --oneline
```

Expected: working tree is clean and the compatibility commits are at the top of `main`.

- [ ] **Step 4: Push the requested main branch**

Run:

```powershell
git push origin main
```

Expected: remote `main` advances to the final compatibility commit.

- [ ] **Step 5: Record verification evidence for handoff**

Report the final commit ID, focused test count, full-suite test count, warning count, and confirmation that current API calls use both temporal chunking and forced unload.
