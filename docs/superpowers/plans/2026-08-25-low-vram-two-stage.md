# Low-VRAM Two-Stage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a separately selectable, fail-fast trained two-stage route for an idle RTX 3070-class 8GB GPU while preserving the existing stable low-VRAM route and exposing an exact 1080p final target.

**Architecture:** Keep `low_vram` as the stable 4–15 second single-sample route. Add `low_vram_two_stage` as an explicit 4-second-only route: approximately 0.20 MP first sampling, learned 1.5x H3 latent upscale, low-sigma second sampling under ComfyUI LOW_VRAM, then RTX VSR ULTRA to a final target no larger than FHD. Reuse the existing U17/U16 trained-route assets and sampler, but give the new route its own deterministic VRAM planner profile and compatibility matrix.

**Tech Stack:** Python 3, PyTorch/ComfyUI nodes, JavaScript DOM extension, pytest, JSON workflow builder.

**Status:** Implemented and verified on 2026-08-25. The generated U11 keeps one external workflow and the original two mutually exclusive internal H3 branches; both branches use the same automatic trained two-stage contract and require no user wiring.

---

### Task 1: Public Preset and Exact FHD Target

**Files:**
- Modify: `nodes/schema.py`
- Modify: `nodes/director.py`
- Modify: `nodes/resolution.py`
- Modify: `js/minimax_h3_director_plus_v9.js`
- Test: `tests/test_schema.py`
- Test: `tests/test_director.py`
- Test: `tests/test_resolution.py`
- Test: `tests/test_frontend_source.py`

- [ ] **Step 1: Write failing tests for both raw low-VRAM labels and exact FHD output**

```python
def test_director_raw_combo_exposes_both_low_vram_presets():
    values = MiniMaxH3DirectorPlus.INPUT_TYPES()["required"]["performance_preset"][0]
    assert "低显存" in values
    assert "低显存二采" in values

def test_exact_fhd_targets_are_not_rounded_above_1080p():
    assert calculate_resolution("1080p FHD", "16:9") == (1920, 1080)
    assert calculate_resolution("1080p FHD", "9:16") == (1080, 1920)
```

- [ ] **Step 2: Run the focused tests and verify they fail because the label and target are absent**

Run: `python -m pytest tests/test_schema.py tests/test_director.py tests/test_resolution.py tests/test_frontend_source.py -q`

Expected: FAIL for the missing `低显存二采`, the existing raw combo omission of `低显存`, and missing `1080p FHD`.

- [ ] **Step 3: Add an explicit user-facing preset tuple, normalized preset, route compatibility, JS option, and exact FHD targets**

```python
USER_PERFORMANCE_PRESET_LABELS = (
    "稳定质量", "质量优先加速", "质量优先二采样", "极速4步",
    "参考图加速", "低显存", "低显存二采", "自定义",
)
```

Use this tuple in the raw ComfyUI combo instead of slicing the normalization dictionary. Add `low_vram_two_stage` to every non-Fish route and add exact `(1920, 1080)` / `(1080, 1920)` FHD output targets.

- [ ] **Step 4: Run the focused tests and verify they pass**

Run: `python -m pytest tests/test_schema.py tests/test_director.py tests/test_resolution.py tests/test_frontend_source.py -q`

Expected: PASS.

### Task 2: Deterministic 8GB Two-Stage Budget

**Files:**
- Modify: `nodes/vram_budget.py`
- Test: `tests/test_vram_budget.py`

- [ ] **Step 1: Write failing tests for the low-VRAM profile boundaries**

```python
def test_idle_8gb_allows_four_second_fhd_low_vram_two_stage():
    plan = plan_two_stage_dimensions(1920, 1080, 4, 8, 7, profile="low_vram")
    assert plan["allowed"] is True
    assert plan["vram_safety_tier"] == "8gb_low_vram_two_stage"
    assert plan["first_stage_megapixels"] <= 0.21
    assert plan["second_stage_megapixels"] <= 0.48

def test_8gb_low_vram_two_stage_rejects_longer_clips():
    plan = plan_two_stage_dimensions(1920, 1080, 5, 8, 7, profile="low_vram")
    assert plan["allowed"] is False
    assert "只支持 4 秒" in plan["reason"]
```

Also cover FHD oversize, busy/free VRAM, and ensure the existing `quality` profile still rejects 8GB.

- [ ] **Step 2: Run the budget tests and verify the new profile argument fails**

Run: `python -m pytest tests/test_vram_budget.py -q`

Expected: FAIL with the unsupported `profile` argument.

- [ ] **Step 3: Implement the low-VRAM planner branch**

Use these fail-fast limits: total VRAM at least 7.5GB, free VRAM at least 6.0GB before generation, exactly 4 seconds, final area no larger than the 1920×1080 pixel budget (plus alignment tolerance), approximately 0.20 MP first-stage grid, 1.5x learned second-stage grid, and FHD-class maximum final output.

- [ ] **Step 4: Run the budget tests and verify they pass**

Run: `python -m pytest tests/test_vram_budget.py -q`

Expected: PASS.

### Task 3: Runtime Routing, Memory Policy, and UI Isolation

**Files:**
- Modify: `nodes/performance.py`
- Modify: `nodes/director.py`
- Modify: `nodes/schema.py`
- Modify: `js/minimax_h3_director_plus_v9.js`
- Test: `tests/test_performance.py`
- Test: `tests/test_two_stage.py`
- Test: `tests/test_director.py`
- Test: `tests/test_schema.py`
- Test: `tests/test_frontend_source.py`

- [ ] **Step 1: Write failing route tests**

```python
def test_low_vram_two_stage_uses_trained_route_and_low_vram_policy():
    guide = {
        "mode": "T2VA", "voice_mode": "none",
        "performance_preset": "low_vram_two_stage",
        "resolved_backend": "fl2va_model",
    }
    values = preset_values("low_vram_two_stage")
    assert values["steps"] == 8
    assert values["minimax_head_chunks"] == 16
    assert acceleration_plan(guide)["route"] == "trained_latent_fl"
```

Add tests that the director invokes the low-VRAM budget profile, postprocessing is locked to RTX VSR, RTX quality resolves to ULTRA, RIFE is off, and Fish S2 does not expose this route.

- [ ] **Step 2: Run the focused routing tests and verify they fail for the absent preset**

Run: `python -m pytest tests/test_performance.py tests/test_two_stage.py tests/test_director.py tests/test_schema.py tests/test_frontend_source.py -q`

Expected: FAIL for the unknown preset and absent routing/UI behavior.

- [ ] **Step 3: Implement the new preset across the existing trained route**

Add an 8-step 4+4 preset with 1.5x trained latent upscale, 16 head chunks, no EasyCache, no RIFE, LOW_VRAM model staging, and the same matched FL/Reference LoRA contracts already used by `quality_two_stage`. In the director, call `plan_two_stage_dimensions(..., profile="low_vram")`, force final RTX VSR ULTRA, and fail before dependency/model loading when duration, target, or idle VRAM is unsafe. In the JS director, expose the extra preset without changing panel geometry and auto-set duration to 4 seconds when selected.

- [ ] **Step 4: Run the focused routing tests and verify they pass**

Run: `python -m pytest tests/test_performance.py tests/test_two_stage.py tests/test_director.py tests/test_schema.py tests/test_frontend_source.py -q`

Expected: PASS.

### Task 4: Workflow, Documentation, and Release Verification

**Files:**
- Modify: `docs/使用说明.md`
- Modify: `docs/API说明.md`
- Modify: `templates/u11_api.json`
- Modify: `D:/ComfyUI_windows_portable-G313/ComfyUI/user/default/workflows/minimaxH3/U11-MiniMaxH3-导演台Plus-中文增强版.json`
- Test: `tests/test_workflow_tools.py`

- [ ] **Step 1: Add workflow/API regression assertions for the new normalized preset**

Assert the generated workflow keeps one director node and one automatic output route; the original mutually exclusive H3 branches may each contain their own trained two-stage sampler, while the API schema accepts `low_vram_two_stage` without adding user wiring.

- [ ] **Step 2: Run workflow tests and verify the new assertions fail before regeneration**

Run: `python -m pytest tests/test_workflow_tools.py -q`

Expected: FAIL because generated artifacts and documentation do not yet expose the new route.

- [ ] **Step 3: Update Chinese documentation and regenerate the one U11 workflow**

Run:

```text
python tools/build_u11_workflow.py --source "D:/ComfyUI_windows_portable-G313/ComfyUI/user/default/workflows/minimaxH3/U10-DaSiWa-MiniMaxH3-MythicAlchemy-v12导演台.json" --output "D:/ComfyUI_windows_portable-G313/ComfyUI/user/default/workflows/minimaxH3/U11-MiniMaxH3-导演台Plus-中文增强版.json"
```

Document that 8GB trained two-stage means an idle GPU, exactly 4 seconds, final FHD maximum, RTX VSR dependency, much slower execution, and no claim of native 1080p generation.

- [ ] **Step 4: Run workflow tests, then the full suite**

Run: `python -m pytest tests/test_workflow_tools.py -q`

Expected: PASS.

Run: `python -m pytest -q`

Expected: all tests pass with zero failures.

- [ ] **Step 5: Inspect generated JSON and repository diff, commit, and push `main`**

Verify the single U11 JSON parses, contains no duplicate director nodes, the repository is clean after commit, and local `main`, `origin/main`, and the remote main SHA match.
