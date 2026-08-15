# 质量优先加速 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a route-safe "质量优先加速" preset that keeps 20 sampling steps, enables SageAttention, and disables Turbo LoRA/EasyCache for all H3 modes.

**Architecture:** Extend the existing preset enum and route whitelist in `nodes/schema.py` and mirror the label in the JS controller. Add a backend preset contract in `nodes/performance.py`; the acceleration router will apply only SageAttention, while failure keeps the original model and 20-step fallback. Update tests and Chinese usage/API documentation without adding nodes or changing geometry.

**Tech Stack:** Python, pytest, ComfyUI custom nodes, vanilla JavaScript controller, Markdown docs.

---

### Task 1: Lock the backend contract with failing tests

**Files:**
- Modify: `tests/test_schema.py`
- Modify: `tests/test_performance.py`

- [ ] Add assertions that `质量优先加速` normalizes to `quality_sage`, is allowed for all five modes with and without voice reference, and returns `steps=20`, `use_sage=True`, `use_cache=False`, and no Turbo LoRA.
- [ ] Run the focused schema/performance tests and confirm they fail because the new preset is absent.

### Task 2: Implement the Python preset and route behavior

**Files:**
- Modify: `nodes/schema.py`
- Modify: `nodes/performance.py`

- [ ] Add the Chinese/API alias `质量优先加速` -> `quality_sage`, include it in every route whitelist, and expose it in `public_schema().properties.performance_preset.enum` and `allowed_by_route`.
- [ ] Add `PRESETS["quality_sage"] = {"steps": 20, "use_sage": True, "use_cache": False, "interpolate": False}` and map the Chinese label in `PRESET_LABELS`.
- [ ] Keep `acceleration_plan()` from selecting either Turbo adapter or Turbo sampler for `quality_sage`; let `_apply_acceleration()` apply Sage only and preserve native sampling on failure.
- [ ] Return a Chinese description explicitly stating 20 steps, Sage enabled, Turbo/EasyCache disabled, and quality-first behavior.
- [ ] Run the focused tests and then the complete Python suite.

### Task 3: Expose the preset in the stable Director UI and docs

**Files:**
- Modify: `js/minimax_h3_director_plus_v9.js`
- Modify: `docs/使用说明.md`
- Modify: `docs/API说明.md`

- [ ] Add `质量优先加速` to the existing preset list and all route-specific option arrays; keep the existing five-column control and node dimensions unchanged.
- [ ] Document the exact runtime contract and explain that this mode does not require an additional model and falls back to native 20-step sampling if Sage is unavailable.
- [ ] Add the new enum and route examples to the API documentation.
- [ ] Run `node --check js/minimax_h3_director_plus_v9.js` and `git diff --check`.

### Task 4: Final verification and commit

**Files:**
- Verify all modified files and generated diff.

- [ ] Run the full pytest suite and JS syntax check.
- [ ] Confirm the working tree contains only the intended feature changes.
- [ ] Commit with `feat: add quality priority sage acceleration` and push `origin main`.
