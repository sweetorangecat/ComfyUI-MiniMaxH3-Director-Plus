# U11 多路线最终放大 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add native, Lanczos, generic AI model, and RTX VSR final-output routes to U11 without manual node connections, while preserving exact target dimensions, audio, and low-VRAM bounded processing.

**Architecture:** The director resolves a postprocess path and validates its dependencies before H3 sampling. The streaming output node consumes that guide and processes decoded frames in bounded chunks; generic AI upscaling loads an installed ComfyUI upscale model only at output time, then releases it. The existing RTX VSR path remains isolated and retains its preflight check.

**Tech Stack:** Python, PyTorch, ComfyUI core `UpscaleModelLoader`/`ImageUpscaleWithModel`, Pillow, ComfyUI workflow JSON, vanilla JavaScript UI, pytest.

---

### Task 1: Extend schema and director contracts

**Files:**
- Modify: `nodes/schema.py`
- Modify: `nodes/director.py`
- Test: `tests/test_schema.py`
- Test: `tests/test_director.py`

- [ ] **Step 1: Write failing tests** for the four postprocess enum values, `ai_upscale_model=auto`, route selection, and native 2K/4K warning.
- [ ] **Step 2: Run the focused tests** and verify they fail because `lanczos`, `ai_upscale`, and the model input are absent.
- [ ] **Step 3: Implement the minimal schema/director changes**: add the enum, normalize the model name, expose the optional input, resolve `lanczos`/`ai_upscale`, and validate the AI model before sampling.
- [ ] **Step 4: Run `pytest tests/test_schema.py tests/test_director.py -q`** and confirm all focused tests pass.

### Task 2: Add model discovery and bounded Lanczos/AI frame iterators

**Files:**
- Modify: `nodes/stream_output.py`
- Create or modify: `nodes/upscale.py` only if the model-discovery/loader boundary cannot remain focused in the existing module
- Test: `tests/test_upscale.py`
- Test: `tests/test_stream_output.py`

- [ ] **Step 1: Write failing tests** for deterministic auto model selection, missing manual model errors, CPU Lanczos chunks, AI model output resized to the exact target, and model cleanup.
- [ ] **Step 2: Run the focused tests** and verify they fail before implementation.
- [ ] **Step 3: Implement model discovery** against `models/upscale_models`, preferring X2 models for <=2x targets and X4 models above 2x; reject missing manual paths.
- [ ] **Step 4: Implement bounded Lanczos and AI iterators** that process one frame or a small chunk, preserve order and frame count, and release the loaded model after completion.
- [ ] **Step 5: Run focused upscale/output tests** and confirm they pass without requiring a real GPU model execution.

### Task 3: Wire output-node behavior and API fields

**Files:**
- Modify: `nodes/stream_output.py`
- Modify: `tools/build_u11_workflow.py`
- Modify: `templates/u11_api.json` via the builder
- Test: `tests/test_api.py`
- Test: `tests/test_workflow_tools.py`

- [ ] **Step 1: Write failing tests** for output routing, exact target metadata, API defaults, and no new manual connections.
- [ ] **Step 2: Run the focused tests** and verify failure.
- [ ] **Step 3: Implement the output route dispatch** for `native_bypass`, `downscale`, `lanczos`, `ai_upscale`, and `rtx_vsr`, keeping audio and FPS inputs unchanged.
- [ ] **Step 4: Add `ai_upscale_model` to the API template and regenerate the single U11 JSON.**
- [ ] **Step 5: Run API/workflow tests and validate the generated workflow** with the existing validator.

### Task 4: Preserve the director UI surface

**Files:**
- Modify: `js/minimax_h3_director_plus_v9.js`
- Test: `tests/test_frontend_source.py`

- [ ] **Step 1: Write failing source tests** for four output labels, the conditional AI model selector, and dependency/error copy.
- [ ] **Step 2: Run the focused frontend tests** and verify failure.
- [ ] **Step 3: Add the controls inside the existing postprocess section** without changing the director node dimensions or layout groups.
- [ ] **Step 4: Run frontend source tests** and confirm they pass.

### Task 5: Full verification and delivery

**Files:**
- Modify: `docs/使用说明.md`, `docs/API说明.md`, or `docs/验证记录.md` only for behavior that is actually implemented

- [ ] **Step 1: Run the complete test suite:** `D:\ComfyUI_windows_portable-G313\python_embeded\python.exe -m pytest -q`.
- [ ] **Step 2: Rebuild and validate the only U11 workflow; confirm node/link counts and zero visible overlaps.**
- [ ] **Step 3: Run `git diff --check` and inspect the final diff.**
- [ ] **Step 4: Commit the implementation and regenerated workflow-related artifacts.**
- [ ] **Step 5: Push `origin/main` and verify `HEAD` equals `origin/main`.**
