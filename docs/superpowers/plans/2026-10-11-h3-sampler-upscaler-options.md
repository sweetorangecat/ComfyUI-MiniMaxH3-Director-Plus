# H3 Sampling and Upscaler Options Implementation Plan

> **For agentic workers:** Use inline execution and verify each task before moving on.

**Goal:** Reduce the director UI to three robust sampling choices and three explicit final upscalers while keeping a low-VRAM generation path and stable VOSR2 24 FPS output.

**Architecture:** Curate visible sampling and output choices in the schema and frontend while retaining legacy server values for existing API payloads. Keep VOSR2/SeedVR2 video paths separate from the frame-wise, tiled RealESRGAN X2 path; validate the low-VRAM route and keep RIFE limited to RTX VSR because the RIFE-before-VOSR2 path doubled work without improving source detail.

**Tech Stack:** Python/pytest, ComfyUI node schema, JavaScript director UI, Markdown docs.

---

### Task 1: Lock down the curated menus in tests

**Files:**
- Modify: `tests/test_schema.py`
- Modify: `tests/test_frontend_source.py`
- Modify: `tests/test_video_sr.py`

- [x] Assert the visible sampling menus contain three choices for standard routes and reference routes, with Fish S2 excluding its incompatible trained two-stage choice.
- [x] Assert the visible output menu is exactly VOSR2, SeedVR2, and AI upscaling for all supported sampling presets, while legacy postprocess values remain accepted where they are already supported.
- [x] Assert low-VRAM plus AI upscaling remains valid, and RIFE 2x remains available only for RTX VSR.
- [x] Run the focused tests and verify the curated menus and execution routes pass.

### Task 2: Curate schema, node defaults, and frontend controls

**Files:**
- Modify: `nodes/schema.py`
- Modify: `nodes/director.py`
- Modify: `js/minimax_h3_director_plus_v9.js`

- [x] Show stable 20-step, trained 8+4 two-stage, and low-VRAM 20-step choices; use the reference-specific stable 20-step preset on REF2VA routes.
- [x] Default new requests to stable quality instead of presenting the smart-auto preset as a fourth sampling choice; keep legacy preset keys normalizable for older API calls.
- [x] Show VOSR2, SeedVR2, and AI X2 only in the final-upscaler selector; keep old backend route values for compatibility and do not substitute a different engine when one of the three is explicitly selected.
- [x] Keep VOSR2 on the single-pass, 24 FPS route; keep RIFE 2x restricted to RTX VSR after the AutoDL run showed doubled VOSR2 work without a visible detail gain.
- [x] Run focused schema and frontend tests and verify they pass.

### Task 3: Update user guidance and verify stream execution

**Files:**
- Modify: `docs/使用说明.md`
- Review: `nodes/stream_output.py`
- Review: `nodes/video_finish.py`
- Review: `tests/test_director.py`
- Review: `tests/test_video_sr.py`

- [x] Document the three sampling/upscaler choices, the 8GB-class low-VRAM AI-X2 path, RIFE 48 FPS constraints, and that actual quality/VRAM still needs an AutoDL run.
- [x] Verify VOSR2 stays on the original 24 FPS path; do not insert RIFE before VOSR2 because it doubles the super-resolution workload.
- [x] Run the available targeted tests, Python compilation checks, and JavaScript syntax validation; report any ComfyUI/AutoDL-only checks that cannot run locally.

### Verification notes

- Available local suite: 784 passed, 3 skipped, 1 deselected. Collection-dependent ComfyUI tests could not run because this environment does not include ComfyUI modules; the deselected output-workflow test references a user-deleted example JSON.
- Focused route/schema/frontend tests, docs tests, Python compilation, JavaScript syntax validation, and `git diff --check` passed.
- AutoDL GPU memory peaks, runtime, and subjective motion/detail quality still require real GPU/video comparison.
