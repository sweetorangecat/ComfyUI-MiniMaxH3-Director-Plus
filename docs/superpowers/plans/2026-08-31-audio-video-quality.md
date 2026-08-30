# H3 Audio and Video Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce generated H3 audio noise and over-sharpened 1080p texture while preserving existing route, VRAM, and output-size contracts.

**Architecture:** Add a small, deterministic audio-cleanup helper in `nodes/stream_output.py`, invoked only for `audio_loudness=auto`; keep `original` byte/object behavior unchanged. Add a route-aware conservative AI-upscale profile for the smart 1080p path, pass it through the existing streaming frame iterator, and expose actual audio/video processing metadata in the output UI report.

**Tech Stack:** Python 3.12, PyTorch tensors, existing ComfyUI/DaSiWa FFmpeg encoder, pytest.

---

### Task 1: Lock audio-cleanup behavior with tests

**Files:**
- Modify: `tests/test_stream_output.py`
- Test: existing `_normalize_output_audio` and `_combine` helpers

- [ ] **Step 1: Write failing tests** for a stable low-level noise floor, no noise injection into silence, bounded gain, and a diagnostic `audio_cleanup` marker returned by the combine UI metadata.

- [ ] **Step 2: Run the focused tests** with `D:\ComfyUI_windows_portable-G313\python_embeded\python.exe -m pytest -q tests/test_stream_output.py -k "audio_cleanup or auto_audio"` and verify failure because the helper and marker do not exist.

- [ ] **Step 3: Commit the failing tests** with `git add tests/test_stream_output.py && git commit -m "test: define clean audio output contract"`.

### Task 2: Implement bounded audio cleanup

**Files:**
- Modify: `nodes/stream_output.py`

- [ ] **Step 1: Add `_clean_output_audio(waveform)`** that converts to CPU float32, replaces non-finite samples with zero, estimates a robust noise floor from low-energy samples, attenuates only samples below the gate threshold, and applies a bounded peak normalization to -1.5 dBFS without exceeding 30 dB gain.

- [ ] **Step 2: Update `_normalize_output_audio(audio, mode)`** so `original` returns the original object, `auto` calls `_clean_output_audio`, and invalid AUDIO structures still raise the existing user-facing errors.

- [ ] **Step 3: Add `audio_cleanup` and `audio_cleanup_reason` to the combine UI metadata and log line, using `disabled`, `auto_gate_peak_limit`, or `bypass_error` explicitly.

- [ ] **Step 4: Run the focused audio tests** and verify they pass without changing existing original-mode tests.

- [ ] **Step 5: Commit the implementation** with `git add nodes/stream_output.py tests/test_stream_output.py && git commit -m "fix: reduce h3 audio noise before encoding"`.

### Task 3: Lock conservative smart-1080p reconstruction

**Files:**
- Modify: `tests/test_stream_output.py`
- Modify: `tests/test_director.py`

- [ ] **Step 1: Write failing tests** asserting that a smart 1080p guide selects a conservative AI-upscale profile, preserves exact portrait/landscape FHD dimensions, and reports the selected profile.

- [ ] **Step 2: Run the focused tests** with `D:\ComfyUI_windows_portable-G313\python_embeded\python.exe -m pytest -q tests/test_stream_output.py tests/test_director.py -k "smart or upscale_profile"` and verify failure because no profile field exists.

- [ ] **Step 3: Commit the failing tests** with `git add tests/test_stream_output.py tests/test_director.py && git commit -m "test: define conservative smart upscale contract"`.

### Task 4: Implement conservative smart-1080p profile and diagnostics

**Files:**
- Modify: `nodes/stream_output.py`
- Modify: `nodes/director.py`

- [ ] **Step 1: Add a route-aware profile resolver** returning `smart_conservative` only for normalized `smart_free_1080p`/smart-resolved guides and `standard` for all other routes.

- [ ] **Step 2: Thread the profile through `_iter_ai_upscale_frame_chunks`** and the existing ComfyUI model call, limiting enhancement/sharpening behavior through supported model-node arguments only; if the installed node exposes no enhancement control, preserve compatibility and record that the standard model path was used.

- [ ] **Step 3: Add stable guide fields** for `upscale_profile`, `audio_cleanup_requested`, and resolved first/second/final dimensions without changing route selection or low-VRAM limits.

- [ ] **Step 4: Raise the default H.264 quality ceiling for smart output only when the existing encoder supports it, while preserving explicit user codec/container/quality selections and trained-two-stage quality rules.

- [ ] **Step 5: Run focused smart-upscale and director tests** and verify exact FHD output metadata remains unchanged.

- [ ] **Step 6: Commit the implementation** with `git add nodes/stream_output.py nodes/director.py tests/test_stream_output.py tests/test_director.py && git commit -m "fix: use conservative smart 1080p reconstruction"`.

### Task 5: Full verification and remote delivery

**Files:**
- Verify: `nodes/stream_output.py`, `nodes/director.py`, `tests/test_stream_output.py`, `tests/test_director.py`

- [ ] **Step 1: Run syntax and diff checks** with `node --check js/minimax_h3_director_plus_v9.js` and `git diff --check`.

- [ ] **Step 2: Run the full suite** with `D:\ComfyUI_windows_portable-G313\python_embeded\python.exe -m pytest -q`; expected result is all tests passing with only known dependency deprecation warnings.

- [ ] **Step 3: Inspect the final diff and status**, confirming no unrelated files changed.

- [ ] **Step 4: Push `main`** with `git push origin main` and verify `git status --short --branch` reports the branch aligned with `origin/main`.
