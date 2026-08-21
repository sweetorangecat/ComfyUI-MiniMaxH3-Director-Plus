# U11 Quality Output and Audio Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve the real detail basis of 2K two-stage output, make RIFE opt-in, and make generated H3 audio audible without breaking saved U11 workflows.

**Architecture:** Keep H3 latent two-stage sampling and the streaming output architecture. Change only the director resolution policy and resolved guide defaults, then normalize AUDIO waveform at the streaming-output boundary before DaSiWa creates its temporary raw-audio file.

**Tech Stack:** Python, PyTorch, ComfyUI node schemas, JavaScript director UI, JSON workflow/API templates, pytest.

---

### Task 1: Lock the corrected resolution policy

**Files:**
- Modify: `tests/test_director.py`
- Modify: `nodes/director.py`

- [ ] Add a failing test asserting that 2K 16:9 quality-two-stage sampling selects at least 0.80 MP and leaves at most about 1.5x RTX VSR enlargement after the 1.5x latent pass.
- [ ] Run the focused director test and confirm it fails with the current 960×544 result.
- [ ] Change the two-stage target VSR scale from 1.9 to 1.5 without raising the official H3 canvas cap.
- [ ] Run the focused director tests and confirm they pass.

### Task 2: Make RIFE explicitly opt-in

**Files:**
- Modify: `tests/test_schema.py`
- Modify: `tests/test_director.py`
- Modify: `tests/test_frontend_source.py`
- Modify: `tests/test_workflow_tools.py`
- Modify: `nodes/schema.py`
- Modify: `nodes/director.py`
- Modify: `js/minimax_h3_director_plus_v9.js`
- Modify: `tools/build_u11_workflow.py`
- Modify: `templates/u11_api.json`

- [ ] Add failing tests for default `off`, legacy `auto -> off`, no RIFE preflight on the default path, and generated workflow/API defaults.
- [ ] Run the focused tests and verify the old auto-RIFE behavior is the failure.
- [ ] Keep backend acceptance of `auto`, resolve it to `off`, expose only off/RIFE in the director UI, and set workflow/API defaults to off.
- [ ] Run the focused schema, director, frontend, and workflow tests.

### Task 3: Add bounded output-audio normalization

**Files:**
- Modify: `tests/test_stream_output.py`
- Modify: `tests/test_schema.py`
- Modify: `tests/test_director.py`
- Modify: `nodes/stream_output.py`
- Modify: `nodes/schema.py`
- Modify: `nodes/director.py`
- Modify: `js/minimax_h3_director_plus_v9.js`
- Modify: `tools/build_u11_workflow.py`
- Modify: `templates/u11_api.json`

- [ ] Add failing tests showing that auto mode raises a -31 dBFS peak to -1.5 dBFS without more than 30 dB gain, attenuates peaks above -1.5 dBFS, preserves silence, and returns the original AUDIO value in original mode.
- [ ] Add a failing combine test proving normalized AUDIO, not the unmodified value, reaches DaSiWa `_audio_file`.
- [ ] Implement `_normalize_output_audio(audio, mode)` with PyTorch scalar peak calculation, a -1.5 dBFS target, a 30 dB maximum boost, and no processing when audio is absent or mode is original.
- [ ] Add `audio_loudness` validation, guide propagation, Chinese UI control, API field, and append-only workflow widget default.
- [ ] Run the focused audio, schema, director, frontend, and workflow tests.

### Task 4: Rebuild, verify, review, and publish

**Files:**
- Modify: `docs/API说明.md`
- Modify: `docs/使用说明.md`
- Regenerate: `templates/u11_api.json`
- Regenerate external workflow: `ComfyUI/user/default/workflows/minimaxH3/U11-MiniMaxH3-导演台Plus-中文增强版.json`

- [ ] Update the Chinese docs with the new real-resolution chain, RIFE opt-in behavior, and audio modes.
- [ ] Rebuild U11 from the existing U10 source and validate 25 nodes, 26 links, one subgraph, and no visible overlaps.
- [ ] Verify timeline remains widget index 18, motion remains the existing appended field, and audio_loudness is appended after it.
- [ ] Run `node --check`, Python compileall, the full pytest suite, and `git diff --check`.
- [ ] Review the final diff for workflow compatibility and resource regressions.
- [ ] Commit the coherent fix and push `origin main`; confirm local HEAD equals `origin/main` and the worktree is clean.
