# Low VRAM Native Generate and Streaming Final-Size Output Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the U11 Director Plus workflow automatically use a safe native H3 resolution for the low-VRAM preset and stream CPU-chunk final-size resizing directly into MP4 output, including 2K/4K targets.

**Architecture:** The Director node keeps the requested target dimensions but derives a separate native sampling size for `low_vram`; the guide carries both sizes. A custom output node receives decoded frames and streams CPU-backed, frame-chunked resize into the existing DaSiWa FFmpeg encoder, so a complete 4K frame batch is never materialized.

**Tech Stack:** ComfyUI custom nodes, PyTorch tensor resize, pytest, generated U11 workflow JSON.

---

### Task 1: Define the low-VRAM resolution contract

**Files:**
- Modify: `nodes/director.py`
- Test: `tests/test_director.py`

- [x] Add a failing test that a 12-second `1.30 MP` low-VRAM request records a safe native size no larger than `0.30 MP` and preserves the requested target size.
- [x] Implement `native_resolution_for_request()` with duration-aware native limits: 4/6/8/10/12/15 seconds map to 1.00/0.65/0.50/0.36/0.30/0.26 MP; leave other presets unchanged.
- [x] Put `native_width`, `native_height`, `target_width`, `target_height`, and `upscale_required` in the guide.
- [x] Run the director tests and confirm the new contract passes.

### Task 2: Add automatic CPU chunked final-size output

**Files:**
- Create: `nodes/stream_output.py`
- Modify: `__init__.py`
- Test: `tests/test_upscale.py`

- [x] Add failing tests for pass-through when no upscale is requested, target-size output, dtype preservation, and bounded frame chunks.
- [x] Implement `MiniMaxH3StreamingVideoCombine` using CPU `torch.nn.functional.interpolate` per small frame chunk and DaSiWa's FFmpeg encoder, without a full target-size batch.
- [x] Register the node with Chinese display names and preserve the existing output controls.
- [x] Run the upscale tests.

### Task 3: Route final frames through streaming output automatically

**Files:**
- Modify: `templates/u11_api.json`
- Modify: `tools/build_u11_workflow.py`
- Modify: `D:\ComfyUI_windows_portable-G313\ComfyUI\user\default\workflows\minimaxH3\U11-MiniMaxH3-导演台Plus-中文增强版.json`
- Test: `tests/test_workflow_tools.py`

- [x] Add a failing workflow assertion that `MiniMaxH3StreamingVideoCombine` receives `MiniMaxH3ColorGuard` frames and the Director guide.
- [x] Replace the output node implementation while preserving the existing Director Plus layout and output controls.
- [x] Run workflow construction tests and validate the JSON parses.

### Task 4: Verify and publish

- [x] Run the complete embedded-Python pytest suite, Python compilation, and `git diff --check`.
- [x] Commit the implementation and push `origin/main`.
- [x] Report the native/target resolution behavior, 3070 usage guidance, test result, and commit hash.
