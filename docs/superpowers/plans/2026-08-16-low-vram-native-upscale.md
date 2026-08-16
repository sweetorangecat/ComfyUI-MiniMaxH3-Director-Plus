# Low VRAM Native Generate and Upscale Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the U11 Director Plus workflow automatically use a safe native H3 resolution for the low-VRAM preset and CPU-chunk upscale the result to the requested target before MP4 output.

**Architecture:** The Director node keeps the requested target dimensions but derives a separate native sampling size for `low_vram`; the guide carries both sizes. A new custom node receives decoded frames and performs CPU-backed, frame-chunked resize only when the guide requests it, so the existing output node remains the sole MP4 writer.

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

### Task 2: Add the automatic CPU chunked upscale node

**Files:**
- Create: `nodes/upscale.py`
- Modify: `__init__.py`
- Test: `tests/test_upscale.py`

- [x] Add failing tests for pass-through when no upscale is requested, target-size output, dtype preservation, and bounded frame chunks.
- [x] Implement `MiniMaxH3VideoUpscale` using CPU `torch.nn.functional.interpolate` per 4-frame chunk, with `bicubic` resize and no GPU output allocation.
- [x] Register the node with Chinese display names and return an explanatory status string.
- [x] Run the upscale tests.

### Task 3: Route the generated frames through upscale automatically

**Files:**
- Modify: `templates/u11_api.json`
- Modify: `tools/build_u11_workflow.py`
- Modify: `D:\ComfyUI_windows_portable-G313\ComfyUI\user\default\workflows\minimaxH3\U11-MiniMaxH3-导演台Plus-中文增强版.json`
- Test: `tests/test_workflow_tools.py`

- [x] Add a failing workflow assertion that `MiniMaxH3VideoUpscale` is between `MiniMaxH3ColorGuard` and `DaSiWa_EnhancedVideoCombine`.
- [x] Add the node and links while preserving the existing Director Plus layout and output node.
- [x] Run workflow construction tests and validate the JSON parses.

### Task 4: Verify and publish

- [x] Run the complete embedded-Python pytest suite, Python compilation, and `git diff --check`.
- [ ] Commit the implementation and push `origin/main`.
- [ ] Report the native/target resolution behavior, 3070 usage guidance, test result, and commit hash.
