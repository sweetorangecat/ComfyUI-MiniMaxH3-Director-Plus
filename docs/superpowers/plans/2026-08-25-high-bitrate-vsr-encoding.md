# High-Bitrate VSR and Encoding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing quality-two-stage route use NVIDIA HIGHBITRATE_ULTRA, H.264 CRF 16, and lossless BT.709 metadata while preserving every other route and the Director UI layout.

**Architecture:** Request normalization owns route compatibility and legacy migration; the RTX wrapper owns SDK enum validation; the streaming output node owns route-scoped encode quality and a post-encode stream-copy metadata pass. The workflow builder remains the single source for the one U11 canvas workflow and API template.

**Tech Stack:** Python 3.12, PyTorch/ComfyUI nodes, NVIDIA nvidia-vfx, FFmpeg, JavaScript LiteGraph extension, pytest, JSON workflow builder.

---

### Task 1: Add the high-bitrate RTX quality contract

**Files:**
- Modify: `nodes/schema.py`
- Modify: `nodes/director.py`
- Modify: `nodes/rtx_vsr_stream.py`
- Test: `tests/test_schema.py`
- Test: `tests/test_director.py`
- Test: `tests/test_rtx_vsr.py`

- [ ] **Step 1: Write failing schema and RTX quality tests.**

Add assertions equivalent to:

```python
assert public_schema()["properties"]["rtx_quality"]["enum"] == [
    "HIGH", "ULTRA", "HIGHBITRATE_ULTRA"
]
request = normalize_request({
    "mode": "T2VA",
    "performance_preset": "质量优先二采样",
    "postprocess_mode": "rtx_vsr",
    "rtx_quality": "ULTRA",
})
assert request["rtx_quality_requested"] == "ULTRA"
assert request["rtx_quality"] == "HIGHBITRATE_ULTRA"
assert resolve_vsr_quality(FakeQuality, "HIGHBITRATE_ULTRA") is FakeQuality.HIGHBITRATE_ULTRA
```

- [ ] **Step 2: Run the focused tests and confirm they fail.**

Run:

```powershell
& 'D:\ComfyUI_windows_portable-G313\python_embeded\python.exe' -m pytest tests/test_schema.py tests/test_director.py tests/test_rtx_vsr.py -q
```

Expected: failures show the missing enum, migration field, and RTX quality support.

- [ ] **Step 3: Implement route-aware quality resolution.**

In `nodes/schema.py`, define:

```python
RTX_QUALITIES = ("HIGH", "ULTRA", "HIGHBITRATE_ULTRA")
RTX_QUALITIES_BY_PERFORMANCE = {
    "quality_two_stage": ("HIGHBITRATE_ULTRA",),
}

def allowed_rtx_qualities(performance_preset):
    preset = PERFORMANCE_PRESETS.get(performance_preset, performance_preset)
    return RTX_QUALITIES_BY_PERFORMANCE.get(preset, ("HIGH", "ULTRA"))
```

After performance-preset normalization, preserve `rtx_quality_requested`; migrate quality-two-stage RTX requests to `HIGHBITRATE_ULTRA`; reject `HIGHBITRATE_ULTRA` on other routes. Publish `allowed_by_performance` in the public schema. Add the third input option in `MiniMaxH3DirectorPlus.INPUT_TYPES`; keep preflight on the resolved request value and put both requested and resolved values into the guide.

In `nodes/rtx_vsr_stream.py`, accept all three public values and mention all three in unsupported-SDK errors.

- [ ] **Step 4: Run the focused tests and confirm they pass.**

Run the command from Step 2. Expected: all selected tests pass.

- [ ] **Step 5: Commit and push the backend contract.**

```powershell
git add nodes/schema.py nodes/director.py nodes/rtx_vsr_stream.py tests/test_schema.py tests/test_director.py tests/test_rtx_vsr.py
git commit -m "feat: add high bitrate RTX VSR quality"
git push origin main
```

### Task 2: Keep the Director UI stable while isolating compatible qualities

**Files:**
- Modify: `js/minimax_h3_director_plus_v9.js`
- Test: `tests/test_frontend_source.py`

- [ ] **Step 1: Write failing frontend source tests.**

Assert that the source contains `HIGHBITRATE_ULTRA（原画源最高保真）`, contains a route-aware quality resolver, and does not add another RTX quality control.

- [ ] **Step 2: Run the frontend tests and confirm they fail.**

```powershell
& 'D:\ComfyUI_windows_portable-G313\python_embeded\python.exe' -m pytest tests/test_frontend_source.py -q
```

- [ ] **Step 3: Implement dynamic options in the existing control.**

Keep the current `valueControl("RTX VSR 质量", ...)` location. Add the third Chinese-labelled option and resolve options as follows:

```javascript
function allowedRtxQualities(performancePreset) {
  return performancePreset === "质量优先二采样"
    ? RTX_QUALITIES.filter(([value]) => value === "HIGHBITRATE_ULTRA")
    : RTX_QUALITIES.filter(([value]) => value !== "HIGHBITRATE_ULTRA");
}
```

When the preset changes to quality-two-stage, set the existing widget to `HIGHBITRATE_ULTRA`; when leaving it, normalize that exclusive value back to `ULTRA`. Do not change node size, section order, or add widgets.

- [ ] **Step 4: Run the frontend tests and confirm they pass.**

Run the command from Step 2. Expected: pass.

- [ ] **Step 5: Commit and push the UI compatibility change.**

```powershell
git add js/minimax_h3_director_plus_v9.js tests/test_frontend_source.py
git commit -m "feat: isolate RTX quality options by route"
git push origin main
```

### Task 3: Apply route-scoped CRF 16 and lossless BT.709 metadata

**Files:**
- Modify: `nodes/stream_output.py`
- Test: `tests/test_stream_output.py`

- [ ] **Step 1: Write failing encoding-policy tests.**

Add direct tests for:

```python
assert _resolved_encode_quality(
    {"performance_preset": "quality_two_stage"}, "rtx_vsr", 20
) == 16
assert _resolved_encode_quality(
    {"performance_preset": "quality"}, "rtx_vsr", 20
) == 20
assert _resolved_encode_quality(
    {"performance_preset": "quality_two_stage"}, "rtx_vsr", 14
) == 14
```

Mock the encoder and assert both quality arguments receive 16 for quality-two-stage. Add a temporary-file test that verifies the BT.709 command contains `-c copy` and:

```text
h264_metadata=video_full_range_flag=0:colour_primaries=1:transfer_characteristics=1:matrix_coefficients=1
```

Add a failure test proving the original output bytes remain unchanged.

- [ ] **Step 2: Run stream-output tests and confirm they fail.**

```powershell
& 'D:\ComfyUI_windows_portable-G313\python_embeded\python.exe' -m pytest tests/test_stream_output.py -q
```

- [ ] **Step 3: Implement the encoding policy and metadata pass.**

Add:

```python
def _resolved_encode_quality(guide, postprocess_path, requested_quality):
    value = int(requested_quality)
    preset = str((guide or {}).get("performance_preset") or "")
    if preset in {"quality_two_stage", "质量优先二采样"} and postprocess_path == "rtx_vsr":
        return min(value, 16)
    return value
```

Use the resolved value for both CQ and CRF arguments. Add `_tag_h264_bt709(ffmpeg, output_path)` using `subprocess.run`, a UUID sibling temporary file, `-map 0`, `-map_metadata 0`, `-c copy`, the H.264 metadata bitstream filter, and `os.replace`. On failure, remove only that exact temporary file, log a warning, and retain the original video. Invoke it only for quality-two-stage RTX VSR H.264 MP4 output. Include actual encode quality and `bt709` status in UI metadata and the final log.

- [ ] **Step 4: Run stream-output tests and confirm they pass.**

Run the command from Step 2. Expected: pass.

- [ ] **Step 5: Commit and push the streaming output change.**

```powershell
git add nodes/stream_output.py tests/test_stream_output.py
git commit -m "feat: preserve quality in two-stage video output"
git push origin main
```

### Task 4: Regenerate the single U11 workflow and document API behavior

**Files:**
- Modify: `tools/build_u11_workflow.py`
- Modify: `templates/u11_api.json`
- Modify: `docs/使用说明.md`
- Modify: `docs/API说明.md`
- Modify: `D:/ComfyUI_windows_portable-G313/ComfyUI/user/default/workflows/minimaxH3/U11-MiniMaxH3-导演台Plus-中文增强版.json`
- Test: `tests/test_api.py`
- Test: `tests/test_workflow_tools.py`

- [ ] **Step 1: Write failing template and workflow tests.**

Require the API/template output-node quality to remain 20 for non-two-stage routes, require `HIGHBITRATE_ULTRA` support, and require the generated U11 Director node/output node to carry the same defaults without adding another workflow.

- [ ] **Step 2: Run template/workflow tests and confirm they fail.**

```powershell
& 'D:\ComfyUI_windows_portable-G313\python_embeded\python.exe' -m pytest tests/test_api.py tests/test_workflow_tools.py -q
```

- [ ] **Step 3: Update builder defaults and documentation.**

Keep the builder/API output-node `quality` at 20. Document that quality-two-stage automatically resolves it to CRF 16 and uses `HIGHBITRATE_ULTRA` plus BT.709; other routes keep their current RTX quality and encoder setting. Explain that final size remains the selected target while perceived detail is limited by the second-stage source and safe scale ratio.

- [ ] **Step 4: Rebuild the existing U11 file in place.**

Use the existing U10 source selected by the repository's documented build command and write only:

```text
D:/ComfyUI_windows_portable-G313/ComfyUI/user/default/workflows/minimaxH3/U11-MiniMaxH3-导演台Plus-中文增强版.json
```

Do not create a second U11 workflow and do not start ComfyUI.

- [ ] **Step 5: Run template/workflow tests and structural validation.**

Run the command from Step 2 and the repository workflow validation command. Expected: all pass and the single U11 JSON validates.

- [ ] **Step 6: Commit and push repository-owned files.**

```powershell
git add tools/build_u11_workflow.py templates/u11_api.json docs/使用说明.md docs/API说明.md tests/test_api.py tests/test_workflow_tools.py
git commit -m "docs: publish high quality U11 output defaults"
git push origin main
```

The user workflow JSON is outside this Git repository and is reported separately as the one rebuilt local artifact.

### Task 5: Full verification and remote-main audit

**Files:**
- Verify: all changed repository files and the single rebuilt U11 JSON

- [ ] **Step 1: Run the complete test suite.**

```powershell
& 'D:\ComfyUI_windows_portable-G313\python_embeded\python.exe' -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 2: Compile Python sources.**

```powershell
& 'D:\ComfyUI_windows_portable-G313\python_embeded\python.exe' -m compileall -q nodes tools
```

Expected: exit code 0.

- [ ] **Step 3: Audit repository and workflow state.**

Confirm the worktree is clean, the workflow contains exactly one Director node and one streaming output node, its visible output quality is 20 while quality-two-stage resolves to 16, and local `HEAD`, `origin/main`, and GitHub `main` resolve to the same SHA.

- [ ] **Step 4: Report the handoff.**

State the pushed SHA, the exact U11 path, tests run, that custom-node code changed, that ComfyUI was not started, and that the user must update/restart the server node before testing.
