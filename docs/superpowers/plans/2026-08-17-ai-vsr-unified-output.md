# U11 Unified AI VSR Output Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one strict, optional RTX VSR final-output path to U11 so every H3 performance preset can produce either native-size MP4 or a single AI-upscaled final MP4 without full 2K/4K GPU batches.

**Architecture:** Keep H3 generation and postprocessing independent. `MiniMaxH3DirectorPlus` resolves requested/native dimensions, low-VRAM target limits, and the requested postprocess mode into the guide; `MiniMaxH3StreamingVideoCombine` performs the actual native, downscale, or RTX VSR frame path before one FFmpeg encode. A focused RTX VSR helper owns dependency checks, effect lifetime, frame conversion, and actionable Windows errors.

**Tech Stack:** Python 3, PyTorch/CUDA, ComfyUI custom-node APIs, NVIDIA `nvvfx`/`nvidia-vfx`, DaSiWa FFmpeg encoder helpers, vanilla JavaScript Director UI, pytest.

---

## File Map

- Create: `nodes/rtx_vsr_stream.py` — strict `nvvfx` loader and one-frame/iterator VSR processor; no UI node and no manual graph connection.
- Modify: `nodes/schema.py` — public postprocess fields, defaults, validation, and low-VRAM target limit helper.
- Modify: `nodes/director.py` — Director widgets, guide fields, target-path resolution, and low-VRAM rejection before execution.
- Modify: `nodes/stream_output.py` — choose native/downscale/RTX VSR path and feed only small frame chunks to FFmpeg.
- Modify: `nodes/status.py` — report `nvvfx`/VideoSuperRes availability and installation state.
- Modify: `api/template.py`, `templates/u11_api.json` — expose postprocess parameters to API prompts while preserving old API default behavior.
- Modify: `js/minimax_h3_director_plus_v9.js` — add stable Chinese postprocess and RTX quality controls without changing the existing Director layout width.
- Modify: `tools/build_u11_workflow.py` and generated U11 JSON — keep the single output node and route the guide to it.
- Create: `tests/test_rtx_vsr.py` — dependency errors, quality mapping, frame conversion, and effect cleanup.
- Modify: `tests/test_director.py`, `tests/test_schema.py`, `tests/test_api.py`, `tests/test_status.py`, `tests/test_workflow_tools.py`, `tests/test_frontend_source.py` — contract and regression coverage.
- Modify: `docs/使用说明.md`, `docs/API说明.md` — installation, mode behavior, and exact output-path semantics.

### Task 1: Add the Postprocess Contract and Low-VRAM Validation

**Files:**
- Test: `tests/test_schema.py`, `tests/test_director.py`
- Modify: `nodes/schema.py`, `nodes/director.py`

- [ ] **Step 1: Write failing schema tests.** Add assertions that `public_schema()` exposes `postprocess_mode` (`native`, `rtx_vsr`) and `rtx_quality` (`HIGH`, `ULTRA`), and that `normalize_request()` defaults API requests without these keys to `native` while rejecting unknown values.

```python
def test_schema_exposes_final_postprocess_controls():
    schema = public_schema()["properties"]
    assert schema["postprocess_mode"]["enum"] == ["native", "rtx_vsr"]
    assert schema["rtx_quality"]["enum"] == ["HIGH", "ULTRA"]

def test_normalize_request_defaults_to_native_postprocess():
    request = normalize_request({"mode": "T2VA", "duration": 4})
    assert request["postprocess_mode"] == "native"
    assert request["rtx_quality"] == "HIGH"

def test_normalize_request_rejects_unknown_postprocess_mode():
    with pytest.raises(RequestError, match="后处理模式"):
        normalize_request({"mode": "T2VA", "postprocess_mode": "magic"})
```

- [ ] **Step 2: Run the focused tests and confirm the expected failure.**

Run: `pytest tests/test_schema.py -q`

Expected: FAIL because the public schema and normalized request do not yet expose the new fields.

- [ ] **Step 3: Implement the request contract.** In `nodes/schema.py`:

  - Add `postprocess_mode` and `rtx_quality` to `PUBLIC_API_KEYS`.
  - Add `POSTPROCESS_MODES = ("native", "rtx_vsr")` and `RTX_QUALITIES = ("HIGH", "ULTRA")`.
  - Add `postprocess_mode: "native"` and `rtx_quality: "HIGH"` to `DEFAULT_REQUEST`.
  - Validate both fields in `normalize_request()` with Chinese `RequestError` messages.
  - Add Chinese names, enum values, and descriptions to `public_schema()`.
  - Add `low_vram_target_limit(duration)` returning `(2560, 1440)` for durations 4–6 and `(1920, 1080)` for durations 8–15.

- [ ] **Step 4: Add Director controls and guide resolution.** In `nodes/director.py`:

  - Add required widgets `postprocess_mode` (`AI 细节重建（RTX VSR）` in the UI mapping, internal `native`/`rtx_vsr`) and `rtx_quality`.
  - Pass them through `build()` into `normalize_request()`.
  - When `performance_preset == "low_vram"`, reject requested dimensions above `low_vram_target_limit(duration)` with a `RequestError` naming the allowed maximum.
  - Compute `postprocess_path` after `native_resolution_for_request()`:

```python
if requested_width == native_width and requested_height == native_height:
    postprocess_path = "native_bypass"
elif requested_width < native_width or requested_height < native_height:
    postprocess_path = "downscale"
elif request["postprocess_mode"] == "rtx_vsr":
    postprocess_path = "rtx_vsr"
else:
    postprocess_path = "native_bypass"
```

  - Store `postprocess_mode`, `rtx_quality`, and `postprocess_path` in the guide. Set `upscale_method` to `"rtx_vsr"`, `"cpu_bicubic"`, or `"none"` based on the resolved path.
  - Preserve all existing backend-specific acceleration routing; no performance preset may rewrite these fields.

- [ ] **Step 5: Run Director and schema tests.**

Run: `pytest tests/test_schema.py tests/test_director.py -q`

Expected: PASS, including same-size bypass, target-upscale selection, downscale selection, and low-VRAM rejection cases added in this task.

- [ ] **Step 6: Commit the contract change.**

```bash
git add nodes/schema.py nodes/director.py tests/test_schema.py tests/test_director.py
git commit -m "feat: add unified postprocess contract and low-vram limits"
```

### Task 2: Implement the Strict RTX VSR Frame Helper

**Files:**
- Test: `tests/test_rtx_vsr.py`
- Create: `nodes/rtx_vsr_stream.py`

- [ ] **Step 1: Write failing helper tests.** Test with monkeypatched modules, without requiring a real RTX SDK:

```python
def test_import_vsr_reports_embedded_python_and_install_command(monkeypatch):
    monkeypatch.setitem(sys.modules, "nvvfx", None)
    with pytest.raises(RuntimeError, match="python_embeded|nvidia-vfx|Broadcast SDK"):
        load_vsr_api()

def test_quality_mapping_uses_vsr_quality_level():
    quality = SimpleNamespace(HIGH="high", ULTRA="ultra")
    assert resolve_vsr_quality(quality, "HIGH") == "high"
    assert resolve_vsr_quality(quality, "ULTRA") == "ultra"

def test_process_frame_returns_hwc_float_frame(monkeypatch):
    processor = VsrFrameProcessor(fake_api, "HIGH", 0, 1920, 1080)
    result = processor.process(torch.zeros(3, 64, 96))
    assert result.shape == (1080, 1920, 3)
    assert result.dtype == torch.float32
```

- [ ] **Step 2: Run the new tests and verify they fail for missing symbols.**

Run: `pytest tests/test_rtx_vsr.py -q`

Expected: FAIL because `load_vsr_api`, `resolve_vsr_quality`, and `VsrFrameProcessor` do not exist.

- [ ] **Step 3: Implement `nodes/rtx_vsr_stream.py`.** Define these stable interfaces:

```python
def load_vsr_api() -> tuple[object, object]:
    raise NotImplementedError

def resolve_vsr_quality(quality_level, quality: str):
    raise NotImplementedError

class VsrFrameProcessor:
    def __init__(self, api, quality: str, device_id: int, output_width: int, output_height: int):
        raise NotImplementedError

    def process(self, chw_frame: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError
```

  - `load_vsr_api()` imports `nvvfx`, locates `VideoSuperRes` and its `QualityLevel`, and raises a Chinese error containing `sys.executable`, the absolute `D:/ComfyUI_windows_portable-G313/ComfyUI/custom_nodes/ComfyUI-DaSiWa-Nodes/requirements.txt` path, `D:/ComfyUI_windows_portable-G313/python_embeded/python.exe -m pip install -r D:/ComfyUI_windows_portable-G313/ComfyUI/custom_nodes/ComfyUI-DaSiWa-Nodes/requirements.txt`, the possible NVIDIA Broadcast SDK requirement, and a restart/verification command.
  - Map only `HIGH` and `ULTRA` to RTX VSR quality attributes; reject all other values.
  - Create the effect with the same constructor compatibility attempts as DaSiWa, set `output_width`/`output_height`, and call `load()` when available.
  - Convert HWC `[0, 1]` frames to contiguous CUDA float32 `CHW`, call `effect.run()`, synchronize CUDA before/after, clone the DLPack image, convert back to contiguous HWC float32 on CPU for FFmpeg.
  - Close/destroy/unload the effect in `close()` and also from a context manager so FFmpeg retry paths cannot leak an effect.

- [ ] **Step 4: Run helper tests and the existing color/upscale tests.**

Run: `pytest tests/test_rtx_vsr.py tests/test_color_guard.py tests/test_upscale.py -q`

Expected: PASS without requiring `nvvfx` because all SDK interactions are mocked.

- [ ] **Step 5: Commit the helper.**

```bash
git add nodes/rtx_vsr_stream.py tests/test_rtx_vsr.py
git commit -m "feat: add strict streaming RTX VSR helper"
```

### Task 3: Route One Final Video Through Native, Downscale, or AI VSR

**Files:**
- Test: `tests/test_stream_output.py` (create if absent)
- Modify: `nodes/stream_output.py`

- [ ] **Step 1: Write failing stream-path tests.** In `tests/test_stream_output.py`, add four tests using a fake DaSiWa encoder and fake VSR processor: `test_native_bypass_does_not_resize_or_require_vsr` asserts source width/height and frame values are unchanged; `test_same_size_rtx_request_is_reported_as_native_bypass` asserts a guide requesting RTX at equal dimensions resolves to `native_bypass`; `test_rtx_vsr_path_processes_frames_and_preserves_frame_count` records one processed output per input frame and asserts the encoder receives the same count, frame rate, and audio object; `test_output_path_never_materializes_full_target_batch` iterates the generator and asserts each yielded chunk has at most four frames and no tensor has the full target batch shape. Each test must also assert `load_vsr_api()` is not called for `native_bypass` or `downscale`.

- [ ] **Step 2: Run the focused tests and confirm failure.**

Run: `pytest tests/test_stream_output.py -q`

Expected: FAIL because `MiniMaxH3StreamingVideoCombine` currently assumes CPU bicubic when resizing and has no resolved postprocess path.

- [ ] **Step 3: Add path-specific frame iterators.** In `nodes/stream_output.py`:

  - Keep `_iter_resized_frame_chunks()` for `downscale` only.
  - Add `_iter_native_frame_chunks()` that yields source slices without copying a complete batch.
  - Add `_iter_vsr_frame_chunks(source, guide, target_width, target_height, quality)` that creates one `VsrFrameProcessor` per encoder attempt, processes one source frame at a time, and yields small CPU HWC chunks. The generator must close the processor in `finally`.
  - Use `guide["postprocess_path"]` as the source of truth. If absent in old workflows, derive the legacy native/bicubic behavior from `target_width`, `target_height`, and `upscale_required`.
  - Keep target dimensions and output metadata aligned with the final encoded frames.
  - For `save_first_frame`/`save_last_frame`, use the same resolved path; if VSR is selected, run the single requested frame through the helper rather than saving a bicubic mismatch.
  - Return an empty `frames` batch by default so ComfyUI cannot retain a full target-size batch; keep `pass_frames` only for native-size output.
  - Never fall back from `rtx_vsr` to CPU bicubic. Propagate the actionable SDK error and include the selected actual path in logs/UI metadata.

- [ ] **Step 4: Run stream tests and regression tests.**

Run: `pytest tests/test_stream_output.py tests/test_workflow_tools.py tests/test_director.py -q`

Expected: PASS; old native workflows still produce MP4, same-size requests bypass VSR, and AI requests fail clearly when the SDK is unavailable.

- [ ] **Step 5: Commit the stream integration.**

```bash
git add nodes/stream_output.py tests/test_stream_output.py
git commit -m "feat: route final output through optional streaming RTX VSR"
```

### Task 4: Expose the Controls in the Director UI, API, and Capability Status

**Files:**
- Test: `tests/test_api.py`, `tests/test_status.py`, `tests/test_frontend_source.py`
- Modify: `js/minimax_h3_director_plus_v9.js`, `api/template.py`, `templates/u11_api.json`, `nodes/status.py`, `docs/使用说明.md`, `docs/API说明.md`

- [ ] **Step 1: Add failing API/status/frontend tests.** Assert that API template/controller inputs contain `postprocess_mode` and `rtx_quality`, API patching preserves them, status includes `postprocess.rtx_vsr`, and the Director source contains the Chinese labels and no second output control surface.

- [ ] **Step 2: Run focused tests and confirm failure.**

Run: `pytest tests/test_api.py tests/test_status.py tests/test_frontend_source.py -q`

Expected: FAIL because the template, status payload, and UI do not expose these controls.

- [ ] **Step 3: Implement API propagation.**

  - Add the two fields to `api/template.py`'s `public_controller_fields`.
  - Add them to node `10` in `templates/u11_api.json`; keep omitted API requests native for backward compatibility.
  - Add Chinese schema descriptions and accepted enums in `nodes/schema.py`.
  - In `nodes/status.py`, probe the same ComfyUI Python environment for `nvvfx` and `VideoSuperRes`; return `postprocess.rtx_vsr = {"node_available": true, "dependency_available": bool, "message": "RTX VSR 依赖已就绪" or "缺少 nvidia-vfx/nvvfx，请安装后重启 ComfyUI"}` without importing or downloading models at startup.

- [ ] **Step 4: Implement the stable Director UI.** In `js/minimax_h3_director_plus_v9.js`:

  - Add `POSTPROCESS_MODES` and `RTX_QUALITIES` constants.
  - Render a compact “最终输出 / AI 细节重建” control beside the existing resolution field, plus `High/Ultra` only when AI mode is selected.
  - Keep the existing `DIRECTOR_UI_WIDTH`/height and sidebar-independent CSS; do not add another output node or a second resolution selector.
  - Show “原生尺寸直出” whenever target and native dimensions match, and show low-VRAM target-limit warnings before queueing.
  - Keep performance preset selection independent so changing it does not erase postprocess mode.

- [ ] **Step 5: Update Chinese usage/API documentation.** Document:

  - `nvidia-vfx` installation in the ComfyUI embedded Python environment and Broadcast SDK note.
  - Native-size bypass, downscale, and RTX VSR path behavior.
  - 3070 8G low-VRAM caps: 4–6 seconds up to 2K; 8–15 seconds up to 1080p.
  - One final MP4 only; no change to frame count, audio, timing, aspect ratio, or composition.

- [ ] **Step 6: Run API/status/frontend tests and commit.**

```bash
pytest tests/test_api.py tests/test_status.py tests/test_frontend_source.py -q
git add api/template.py templates/u11_api.json nodes/status.py js/minimax_h3_director_plus_v9.js docs/使用说明.md docs/API说明.md tests/test_api.py tests/test_status.py tests/test_frontend_source.py
git commit -m "feat: expose unified AI output controls"
```

### Task 5: Regenerate and Validate the Single U11 Workflow

**Files:**
- Modify: `tools/build_u11_workflow.py`
- Regenerate: `D:/ComfyUI_windows_portable-G313/ComfyUI/user/default/workflows/minimaxH3/U11-MiniMaxH3-导演台Plus-中文增强版.json`
- Test: `tests/test_workflow_tools.py`

- [ ] **Step 1: Add failing workflow assertions.** Assert that the built U11 graph contains exactly one `MiniMaxH3StreamingVideoCombine`, the guide reaches it, its output remains H.264/MP4, and no performance preset has a separate postprocess branch or extra resolution widget.

- [ ] **Step 2: Run the workflow tests and confirm failure.**

Run: `pytest tests/test_workflow_tools.py -q`

Expected: FAIL on the new postprocess field/metadata assertions.

- [ ] **Step 3: Update the builder/template metadata.** Keep the single output node and existing organized positions; add only the new Director widgets/guide metadata and the output-path note. Do not import U13's four duplicated mode graphs or `CreateVideo → SaveVideo` full-batch chain.

- [ ] **Step 4: Regenerate the JSON and validate layout.** Run the repository's builder command, then parse the generated JSON to assert node/link counts, no visible rectangle overlap, and a single final output node. Verify the output node still receives the `MiniMaxH3ColorGuard`/guide route.

- [ ] **Step 5: Commit the workflow.**

```bash
git add tools/build_u11_workflow.py tests/test_workflow_tools.py "D:/ComfyUI_windows_portable-G313/ComfyUI/user/default/workflows/minimaxH3/U11-MiniMaxH3-导演台Plus-中文增强版.json"
git commit -m "feat: regenerate U11 with unified final AI output"
```

### Task 6: Full Verification and Remote Main Push

**Files:**
- No new production files; inspect all commits and generated artifacts.

- [ ] **Step 1: Run the complete test suite.**

Run: `pytest -q`

Expected: all existing tests plus new tests pass; no warnings caused by missing optional `nvvfx` during tests.

- [ ] **Step 2: Run static workflow/API checks.**

Run the project workflow validator and verify:

  - U11 has one final output node.
  - Native-size, downscale, and RTX VSR metadata paths are represented.
  - Low-VRAM target caps reject invalid requests before queueing.
  - API schema and `u11_api.json` agree on field names/defaults.

- [ ] **Step 3: Verify dependency error text in the embedded environment.**

Run: `D:/ComfyUI_windows_portable-G313/python_embeded/python.exe -c "import importlib.util; print(importlib.util.find_spec('nvvfx'))"`

Expected on the current machine before installation: `None`; executing a real `rtx_vsr` request must show the actionable install message and produce no MP4.

- [ ] **Step 4: If verification finds a defect, fix it in the owning task's files, rerun the affected focused test, and create a separate commit named `fix: correct unified AI output verification failure`; otherwise leave the tree unchanged.**

- [ ] **Step 5: Push every implementation commit to remote main.**

```bash
git push origin main
```

Expected: remote `main` advances to the final verified commit; report the commit hash and test count to the user.
