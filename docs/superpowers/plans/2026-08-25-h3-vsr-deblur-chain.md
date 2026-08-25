# H3 RTX Deblur Quality Chain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an isolated `DEBLUR_LOW -> HIGHBITRATE_ULTRA` RTX chain for high-VRAM H3 quality two-stage output without changing H3 sampling memory or low-VRAM routes.

**Architecture:** Extend the existing single-effect NVIDIA frame processor with a CUDA-preserving `process_cuda()` boundary and a two-effect owner that closes both effects deterministically. The director selects and probes the chain only for `quality_two_stage + rtx_vsr`; the streaming encoder center-crops frames to the target aspect ratio, then processes one frame at a time through the selected chain.

**Tech Stack:** Python 3.12, PyTorch CUDA/DLPack, NVIDIA `nvidia-vfx`, ComfyUI custom nodes, pytest, existing DaSiWa FFmpeg encoder, JSON workflow builder.

---

## File map

- Modify `nodes/rtx_vsr_stream.py`: CUDA-preserving single-effect output, dual-effect processor, real-frame chain preflight.
- Modify `nodes/stream_output.py`: aspect-safe center crop, route isolation, streaming dual-effect selection.
- Modify `nodes/director.py`: preflight routing and explicit guide metadata.
- Modify `tests/test_rtx_vsr.py`: processor order, shapes, cleanup, and preflight failure tests.
- Modify `tests/test_stream_output.py`: route isolation, center crop, frame count, and bounded-chunk tests.
- Modify `tests/test_director.py`: high-quality chain selection and low-VRAM/non-two-stage isolation.
- Modify `docs/使用说明.md`, `docs/API说明.md`, `docs/故障排查.md`: Chinese usage, API visibility, and error guidance.
- Regenerate `U11-MiniMaxH3-导演台Plus-中文增强版.json` with `tools/build_u11_workflow.py` and validate it with existing workflow tests.

### Task 1: CUDA-preserving RTX processor and two-effect owner

**Files:**
- Modify: `nodes/rtx_vsr_stream.py`
- Test: `tests/test_rtx_vsr.py`

- [ ] **Step 1: Write failing tests for CUDA chaining and cleanup order**

Add tests that import `DeblurVsrFrameProcessor` and assert one frame is processed by `DEBLUR_LOW` before `HIGHBITRATE_ULTRA`, with the intermediate tensor remaining CHW and both child processors closed even when the second effect fails:

```python
def test_deblur_vsr_processor_runs_low_deblur_before_upscale(monkeypatch):
    calls = []

    class FakeProcessor:
        def __init__(self, api, quality, device_id, width, height):
            calls.append(("create", quality, width, height))
            self.quality = quality
            self.width = width
            self.height = height

        def process_cuda(self, frame):
            calls.append(("cuda", self.quality, tuple(frame.shape)))
            return torch.ones(3, self.height, self.width)

        def process(self, frame):
            calls.append(("cpu", self.quality, tuple(frame.shape)))
            return torch.ones(self.height, self.width, 3)

        def close(self):
            calls.append(("close", self.quality))

    monkeypatch.setattr(rtx_vsr_stream, "VsrFrameProcessor", FakeProcessor)
    processor = rtx_vsr_stream.DeblurVsrFrameProcessor(
        object(), "HIGHBITRATE_ULTRA", 0, 640, 360, 1280, 720
    )
    output = processor.process(torch.zeros(3, 360, 640))
    processor.close()

    assert output.shape == (720, 1280, 3)
    assert calls[:4] == [
        ("create", "DEBLUR_LOW", 640, 360),
        ("create", "HIGHBITRATE_ULTRA", 1280, 720),
        ("cuda", "DEBLUR_LOW", (3, 360, 640)),
        ("cpu", "HIGHBITRATE_ULTRA", (3, 360, 640)),
    ]
    assert calls[-2:] == [
        ("close", "HIGHBITRATE_ULTRA"),
        ("close", "DEBLUR_LOW"),
    ]
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```powershell
D:\ComfyUI_windows_portable-G313\python_embeded\python.exe -m pytest tests/test_rtx_vsr.py::test_deblur_vsr_processor_runs_low_deblur_before_upscale -q
```

Expected: collection or import fails because `DeblurVsrFrameProcessor` does not exist.

- [ ] **Step 3: Add `process_cuda()` and the dual-effect owner**

Refactor the existing processor so `process_cuda()` returns an immediately cloned CUDA CHW tensor and `process()` performs only the final CPU HWC conversion. Add the fixed two-effect owner:

```python
class DeblurVsrFrameProcessor:
    def __init__(self, api, quality, device_id, input_width, input_height,
                 output_width, output_height):
        self._deblur = VsrFrameProcessor(
            api, "DEBLUR_LOW", device_id, input_width, input_height
        )
        try:
            self._upscale = VsrFrameProcessor(
                api, quality, device_id, output_width, output_height
            )
        except Exception:
            self._deblur.close()
            raise

    def process(self, chw_frame):
        deblurred = self._deblur.process_cuda(chw_frame)
        return self._upscale.process(deblurred)

    def close(self):
        primary = None
        try:
            self._upscale.close()
        except Exception as exc:
            primary = exc
        try:
            self._deblur.close()
        except Exception:
            if primary is None:
                raise
        if primary is not None:
            raise primary
```

- [ ] **Step 4: Run RTX processor tests and verify GREEN**

Run:

```powershell
D:\ComfyUI_windows_portable-G313\python_embeded\python.exe -m pytest tests/test_rtx_vsr.py -q
```

Expected: all `test_rtx_vsr.py` tests pass.

- [ ] **Step 5: Commit the processor boundary**

```powershell
git add nodes/rtx_vsr_stream.py tests/test_rtx_vsr.py
git commit -m "feat: add chained RTX deblur processor"
```

### Task 2: Real-frame dual-effect preflight

**Files:**
- Modify: `nodes/rtx_vsr_stream.py`
- Test: `tests/test_rtx_vsr.py`

- [ ] **Step 1: Write failing tests for preflight order and named failures**

Add a test that patches `DeblurVsrFrameProcessor`, verifies the real CUDA test shape and expects a Chinese failure identifying the chain:

```python
def test_probe_vsr_deblur_chain_uses_real_360p_frame(monkeypatch):
    calls = []

    class FakeChain:
        def __init__(self, api, quality, device_id, input_width, input_height,
                     output_width, output_height):
            calls.append((quality, input_width, input_height, output_width, output_height))

        def process(self, frame):
            calls.append(tuple(frame.shape))
            return torch.zeros(720, 1280, 3)

        def close(self):
            calls.append("close")

    monkeypatch.setattr(rtx_vsr_stream, "DeblurVsrFrameProcessor", FakeChain)
    monkeypatch.setattr(rtx_vsr_stream, "load_vsr_api", lambda: object())
    _mock_probe_cuda(monkeypatch)
    _mock_probe_input(monkeypatch)

    assert rtx_vsr_stream.probe_vsr_deblur_chain("HIGHBITRATE_ULTRA", 0)
    assert calls[0] == ("HIGHBITRATE_ULTRA", 640, 360, 1280, 720)
    assert calls[1:] == [(3, 360, 640), "close"]
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```powershell
D:\ComfyUI_windows_portable-G313\python_embeded\python.exe -m pytest tests/test_rtx_vsr.py::test_probe_vsr_deblur_chain_uses_real_360p_frame -q
```

Expected: FAIL because `probe_vsr_deblur_chain` is absent.

- [ ] **Step 3: Implement strict chain preflight**

Add `probe_vsr_deblur_chain()` using the existing probe constants, `_probe_failure()` wrapper, primary-error preservation, cleanup checks, and this success log:

```python
LOGGER.info(
    "RTX 轻度去模糊 + VSR 前置检查成功：deblur=DEBLUR_LOW "
    "quality=%s gpu=%s input=%sx%s output=%sx%s",
    quality, device_id, PROBE_INPUT_WIDTH, PROBE_INPUT_HEIGHT,
    PROBE_OUTPUT_WIDTH, PROBE_OUTPUT_HEIGHT,
)
```

The failure text must include `RTX 轻度去模糊 + VSR 前置检查失败，尚未开始 H3 视频生成` and retain the original NVIDIA error as `__cause__`.

- [ ] **Step 4: Run RTX tests and verify GREEN**

Run the full `tests/test_rtx_vsr.py`; expected result is PASS with no CUDA allocation on machines where CUDA is mocked.

- [ ] **Step 5: Commit the preflight**

```powershell
git add nodes/rtx_vsr_stream.py tests/test_rtx_vsr.py
git commit -m "feat: preflight chained RTX quality effects"
```

### Task 3: Aspect-safe streaming and route isolation

**Files:**
- Modify: `nodes/stream_output.py`
- Test: `tests/test_stream_output.py`

- [ ] **Step 1: Write failing tests for center crop and isolated chain selection**

Add tests for a 1920×1056 source targeting 2560×1440. The helper must crop width to an even 1876 pixels, retain all 1056 rows, and only the quality two-stage route may instantiate `DeblurVsrFrameProcessor`:

```python
def test_center_crop_to_target_aspect_prevents_non_uniform_vsr_scale():
    frame = torch.zeros(1056, 1920, 3)
    cropped = stream_output._center_crop_to_target_aspect(frame, 2560, 1440)
    assert cropped.shape == (1056, 1876, 3)
    assert abs(cropped.shape[1] / cropped.shape[0] - 16 / 9) < 0.002


def test_quality_two_stage_uses_deblur_vsr_chain(monkeypatch):
    created = []
    captured = []

    class FakeChain:
        def __init__(self, *args):
            created.append(args)
            self.height, self.width = args[-1], args[-2]
        def process(self, frame):
            return torch.zeros(self.height, self.width, 3)
        def close(self):
            pass

    monkeypatch.setattr(stream_output, "load_vsr_api", lambda: "api")
    monkeypatch.setattr(stream_output, "DeblurVsrFrameProcessor", FakeChain)
    monkeypatch.setattr(
        stream_output,
        "VsrFrameProcessor",
        lambda *args: (_ for _ in ()).throw(AssertionError("ordinary VSR used")),
    )
    monkeypatch.setattr(stream_output, "release_sampling_models", lambda: None)

    _combine(
        monkeypatch,
        {
            "performance_preset": "quality_two_stage",
            "target_width": 16,
            "target_height": 9,
            "postprocess_path": "rtx_vsr",
            "rtx_quality": "HIGHBITRATE_ULTRA",
            "rtx_deblur_mode": "DEBLUR_LOW",
        },
        torch.zeros(3, 6, 11, 3),
        captured,
    )

    assert len(created) == 1
    assert sum(len(chunk) for chunk in captured) == 3
    assert all(chunk.shape[1:] == (9, 16, 3) for chunk in captured)
```

- [ ] **Step 2: Run focused stream tests and verify RED**

Run the two new tests. Expected: missing crop helper and no dual-effect route.

- [ ] **Step 3: Implement aspect crop and explicit processor factory**

Add:

```python
def _center_crop_to_target_aspect(frame, target_width, target_height):
    height, width = int(frame.shape[0]), int(frame.shape[1])
    source_ratio = width / height
    target_ratio = int(target_width) / int(target_height)
    if abs(source_ratio - target_ratio) < 1e-6:
        return frame
    if source_ratio > target_ratio:
        crop_width = max(2, int(height * target_ratio) // 2 * 2)
        left = (width - crop_width) // 2
        return frame[:, left:left + crop_width]
    crop_height = max(2, int(width / target_ratio) // 2 * 2)
    top = (height - crop_height) // 2
    return frame[top:top + crop_height]
```

Pass `deblur_before_upscale=True` only when the guide resolves to `quality_two_stage` and `rtx_deblur_mode == "DEBLUR_LOW"`. Initialize the selected processor after the first cropped frame reveals its exact input dimensions. Continue yielding at most four CPU frames per chunk and close the active processor and upstream generator in `finally`.

- [ ] **Step 4: Run stream tests and verify GREEN**

Run:

```powershell
D:\ComfyUI_windows_portable-G313\python_embeded\python.exe -m pytest tests/test_stream_output.py -q
```

Expected: all stream tests pass; existing regular RTX and RIFE paths still use `VsrFrameProcessor`.

- [ ] **Step 5: Commit the streaming route**

```powershell
git add nodes/stream_output.py tests/test_stream_output.py
git commit -m "feat: stream aspect-safe RTX deblur quality output"
```

### Task 4: Director routing, Chinese metadata, and API contract

**Files:**
- Modify: `nodes/director.py`
- Modify: `docs/使用说明.md`
- Modify: `docs/API说明.md`
- Modify: `docs/故障排查.md`
- Test: `tests/test_director.py`
- Test: `tests/test_docs.py`

- [ ] **Step 1: Write failing director isolation tests**

Update the quality two-stage build test to patch `probe_vsr_deblur_chain`, assert the ordinary probe is not called, and require:

```python
assert guide["rtx_deblur_mode"] == "DEBLUR_LOW"
assert guide["rtx_quality"] == "HIGHBITRATE_ULTRA"
assert chain_probes == [("HIGHBITRATE_ULTRA", 0)]
assert normal_probes == []
```

Add a low-VRAM/ordinary RTX test asserting `rtx_deblur_mode == "off"` and only `probe_vsr_capability()` runs.

- [ ] **Step 2: Run focused director tests and verify RED**

Expected: FAIL because the director has no chain probe import or `rtx_deblur_mode` guide field.

- [ ] **Step 3: Implement the director route**

Import `probe_vsr_deblur_chain`. Before H3 generation:

```python
if postprocess_path == "rtx_vsr":
    if request["performance_preset"] == "quality_two_stage":
        probe_vsr_deblur_chain(request["rtx_quality"], device_id=0)
        request["rtx_deblur_mode"] = "DEBLUR_LOW"
    else:
        probe_vsr_capability(request["rtx_quality"], device_id=0)
        request["rtx_deblur_mode"] = "off"
else:
    request["rtx_deblur_mode"] = "off"
```

Wrap failures with the existing `RequestError` pre-generation message without replacing the original cause. Add the guide field and Chinese warning `质量二采将执行 DEBLUR_LOW 轻度去模糊，再执行 HIGHBITRATE_ULTRA 高码率 RTX VSR。`.

- [ ] **Step 4: Update Chinese docs and tests**

Document that the API does not add a required input; clients inspect `rtx_deblur_mode`, `postprocess_path`, `source_width/source_height`, and final target dimensions from guide/status metadata. Explain that random seeds can still change generated texture and that the chain does not turn the H3 source into native 2K.

- [ ] **Step 5: Run director and docs tests and verify GREEN**

Run:

```powershell
D:\ComfyUI_windows_portable-G313\python_embeded\python.exe -m pytest tests/test_director.py tests/test_docs.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit route and documentation**

```powershell
git add nodes/director.py tests/test_director.py docs/使用说明.md docs/API说明.md docs/故障排查.md tests/test_docs.py
git commit -m "feat: isolate RTX deblur chain for quality two-stage"
```

### Task 5: Workflow rebuild, complete verification, and remote main

**Files:**
- Regenerate: `D:/ComfyUI_windows_portable-G313/ComfyUI/user/default/workflows/minimaxH3/U11-MiniMaxH3-导演台Plus-中文增强版.json`
- Verify: all repository tests

- [ ] **Step 1: Rebuild the single U11 workflow without starting ComfyUI**

Run the existing builder with its documented U10 source and U11 destination. Confirm it reports one workflow, one director subgraph, and no manual connection requirement.

- [ ] **Step 2: Run workflow structure tests**

```powershell
D:\ComfyUI_windows_portable-G313\python_embeded\python.exe -m pytest tests/test_workflow_tools.py tests/test_frontend_source.py -q
```

Expected: PASS with zero overlap failures and no second U11 output.

- [ ] **Step 3: Run the complete repository suite**

```powershell
D:\ComfyUI_windows_portable-G313\python_embeded\python.exe -m pytest tests -q
```

Expected: all tests pass; no new warning beyond the repository's known baseline.

- [ ] **Step 4: Run static and repository checks**

```powershell
git diff --check
git status --short
```

Expected: no whitespace errors; only intended node, test, docs, and workflow changes are present.

- [ ] **Step 5: Commit remaining generated changes and push**

```powershell
git add -A
git commit -m "feat: sharpen high quality H3 RTX output"
git push origin main
```

Expected: remote `main` advances and the worktree is clean. Do not start ComfyUI locally; report that the custom node code changed and the server must pull and restart before testing.
