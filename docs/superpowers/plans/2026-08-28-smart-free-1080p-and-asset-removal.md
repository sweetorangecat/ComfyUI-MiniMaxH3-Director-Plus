# Smart Free 1080p and Asset Removal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make “免费智能 1080p” the safe default for every H3 mode, route each request by the resolved backend and real VRAM state, run exactly one local RealESRGAN X2 reconstruction, reject structurally invalid frames before upscale, and let users remove or compact uploaded media references safely inside the Director UI.

**Architecture:** Add one pure policy module that owns smart route selection, 1080p target geometry, and the low-VRAM duration contract. Keep schema normalization responsible for public aliases and compatibility fallback, then let the Director resolve the smart preset after `resolved_backend` and CUDA memory are known. Reuse the existing streaming AI-upscale path, but add pre-upscale frame validation and reuse processed edge frames so preview exports never invoke a second super-resolution pass. Extract upload-slot compaction into a browser-independent ES module and have the Director DOM call it when clearing widgets; disk files remain untouched.

**Tech Stack:** Python 3.10+, PyTorch, pytest, ComfyUI custom nodes, browser ES modules, Node.js `node:test`, existing RealESRGAN/ComfyUI upscale-model loader, existing streaming H.264/MP4 encoder.

---

## Task 1: Add the pure smart-1080p policy

**Files:**
- Create: `nodes/smart_1080p.py`
- Create: `tests/test_smart_1080p.py`

- [ ] **Step 1: Write failing route and target-size tests**

```python
import pytest

from nodes.schema import RequestError
from nodes.smart_1080p import resolve_smart_1080p_plan, smart_1080p_target


@pytest.mark.parametrize(
    ("width", "height", "expected"),
    [
        (16, 9, (1920, 1080)),
        (9, 16, (1080, 1920)),
        (1, 1, (1080, 1080)),
        (21, 9, (2520, 1080)),
        (9, 21, (1080, 2520)),
    ],
)
def test_smart_target_keeps_aspect_and_uses_1080_short_edge(width, height, expected):
    assert smart_1080p_target(width, height) == expected


@pytest.mark.parametrize("backend", ["ref2va_model", "fl2va_model"])
def test_normal_vram_uses_backend_specific_safe_route(backend):
    plan = resolve_smart_1080p_plan(backend, duration=6, total_vram_gb=24, free_vram_gb=20)
    expected = "quality_sage" if backend == "ref2va_model" else "fast_4step"
    assert plan["performance_preset"] == expected
    assert plan["postprocess_mode"] == "ai_upscale"
    assert plan["ai_upscale_model"] == "RealESRGAN_x2plus.pth"
    assert plan["use_easycache"] is False


def test_low_vram_base_may_use_trained_two_stage_but_reference_never_does():
    base = resolve_smart_1080p_plan("fl2va_model", 6, 8, 7)
    reference = resolve_smart_1080p_plan("ref2va_model", 6, 8, 7)
    assert base["performance_preset"] == "low_vram_two_stage"
    assert reference["performance_preset"] == "low_vram"
    assert reference["two_stage_route"] == "bypass"


def test_low_vram_rejects_seven_seconds_with_memory_details():
    with pytest.raises(RequestError, match=r"请求 7 秒.*最多 6 秒.*总显存 8\.0GB.*空闲显存 7\.0GB"):
        resolve_smart_1080p_plan("ref2va_model", 7, 8, 7)


def test_insufficient_free_vram_is_rejected_even_when_total_vram_is_large():
    with pytest.raises(RequestError, match=r"空闲显存 5\.5GB.*至少需要 6\.0GB"):
        resolve_smart_1080p_plan("fl2va_model", 4, 24, 5.5)
```

- [ ] **Step 2: Run the new tests and confirm the missing-module failure**

Run: `python -m pytest tests/test_smart_1080p.py -q`

Expected: FAIL with `ModuleNotFoundError: nodes.smart_1080p`.

- [ ] **Step 3: Implement the deterministic policy**

Create `nodes/smart_1080p.py` with these public contracts:

```python
from .schema import RequestError


SMART_PRESET = "smart_free_1080p"
SMART_UPSCALE_MODEL = "RealESRGAN_x2plus.pth"
LOW_VRAM_MAX_SECONDS = 6
LOW_VRAM_MIN_FREE_GB = 6.0
LOW_VRAM_TOTAL_GB = 16.0


def _even(value):
    return max(2, int(round(float(value) / 2.0)) * 2)


def smart_1080p_target(width, height):
    """Return an even-sized target with a 1080px short edge."""
    width, height = int(width), int(height)
    if width <= 0 or height <= 0:
        raise RequestError("视频画幅宽高必须为正整数")
    if width >= height:
        return _even(1080 * width / height), 1080
    return 1080, _even(1080 * height / width)


def resolve_smart_1080p_plan(backend, duration, total_vram_gb, free_vram_gb):
    """Return a concrete safe preset; never return SMART_PRESET itself."""
    if backend not in {"fl2va_model", "ref2va_model"}:
        raise RequestError(f"免费智能 1080p 无法识别实际后端：{backend}")
    duration = int(duration)
    total_vram_gb = float(total_vram_gb)
    free_vram_gb = float(free_vram_gb)
    low_vram = total_vram_gb <= LOW_VRAM_TOTAL_GB
    if low_vram and free_vram_gb < LOW_VRAM_MIN_FREE_GB:
        raise RequestError(
            f"当前空闲显存 {free_vram_gb:.1f}GB，低显存 1080p 至少需要 "
            f"{LOW_VRAM_MIN_FREE_GB:.1f}GB；请关闭其他任务、等待模型卸载或重启 ComfyUI。"
        )
    if low_vram and not 4 <= duration <= LOW_VRAM_MAX_SECONDS:
        raise RequestError(
            f"低显存 1080p 请求 {duration} 秒，但当前档位最多 {LOW_VRAM_MAX_SECONDS} 秒；"
            f"总显存 {total_vram_gb:.1f}GB，空闲显存 {free_vram_gb:.1f}GB。"
            "请缩短时长或拆成多段。"
        )
    preset = (
        "low_vram" if backend == "ref2va_model" else "low_vram_two_stage"
    ) if low_vram else (
        "quality_sage" if backend == "ref2va_model" else "fast_4step"
    )
    return {
        "performance_preset": preset,
        "postprocess_mode": "ai_upscale",
        "ai_upscale_model": SMART_UPSCALE_MODEL,
        "motion_smoothing": "off",
        "use_easycache": False,
        "low_vram": low_vram,
        "max_duration": LOW_VRAM_MAX_SECONDS if low_vram else 15,
        "two_stage_route": "trained_latent_fl" if preset == "low_vram_two_stage" else "bypass",
        "warning": (
            "已启用低显存 1080p 模式。当前显存档位最多支持 6 秒；"
            "系统会降低生成阶段分辨率，并在生成后免费超分到目标 1080p 尺寸。"
            if low_vram else ""
        ),
    }
```

Return a plain dictionary containing `performance_preset`, `postprocess_mode`, `ai_upscale_model`, `motion_smoothing`, `use_easycache`, `low_vram`, `max_duration`, `two_stage_route`, and a Chinese `warning`. Reject an unknown backend rather than guessing.

- [ ] **Step 4: Run the policy tests**

Run: `python -m pytest tests/test_smart_1080p.py -q`

Expected: PASS.

- [ ] **Step 5: Commit the policy module**

```bash
git add nodes/smart_1080p.py tests/test_smart_1080p.py
git commit -m "feat: add smart free 1080p policy"
```

## Task 2: Expose the preset and close unsafe reference routes

**Files:**
- Modify: `nodes/schema.py`
- Modify: `tests/test_schema.py`
- Modify: `tests/test_two_stage.py`

- [ ] **Step 1: Write failing schema tests**

Add coverage for all of these assertions:

```python
def test_smart_free_1080p_is_the_public_default_for_every_route():
    assert normalize_request({"mode": "T2VA"})["performance_preset"] == "smart_free_1080p"
    for mode, voice in [("T2VA", "none"), ("FL2VA", "none"), ("REF2VA", "none"), ("T2VA", "h3_reference"), ("T2VA", "fish_lock")]:
        assert "smart_free_1080p" in allowed_performance_presets(mode, voice)


def test_smart_route_is_locked_to_one_local_ai_upscale_and_no_rife():
    assert allowed_postprocess_modes("smart_free_1080p") == ("ai_upscale",)
    assert allowed_motion_smoothing("smart_free_1080p", "ai_upscale") == ("off",)


@pytest.mark.parametrize("unsafe", ["quality_two_stage", "low_vram_two_stage", "reference_fast", "fl_quality_fast_v4"])
def test_legacy_unsafe_reference_presets_fall_back_with_warning(unsafe):
    request = normalize_request({
        "mode": "REF2VA",
        "references": [object()],
        "performance_preset": unsafe,
        "postprocess_mode": "ai_upscale" if unsafe == "low_vram_two_stage" else "native",
    })
    expected = "low_vram" if unsafe == "low_vram_two_stage" else "quality_sage"
    assert request["performance_preset"] == expected
    assert any(unsafe in item and expected in item for item in request["warnings"])
```

Retain explicit `ref_fast_4step` and legacy `fast_4step` as opt-in official REF Turbo routes. Update the existing tests that currently expect `quality_two_stage`, `low_vram_two_stage`, or `reference_fast` to remain valid on a reference backend.

- [ ] **Step 2: Run focused tests and confirm failures**

Run: `python -m pytest tests/test_schema.py tests/test_two_stage.py -q`

Expected: FAIL because the alias/default/compatibility mapping does not exist and unsafe REF presets remain allowed.

- [ ] **Step 3: Add the public alias and safe compatibility resolver**

In `nodes/schema.py`:

```python
PERFORMANCE_PRESETS = {
    "免费智能 1080p": "smart_free_1080p",
    "smart_free_1080p": "smart_free_1080p",
    # existing entries remain loadable
}

REFERENCE_UNSAFE_FALLBACKS = {
    "quality_two_stage": "quality_sage",
    "low_vram_two_stage": "low_vram",
    "reference_fast": "quality_sage",
    "fl_quality_fast_v4": "quality_sage",
}
```

Make `免费智能 1080p` the first `USER_PERFORMANCE_PRESET_LABELS` item and set `DEFAULT_REQUEST` to `performance_preset="smart_free_1080p"`, `postprocess_mode="ai_upscale"`, `ai_upscale_model="RealESRGAN_x2plus.pth"`, and `motion_smoothing="off"`.

Put `smart_free_1080p` in all route lists. The reference list must contain only `smart_free_1080p`, `quality`, `quality_sage`, `ref_quality_native`, `ref_fast_4step`, legacy `fast_4step`, `low_vram`, and `custom`. Apply `REFERENCE_UNSAFE_FALLBACKS` after `resolved_backend` is known and before postprocess compatibility is finalized, so old serialized workflows load safely and get an explicit warning instead of failing on their old postprocess value.

- [ ] **Step 4: Run focused schema tests**

Run: `python -m pytest tests/test_schema.py tests/test_two_stage.py -q`

Expected: PASS.

- [ ] **Step 5: Commit the schema change**

```bash
git add nodes/schema.py tests/test_schema.py tests/test_two_stage.py
git commit -m "feat: expose safe smart 1080p preset"
```

## Task 3: Resolve the concrete route in Director and performance nodes

**Files:**
- Modify: `nodes/director.py`
- Modify: `nodes/performance.py`
- Modify: `nodes/two_stage_assets.py`
- Modify: `tests/test_director.py`
- Modify: `tests/test_performance.py`

- [ ] **Step 1: Add failing Director route tests**

Use the existing Director build fixture/helper style and supply every required `build()` argument exactly as nearby tests do. Mock `_cuda_memory_gb()` and `_available_upscale_models()`; cover these exact assertions:

- `test_smart_base_24gb_resolves_official_turbo_and_x2`: T2VA at 24GB total/20GB free keeps `requested_performance_preset == "smart_free_1080p"`, resolves `performance_preset == "fast_4step"`, `postprocess_path == "ai_upscale"`, `target_width/height == 1920/1080`, `ai_upscale_model == "RealESRGAN_x2plus.pth"`, and `motion_smoothing == "off"`.
- `test_smart_reference_24gb_resolves_twenty_step_sage_without_cache`: REF2VA resolves `quality_sage`; `acceleration_plan(guide)` reports 20 steps, Sage enabled, cache disabled, and `two_stage_route == "bypass"`.
- `test_smart_reference_8gb_uses_single_sample_low_vram_not_trained_ref`: REF2VA at 8GB total/7GB free resolves `low_vram`, `resolved_two_stage_route == "bypass"`, and no required asset contains `trained_latent_ref`.
- `test_smart_low_vram_rejects_seven_seconds_before_model_probe`: the same complete build input with duration 7 raises `RequestError` containing `最多 6 秒`; the mocked upscale/model probes remain uncalled.

Also test `1:1`, `21:9`, and portrait target dimensions, and verify an explicit old preset still retains its old requested target instead of being silently converted to smart mode.

- [ ] **Step 2: Run focused tests and confirm failures**

Run: `python -m pytest tests/test_director.py tests/test_performance.py -q`

Expected: FAIL because `smart_free_1080p` is not resolved to a concrete performance plan.

- [ ] **Step 3: Resolve smart mode before native/two-stage planning**

In `MiniMaxH3DirectorPlus.build`, immediately after `normalize_request(...)`:

1. Save `requested_performance_preset`.
2. If it is `smart_free_1080p`, call `_cuda_memory_gb()` once and pass the values plus `request["resolved_backend"]` and duration to `resolve_smart_1080p_plan`.
3. Replace only the internal execution fields with the returned concrete plan.
4. Compute the final target with `smart_1080p_target(custom/aspect width, custom/aspect height)`; do not trust a sub-1080 named resolution while smart mode is active.
5. Resolve `RealESRGAN_x2plus.pth` before H3 sampling. If absent, raise: `免费智能 1080p 需要 RealESRGAN_x2plus.pth，请放入 ComfyUI/models/upscale_models。`
6. Append the low-VRAM explanation from the policy to `warnings`.

The resulting guide must keep both fields:

```python
"requested_performance_preset": requested_performance_preset,
"performance_preset": request["performance_preset"],
```

Do not add `smart_free_1080p` to `performance.PRESETS`; every guide reaching a sampler must already contain a concrete preset. Add a defensive error in `preset_values` if the unresolved smart name arrives there.

- [ ] **Step 4: Make trained-route resolution backend-safe**

In `nodes/two_stage_assets.py`, reject any request to resolve `trained_latent_ref` from `low_vram_two_stage`. In `nodes/performance.py`, assert that a `ref2va_model` guide with either two-stage preset cannot reach `acceleration_plan`; return a clear error naming the safe replacements (`quality_sage`, `low_vram`, or `ref_fast_4step`). Keep Fish on the same rule.

- [ ] **Step 5: Run focused route tests**

Run: `python -m pytest tests/test_director.py tests/test_performance.py -q`

Expected: PASS.

- [ ] **Step 6: Commit route integration**

```bash
git add nodes/director.py nodes/performance.py nodes/two_stage_assets.py tests/test_director.py tests/test_performance.py
git commit -m "feat: route smart 1080p by backend and vram"
```

## Task 4: Guard decoded frames and guarantee one AI reconstruction pass

**Files:**
- Modify: `nodes/upscale.py`
- Modify: `nodes/stream_output.py`
- Modify: `tests/test_upscale.py`
- Modify: `tests/test_stream_output.py`

- [ ] **Step 1: Write failing structural-validation tests**

```python
@pytest.mark.parametrize(
    "images",
    [
        torch.full((4, 16, 16, 3), float("nan")),
        torch.full((4, 16, 16, 3), float("inf")),
        torch.zeros((4, 16, 16, 3)),
        torch.full((4, 16, 16, 3), 0.5),
    ],
)
def test_invalid_decoded_video_is_rejected_before_ai_upscale(images):
    with pytest.raises(ValueError, match="生成阶段输出无效，请重新生成"):
        validate_frames_for_reconstruction(images)


def test_low_saturation_but_structured_video_is_valid():
    ramp = torch.linspace(0.1, 0.9, 16).view(1, 1, 16, 1).repeat(4, 16, 1, 3)
    validate_frames_for_reconstruction(ramp)
```

Include wrong rank, wrong channel count, zero frames, and dimensions below 8px. The test must establish that saturation is not inspected.

- [ ] **Step 2: Write a failing single-pass streaming test**

Monkeypatch the AI-upscale iterator/model loader, enable `save_first_frame=True` and `save_last_frame=True`, and assert:

```python
assert upscale_model_loads == 1
assert ai_reconstruction_iterators == 1
assert exported_first.shape[-3:-1] == (1080, 1920)
assert exported_last.shape[-3:-1] == (1080, 1920)
```

This test must fail on the current behavior because `_save_ai_upscale_frame` reconstructs edge frames again after the main video pass.

- [ ] **Step 3: Run focused tests and confirm failures**

Run: `python -m pytest tests/test_upscale.py tests/test_stream_output.py -q`

Expected: FAIL on missing validation and repeated edge-frame upscale.

- [ ] **Step 4: Implement the frame guard**

Add to `nodes/upscale.py`:

```python
def validate_frames_for_reconstruction(images, minimum_dynamic_range=1.0 / 1024.0):
    """Reject malformed/non-finite/near-constant BHWC video without judging style."""
```

Require a finite four-dimensional BHWC tensor, at least one frame, height/width >= 8, and channels in `{3, 4}`. Compare the global finite max/min only; do not calculate HSV saturation. Log shape, dtype, min, max, and finite status on failure without dumping frame pixels.

Call it in `MiniMaxH3StreamingVideoCombine.combine` after resolving `postprocess_path` and before `resolve_upscale_model_name`, model release, output-path creation, or encoder startup, only when the path is `ai_upscale`.

- [ ] **Step 5: Reuse reconstructed edge frames**

Wrap the one `_iter_ai_upscale_frame_chunks(...)` iterator with a small collector that stores CPU copies of the first and final reconstructed frame as chunks pass to the encoder. Pass those cached frames to the existing PNG writer. Remove `_save_ai_upscale_frame` calls from the post-encode first/last export branch; do not retain the whole video in memory.

Keep audio muxing, frame count, frame rate, duration, H.264/MP4 selection, and center-crop/downscale behavior unchanged. The RealESRGAN model may be loaded once and the reconstructed stream may be consumed once.

- [ ] **Step 6: Run output tests**

Run: `python -m pytest tests/test_upscale.py tests/test_stream_output.py -q`

Expected: PASS.

- [ ] **Step 7: Commit output safety**

```bash
git add nodes/upscale.py nodes/stream_output.py tests/test_upscale.py tests/test_stream_output.py
git commit -m "fix: validate frames and reuse one upscale pass"
```

## Task 5: Add tested upload removal and slot compaction

**Files:**
- Create: `js/media_slot_state.mjs`
- Create: `tests/js/media_slot_state.test.mjs`
- Create: `tests/test_frontend_state.py`
- Modify: `js/minimax_h3_director_plus_v9.js`
- Modify: `tests/test_frontend_source.py`

- [ ] **Step 1: Write browser-independent failing state tests**

In `tests/js/media_slot_state.test.mjs`, use `node:test` and `node:assert/strict`:

```javascript
import test from "node:test";
import assert from "node:assert/strict";
import { clearSlot, compactSlots, compactBoundSlots } from "../../js/media_slot_state.mjs";

test("clearing one endpoint leaves the other endpoint unchanged", () => {
  assert.deepEqual(clearSlot(["first.png", "last.png"], 0), ["", "last.png"]);
});

test("removing a middle REF image compacts later pictures", () => {
  assert.deepEqual(compactSlots(["a.png", "b.png", "c.png", ""], 1), ["a.png", "c.png", "", ""]);
});

test("audio filename and role name move as one unit", () => {
  assert.deepEqual(
    compactBoundSlots(["a.wav", "b.wav", "c.wav"], ["S1", "S2", "S3"], 1),
    { files: ["a.wav", "c.wav", ""], names: ["S1", "S3", ""] },
  );
});
```

In `tests/test_frontend_state.py`, run `node --test tests/js/media_slot_state.test.mjs` with `subprocess.run(..., check=False, capture_output=True, text=True)`, skip only when `node` is absent, and assert return code zero.

- [ ] **Step 2: Run the state test and confirm failure**

Run: `python -m pytest tests/test_frontend_state.py -q`

Expected: FAIL because `js/media_slot_state.mjs` does not exist.

- [ ] **Step 3: Implement pure slot operations**

`js/media_slot_state.mjs` must export only pure functions. Normalize missing entries to `""`, preserve original array lengths, validate the index, and never perform network or DOM operations.

- [ ] **Step 4: Add failing DOM-source contract tests**

Extend `tests/test_frontend_source.py` to require:

- populated slots show both `更换` and `移除`;
- empty slots show `选择文件`;
- `syncUploadWidget` accepts `""` and calls the widget callback;
- remove handling never calls `/h3-director-plus/assets` with `DELETE` and contains no `unlink`/file-delete request;
- REF image fields use compaction and set the Picture-number warning;
- audio fields compact together with `voice_reference_name_1..3` and set the Audio/S warning;
- endpoint slots use independent clear, not REF compaction, outside REF2VA.

- [ ] **Step 5: Wire removal into the Director DOM**

Import the pure helpers at the top of `minimax_h3_director_plus_v9.js`. Change `syncUploadWidget` to reject only a missing widget, not an empty value:

```javascript
function syncUploadWidget(node, widgetName, value) {
  const item = widget(node, widgetName);
  if (!item) return;
  const normalized = value || "";
  // Only add non-empty uploaded filenames to combo values.
  item.value = normalized;
  item.callback?.(normalized);
  // dirty both canvases
}
```

Add `data-h3p-remove-file` buttons only for populated controls. Removal behavior:

1. REF2VA image: read `REF2VA_IMAGE_SLOTS`, compact filenames, write every affected widget, set `node._h3pAssetNotice = "参考图已重新编号，请检查提示词中的 <Picture N> 引用。"`.
2. Voice audio: compact the three audio file widgets and the three role-name widgets together, then warn about `<Audio N>` and `(Sx)`.
3. First/last endpoints outside REF2VA: clear only the selected widget and display the existing missing-required-input warning when applicable.
4. Re-render previews and mark the canvas dirty. Do not issue an HTTP request during removal.

Render the notice inside the Director section with `role="status"`. Keep the uploaded file on disk.

- [ ] **Step 6: Run frontend tests**

Run: `python -m pytest tests/test_frontend_state.py tests/test_frontend_source.py -q`

Expected: PASS.

- [ ] **Step 7: Commit asset removal**

```bash
git add js/media_slot_state.mjs js/minimax_h3_director_plus_v9.js tests/js/media_slot_state.test.mjs tests/test_frontend_state.py tests/test_frontend_source.py
git commit -m "feat: remove and compact director media slots"
```

## Task 6: Fix Director geometry without losing stable node sizing

**Files:**
- Modify: `js/minimax_h3_director_plus_v9.js`
- Modify: `tests/test_frontend_source.py`
- Modify: `tools/build_u11_workflow.py`
- Modify: `tests/test_workflow_tools.py`

- [ ] **Step 1: Replace the old fixed-width source test with failing boundary contracts**

The updated test must reject `width:${DIRECTOR_UI_WIDTH}px`, `min-width:${DIRECTOR_UI_WIDTH}px`, `max-width:none`, and `flex:0 0 ${DIRECTOR_UI_WIDTH}px` on `.h3p`. Require:

```javascript
.h3p{box-sizing:border-box;width:100%;min-width:0;max-width:100%;
```

Also require an internal viewport with `overflow-y:auto`, `overflow-x:hidden`, and a DOM widget computed width smaller than the node outer width by an explicit inset constant.

Extend workflow tests to calculate rectangles for every generated node/group and assert:

```python
assert group_left <= node_left
assert node_right <= group_right
assert group_top <= node_top
assert node_bottom <= group_bottom
assert output_group_left - director_group_right >= 40
```

- [ ] **Step 2: Run geometry tests and confirm failures**

Run: `python -m pytest tests/test_frontend_source.py tests/test_workflow_tools.py -q`

Expected: FAIL on the old hard-locked root width and any group rectangles that do not contain their nodes.

- [ ] **Step 3: Implement the contained viewport**

Keep `DIRECTOR_UI_WIDTH` and the stable outer `node.setSize` call. Add explicit `DIRECTOR_CONTENT_INSET` and `DIRECTOR_VIEWPORT_HEIGHT`; set the `.h3p` root to 100% of the DOM widget content box. Make `domWidget.computeSize()` return `[DIRECTOR_UI_WIDTH - DIRECTOR_CONTENT_INSET, DIRECTOR_VIEWPORT_HEIGHT]`, and make `getHeight()` and CSS viewport height agree. Expanded previews and Fish panels must scroll vertically inside `.h3p`, never enlarge the DOM beyond the node.

Update generated workflow group bounds so the Director node rectangle is fully contained with at least 24px on each side, and preserve at least a 40px horizontal gutter before the output group.

- [ ] **Step 4: Run automated geometry tests**

Run: `python -m pytest tests/test_frontend_source.py tests/test_workflow_tools.py -q`

Expected: PASS.

- [ ] **Step 5: Perform real-browser visual acceptance**

Start the existing ComfyUI instance, load the generated U11 workflow, and inspect the Director at 100% and approximately 53% canvas zoom with the right sidebar both closed and open. Capture screenshots and verify: content stays inside the node, vertical scrolling reaches the last control, the node stays inside its group, the output group does not overlap, and remove buttons remain clickable. If any check fails, fix CSS/geometry and rerun Step 4 before proceeding.

- [ ] **Step 6: Commit geometry fixes**

```bash
git add js/minimax_h3_director_plus_v9.js tools/build_u11_workflow.py tests/test_frontend_source.py tests/test_workflow_tools.py
git commit -m "fix: contain director ui within node bounds"
```

## Task 7: Update generated workflows, API schema, and user documentation

**Files:**
- Modify: `tools/build_u11_workflow.py`
- Modify: `tests/test_workflow_tools.py`
- Modify: `tests/test_api.py`
- Modify: `docs/使用说明.md`
- Modify: `docs/API说明.md`
- Modify: `docs/故障排查.md`
- Modify: `tests/test_docs.py`
- Modify generated workflow/template files identified by `tests/test_workflow_tools.py`

- [ ] **Step 1: Write failing workflow/API/doc tests**

Require newly built workflows and API defaults to use:

```python
"performance_preset": "免费智能 1080p"
"postprocess_mode": "ai_upscale"
"ai_upscale_model": "RealESRGAN_x2plus.pth"
"motion_smoothing": "off"
```

Require public schema to expose the smart alias, route-specific allowed presets, the 1080-short-edge rule, the 6-second low-VRAM maximum, and the unsafe REF fallback behavior. Preserve explicit old serialized choices when they are still safe.

Doc tests must look for exact concepts: “生成阶段分辨率不等于最终输出分辨率”, “本地免费”, `RealESRGAN_x2plus.pth`, “REF2VA 不使用训练型 reference latent 二采”, “最多 6 秒”, “空闲显存”, “移除只取消节点引用，不删除原文件”, and the Picture/Audio renumber warnings.

- [ ] **Step 2: Run focused tests and confirm failures**

Run: `python -m pytest tests/test_workflow_tools.py tests/test_api.py tests/test_docs.py -q`

Expected: FAIL on old defaults and missing documentation.

- [ ] **Step 3: Update builders and regenerate committed artifacts**

Change only default values and route notes; do not add a paid service or a second output node. Use the repository's existing workflow regeneration command/functions referenced at the top of `tests/test_workflow_tools.py`, then inspect `git diff --stat` and `git diff --check`. Do not hand-edit generated JSON when the builder owns it.

- [ ] **Step 4: Update the three Chinese documents**

Document the route matrix, required local model path, target dimensions, low-VRAM blocking message, old REF preset migration, structural failure error, and non-destructive upload removal. State clearly that 1080p is the final reconstructed output, not native H3 sampling.

- [ ] **Step 5: Run focused documentation/workflow tests**

Run: `python -m pytest tests/test_workflow_tools.py tests/test_api.py tests/test_docs.py -q`

Expected: PASS.

- [ ] **Step 6: Commit generated artifacts and docs**

```bash
git add tools/build_u11_workflow.py tests/test_workflow_tools.py tests/test_api.py docs/使用说明.md docs/API说明.md docs/故障排查.md tests/test_docs.py workflows api
git commit -m "docs: make smart free 1080p the default workflow"
```

Adjust the final `git add` paths to the exact generated files reported by `git status`; never add unrelated user files.

## Task 8: Full regression and real GPU acceptance

**Files:**
- Modify only if a failing test exposes an implementation defect.
- Create user-facing evidence only under the repository's existing test/output conventions; do not commit generated videos unless the repository already tracks such fixtures.

- [ ] **Step 1: Run the complete automated suite**

Run: `python -m pytest -q`

Expected: PASS with no skipped new JS state tests when Node.js is installed.

- [ ] **Step 2: Run repository hygiene checks**

Run: `git diff --check`

Expected: no output.

Run: `git status --short`

Expected: clean, except explicitly documented local GPU outputs.

- [ ] **Step 3: Execute the real base-backend matrix**

Generate 4-second and 6-second clips through at least T2VA and FL2VA smart mode. Record the resolved preset, peak/available VRAM, H3 native dimensions, final dimensions, RealESRGAN model name, super-resolution invocation count, frame count, audio duration, and second consecutive run result. Verify official Turbo 4-step, no EasyCache, one X2 pass, at least the requested 1080p target, and no memory leak across the second run.

- [ ] **Step 4: Execute the real reference-backend matrix**

Generate 4-second and 6-second REF2VA clips, plus one H3/Fish audio-reference route if assets are available. Verify 20-step Sage on normal VRAM or single-sample `low_vram` on low VRAM, never `trained_latent_ref`, never EasyCache, one X2 pass, preserved audio, non-gray output, recognizable reference identity, and a successful consecutive second run.

- [ ] **Step 5: Verify failure paths before queueing**

Test a 7-second low-VRAM request, <6GB free VRAM simulation, missing `RealESRGAN_x2plus.pth`, unsafe legacy REF two-stage input, and a near-constant frame tensor. Each must fail or safely normalize with the exact Chinese guidance specified above; none may silently lower final resolution or call another upscaler.

- [ ] **Step 6: Verify Director removal manually**

Upload three REF pictures and three audio samples with role names. Remove slot 2 from each group; confirm compaction and warnings. Clear FL2VA first and last endpoints independently; confirm the missing-input warning. Confirm the original files still exist in ComfyUI's input directory.

- [ ] **Step 7: Fix, retest, and make a final verification commit only if needed**

For any acceptance defect, add a reproducing automated test first, implement the smallest fix, rerun the focused test and `python -m pytest -q`, then commit with a defect-specific message. Do not mark the work complete while any real GPU or browser acceptance item is unverified.

## Final self-review checklist

- [ ] Every success criterion in `docs/superpowers/specs/2026-08-28-smart-free-1080p-and-asset-removal-design.md` maps to at least one automated or manual acceptance step above.
- [ ] No `TODO`, placeholder function body, ellipsis, or invented path remains in implementation code.
- [ ] `smart_free_1080p` is public/requested state only; sampler code receives a concrete preset.
- [ ] `resolved_backend`, not the visible mode label, selects the acceleration route.
- [ ] No reference backend can resolve `trained_latent_ref` from smart or low-VRAM operation.
- [ ] The final output invokes one local X2 reconstruction and no RTX VSR/RIFE/second upscaler.
- [ ] Media removal clears widget serialization and callbacks but never deletes disk files.
- [ ] Python annotations, guide keys, JS widget names, API enum values, and Chinese labels are consistent across implementation, tests, builders, and docs.
- [ ] The full suite, browser geometry check, and available real-GPU matrix have recorded results before completion is claimed.
