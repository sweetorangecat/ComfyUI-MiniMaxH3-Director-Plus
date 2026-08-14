# Mode Performance Isolation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restrict U11 performance presets by mode and voice-reference route while keeping legacy workflows executable.

**Architecture:** Add one route-aware preset matrix in `nodes/schema.py`. Request normalization and the performance node use that matrix for validation and safe fallback. The Director Plus custom UI derives its visible preset options from the same rules using mode and voice mode, while the serialized native widget remains backward-compatible.

**Tech Stack:** Python, pytest, ComfyUI custom-node schema, vanilla JavaScript Director Plus frontend.

---

### Task 1: Add failing route-matrix tests

**Files:**
- Modify: `D:\ComfyUI_windows_portable-G313\ComfyUI\custom_nodes\ComfyUI-MiniMaxH3-Director-Plus\tests\test_schema.py`

- [ ] **Step 1: Add tests for route-specific visible presets and safe defaults.**

```python
@pytest.mark.parametrize(
    ("mode", "voice_mode", "expected"),
    [
        ("T2VA", "none", ("quality", "low_vram")),
        ("I2VA", "none", ("quality", "fast_4step", "low_vram")),
        ("FL2VA", "none", ("quality", "fast_4step", "low_vram")),
        ("L2VA", "none", ("quality", "fast_4step", "low_vram")),
        ("REF2VA", "none", ("quality", "reference_fast", "fast_4step", "low_vram")),
        ("I2VA", "h3_reference", ("quality", "reference_fast", "fast_4step", "low_vram")),
        ("FL2VA", "fish_lock", ("quality", "reference_fast", "fast_4step", "low_vram")),
        ("L2VA", "h3_reference", ("quality", "reference_fast", "fast_4step", "low_vram")),
    ],
)
def test_allowed_performance_presets_follow_mode_and_voice(mode, voice_mode, expected):
    assert allowed_performance_presets(mode, voice_mode) == expected


def test_invalid_t2va_reference_acceleration_falls_back_with_warning():
    request = normalize_request({"mode": "T2VA", "performance_preset": "reference_fast"})
    assert request["performance_preset"] == "quality"
    assert any("T2VA" in warning and "稳定质量" in warning for warning in request["warnings"])


def test_voice_reference_uses_ref2va_performance_set_even_in_fl2va():
    request = normalize_request({
        "mode": "FL2VA",
        "voice_mode": "h3_reference",
        "voice_reference_audio": object(),
    })
    assert request["resolved_backend"] == "ref2va_model"
    assert "reference_fast" in allowed_performance_presets("FL2VA", "h3_reference")
```

- [ ] **Step 2: Run the focused tests and verify they fail because the helper is missing.**

Run: `pytest tests/test_schema.py -k "allowed_performance or invalid_t2va or voice_reference_uses" -q`

Expected: FAIL with an import or attribute error for `allowed_performance_presets`.

### Task 2: Implement the shared matrix and normalization fallback

**Files:**
- Modify: `D:\ComfyUI_windows_portable-G313\ComfyUI\custom_nodes\ComfyUI-MiniMaxH3-Director-Plus\nodes\schema.py`

- [ ] **Step 1: Add the route matrix and helper.**

```python
PERFORMANCE_PRESETS_BY_ROUTE = {
    "t2va": ("quality", "low_vram"),
    "endpoint": ("quality", "fast_4step", "low_vram"),
    "reference": ("quality", "reference_fast", "fast_4step", "low_vram"),
}


def allowed_performance_presets(mode, voice_mode="none"):
    if voice_mode != "none" or mode == "REF2VA":
        return PERFORMANCE_PRESETS_BY_ROUTE["reference"]
    if mode == "T2VA":
        return PERFORMANCE_PRESETS_BY_ROUTE["t2va"]
    return PERFORMANCE_PRESETS_BY_ROUTE["endpoint"]
```

- [ ] **Step 2: Normalize invalid combinations after backend resolution.**

```python
    allowed = allowed_performance_presets(request["mode"], request["voice_mode"])
    if request["performance_preset"] not in allowed:
        requested = request["performance_preset"]
        request["performance_preset"] = "quality"
        request["warnings"].append(
            f"{request['mode']} / {request['voice_mode']} 不支持性能预设 {requested}，已自动切换为稳定质量。"
        )
```

Initialize `request["warnings"]` before this fallback block, then compute `resolved_backend`. Internal `custom` remains accepted when supplied by legacy/API callers, but the normal matrix never advertises it.

- [ ] **Step 3: Run the focused schema tests and verify they pass.**

Run: `pytest tests/test_schema.py -k "allowed_performance or invalid_t2va or voice_reference_uses" -q`

Expected: PASS.

### Task 3: Make the performance node honor the same route

**Files:**
- Modify: `D:\ComfyUI_windows_portable-G313\ComfyUI\custom_nodes\ComfyUI-MiniMaxH3-Director-Plus\nodes\performance.py`
- Modify: `D:\ComfyUI_windows_portable-G313\ComfyUI\custom_nodes\ComfyUI-MiniMaxH3-Director-Plus\tests\test_performance.py`

- [ ] **Step 1: Add a failing test that a stale T2VA reference preset returns quality steps and no cache.**

```python
def test_performance_node_safely_downgrades_invalid_t2va_reference_preset():
    result = MiniMaxH3PerformancePreset().apply({
        "mode": "T2VA",
        "voice_mode": "none",
        "performance_preset": "reference_fast",
    })
    assert result[0] == 20
    assert result[1] is False
    assert result[2] is False
    assert "稳定质量" in result[3]
```

- [ ] **Step 2: Run the new test and verify it fails because the node currently returns 6-step cache acceleration.**

Run: `pytest tests/test_performance.py::test_performance_node_safely_downgrades_invalid_t2va_reference_preset -q`

Expected: FAIL with `assert 6 == 20` or equivalent.

- [ ] **Step 3: Import the matrix helper and downgrade stale guide values before `preset_values`.**

The node should use `allowed_performance_presets(guide.get("mode", "FL2VA"), guide.get("voice_mode", "none"))`. If the normalized preset is not allowed, use `quality` for `values` and return a Chinese fallback description. Valid REF2VA and voice-reference guides keep their existing acceleration behavior.

- [ ] **Step 4: Run the performance test file.**

Run: `pytest tests/test_performance.py -q`

Expected: PASS.

### Task 4: Filter the Director Plus preset control by mode and voice

**Files:**
- Modify: `D:\ComfyUI_windows_portable-G313\ComfyUI\custom_nodes\ComfyUI-MiniMaxH3-Director-Plus\js\minimax_h3_director_plus_v9.js`
- Modify: `D:\ComfyUI_windows_portable-G313\ComfyUI\custom_nodes\ComfyUI-MiniMaxH3-Director-Plus\tests\test_frontend_source.py`

- [ ] **Step 1: Add source assertions for a route-aware preset list and automatic safe fallback.**

```python
def test_frontend_filters_performance_presets_by_mode_and_voice():
    source = FRONTEND.read_text(encoding="utf-8")
    assert "allowedPerformancePresets" in source
    assert "voiceMode !== \"none\"" in source
    assert "setWidget(node, \"performance_preset\", \"稳定质量\")" in source
```

- [ ] **Step 2: Run the source test and verify it fails.**

Run: `pytest tests/test_frontend_source.py -k performance -q`

Expected: FAIL because the helper and fallback are not present.

- [ ] **Step 3: Add a frontend `allowedPerformancePresets(mode, voiceMode)` helper matching the Python matrix.**

Use the route-aware list when rendering `valueControl("性能预设", ...)`. Before rendering, if the current widget value is not in the list, call `setWidget(node, "performance_preset", "稳定质量")`; keep the existing render callback flow so changing mode or voice mode refreshes the control without changing node geometry.

- [ ] **Step 4: Run frontend source tests and the complete unit suite.**

Run: `pytest tests/test_frontend_source.py -q` and then `pytest -q`.

Expected: PASS with no new warnings.

### Task 5: Rebuild, validate, commit, and push

**Files:**
- Regenerate: `D:\ComfyUI_windows_portable-G313\ComfyUI\user\default\workflows\minimaxH3\U11-MiniMaxH3-导演台Plus-中文增强版.json`

- [ ] **Step 1: Rebuild the single U11 workflow with the existing builder.**

Run: `python tools/build_u11_workflow.py` from the custom-node repository.

- [ ] **Step 2: Validate the generated workflow.**

Run: `python tools/validate_workflow.py "D:\ComfyUI_windows_portable-G313\ComfyUI\user\default\workflows\minimaxH3\U11-MiniMaxH3-导演台Plus-中文增强版.json"`.

Expected: validation succeeds, required custom nodes are present, and only one U11 workflow is regenerated.

- [ ] **Step 3: Run the complete test suite once more.**

Run: `pytest -q`.

- [ ] **Step 4: Commit the implementation and workflow.**

```bash
git add nodes/schema.py nodes/performance.py js/minimax_h3_director_plus_v9.js tests docs/superpowers/plans
git commit -m "fix: isolate H3 performance presets by route"
```

- [ ] **Step 5: Push the implementation to the configured remote main branch.**

Run: `git push origin main`.
