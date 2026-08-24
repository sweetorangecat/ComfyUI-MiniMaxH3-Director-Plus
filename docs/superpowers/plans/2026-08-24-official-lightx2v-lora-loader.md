# 官方 LightX2V LoRA 加载修复 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 U11 通过 ComfyUI 内置加载器应用官方 LightX2V LoRA，并在采样前阻止零补丁或加载失败的假加速任务。

**Architecture:** `nodes/performance.py` 保留现有模型路径解析入口，但将官方 LoRA 的加载实现切换到 `LoraLoaderModelOnly`。加载前后统计 `ModelPatcher.patches` 的补丁条目总数；新增条目为零或无法验证时抛出专用异常，模型路由记录中文原因后继续向上抛出，使任务在采样前停止。

**Tech Stack:** Python 3.12、ComfyUI `ModelPatcher` / `LoraLoaderModelOnly`、pytest、Git。

---

## 文件职责

- `nodes/performance.py`：官方 LoRA 加载、补丁计数、专用错误和加速路由失败策略。
- `tests/test_performance.py`：内置加载器契约、补丁增量验证及采样前失败行为。
- `docs/使用说明.md`：中文界面用户可见的加载成功日志与失败处理说明。
- `docs/API说明.md`：API 调用方可依赖的失败契约，字段保持不变。

### Task 1: 用内置加载器替换错误的 LoRA 注入器

**Files:**
- Modify: `tests/test_performance.py:17-111`
- Modify: `nodes/performance.py:86-119`

- [ ] **Step 1: 写入内置加载和补丁增量的失败测试**

将现有三个“优先使用 H3 专用注入器”的测试替换为以下契约测试：

```python
def test_official_lora_uses_core_loader_and_accepts_positive_patch_delta(monkeypatch):
    import sys

    class Model:
        def __init__(self, patches):
            self.patches = patches

    base_model = Model({})
    patched_model = Model({"diffusion_model.blocks.0.attn.qkv_proj.weight": [("lora",)]})
    calls = []

    class CoreLoader:
        def load_lora_model_only(self, model, name, strength):
            calls.append((model, name, strength))
            return (patched_model,)

    monkeypatch.setitem(
        sys.modules,
        "nodes",
        type("CoreNodes", (), {"LoraLoaderModelOnly": CoreLoader})(),
    )
    monkeypatch.setattr(
        performance,
        "resolve_registered_model_name",
        lambda category, name: "minimax/adapter.safetensors",
    )
    monkeypatch.setattr(
        performance,
        "_turbo_class",
        lambda _name: (_ for _ in ()).throw(AssertionError("旧注入器不得用于官方 LoRA")),
    )

    result = performance._load_lightx2v_lora(
        base_model,
        "adapter.safetensors",
        strength=0.75,
    )

    assert result is patched_model
    assert calls == [(base_model, "minimax/adapter.safetensors", 0.75)]


def test_official_lora_rejects_zero_patch_delta(monkeypatch):
    import sys

    class Model:
        patches = {}

    model = Model()

    class CoreLoader:
        def load_lora_model_only(self, model, name, strength):
            return (model,)

    monkeypatch.setitem(
        sys.modules,
        "nodes",
        type("CoreNodes", (), {"LoraLoaderModelOnly": CoreLoader})(),
    )
    monkeypatch.setattr(
        performance,
        "resolve_registered_model_name",
        lambda category, name: name,
    )

    with pytest.raises(performance.H3LoRAApplicationError, match="未应用任何模型补丁"):
        performance._load_lightx2v_lora(model, "adapter.safetensors")


def test_official_lora_wraps_core_loader_failure(monkeypatch):
    import sys

    class Model:
        patches = {}

    class CoreLoader:
        def load_lora_model_only(self, model, name, strength):
            raise KeyError("unsupported key layout")

    monkeypatch.setitem(
        sys.modules,
        "nodes",
        type("CoreNodes", (), {"LoraLoaderModelOnly": CoreLoader})(),
    )
    monkeypatch.setattr(
        performance,
        "resolve_registered_model_name",
        lambda category, name: name,
    )

    with pytest.raises(performance.H3LoRAApplicationError, match="官方 H3 Turbo LoRA 加载失败"):
        performance._load_lightx2v_lora(Model(), "adapter.safetensors")
```

- [ ] **Step 2: 运行测试并确认旧实现失败**

Run:

```powershell
D:\ComfyUI_windows_portable-G313\python_embeded\python.exe -m pytest tests/test_performance.py -k "official_lora" -v
```

Expected: FAIL；旧实现仍调用 `_turbo_class("MiniMaxH3TurboLoRA")`，且不存在 `H3LoRAApplicationError`。

- [ ] **Step 3: 实现专用异常、补丁条目计数和内置加载器**

在 `PRESET_LABELS` 后加入：

```python
class H3LoRAApplicationError(RuntimeError):
    """Raised when an official H3 LoRA cannot be proven active."""


def _patch_entry_count(model):
    patches = getattr(model, "patches", None)
    if not isinstance(patches, dict):
        raise H3LoRAApplicationError("无法读取模型补丁，不能验证官方 H3 Turbo LoRA 是否生效")
    total = 0
    for entries in patches.values():
        if isinstance(entries, (list, tuple)):
            total += len(entries)
        elif entries is not None:
            total += 1
    return total
```

将 `_load_lightx2v_lora()` 替换为：

```python
def _load_lightx2v_lora(model, lora_name=None, strength=1.0, low_vram=False):
    """Apply an official LightX2V LoRA through ComfyUI's core loader."""
    del low_vram  # Core ModelPatcher loading follows ComfyUI's active VRAM policy.
    requested_lora_name = lora_name or TURBO_LORA_NAME
    resolved_lora_name = resolve_registered_model_name("loras", requested_lora_name)
    if resolved_lora_name is None:
        raise H3LoRAApplicationError(f"缺少 H3 Turbo LoRA: {requested_lora_name}")
    if resolved_lora_name != requested_lora_name:
        LOGGER.info("[H3 LoRA] resolved %s -> %s", requested_lora_name, resolved_lora_name)

    before = _patch_entry_count(model)
    try:
        import nodes as comfy_nodes
        patched_model = comfy_nodes.LoraLoaderModelOnly().load_lora_model_only(
            model,
            resolved_lora_name,
            float(strength),
        )[0]
    except H3LoRAApplicationError:
        raise
    except (AttributeError, ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise H3LoRAApplicationError(
            f"官方 H3 Turbo LoRA 加载失败: {resolved_lora_name}: {exc}"
        ) from exc

    after = _patch_entry_count(patched_model)
    patch_delta = after - before
    if patch_delta <= 0:
        raise H3LoRAApplicationError(
            f"官方 H3 Turbo LoRA 未应用任何模型补丁: {resolved_lora_name}"
        )
    LOGGER.info(
        "[H3 LoRA] 官方加载成功 name=%s strength=%.2f patch_delta=%s",
        resolved_lora_name,
        float(strength),
        patch_delta,
    )
    return patched_model
```

- [ ] **Step 4: 运行新契约测试并确认通过**

Run:

```powershell
D:\ComfyUI_windows_portable-G313\python_embeded\python.exe -m pytest tests/test_performance.py -k "official_lora" -v
```

Expected: `3 passed`。

- [ ] **Step 5: 提交并推送第一阶段**

```powershell
git add nodes/performance.py tests/test_performance.py
git commit -m "fix: load official LightX2V LoRAs with core loader"
git push origin main
```

### Task 2: 零补丁错误必须在采样前停止

**Files:**
- Modify: `tests/test_performance.py`
- Modify: `nodes/performance.py:514-611`

- [ ] **Step 1: 写入单采和二采的失败关闭测试**

```python
def test_fast_route_stops_when_official_lora_has_no_verified_patches(monkeypatch):
    error = performance.H3LoRAApplicationError("官方 LoRA 未应用任何模型补丁")
    monkeypatch.setattr(
        performance,
        "_load_lightx2v_lora",
        lambda *args, **kwargs: (_ for _ in ()).throw(error),
    )
    guide = {
        "mode": "FL2VA",
        "performance_preset": "fast_4step",
        "resolved_backend": "fl2va_model",
    }

    with pytest.raises(performance.H3LoRAApplicationError, match="未应用任何模型补丁"):
        MiniMaxH3AccelerationRouter().apply(object(), guide)

    assert guide["turbo_lora_applied"] is False
    assert "未应用任何模型补丁" in guide["turbo_lora_error"]


def test_two_stage_route_stops_when_official_lora_has_no_verified_patches(monkeypatch):
    error = performance.H3LoRAApplicationError("官方 LoRA 未应用任何模型补丁")
    monkeypatch.setattr(
        performance,
        "_load_lightx2v_lora",
        lambda *args, **kwargs: (_ for _ in ()).throw(error),
    )
    guide = {
        "mode": "T2VA",
        "performance_preset": "quality_two_stage",
        "resolved_backend": "fl2va_model",
    }

    with pytest.raises(performance.H3LoRAApplicationError, match="未应用任何模型补丁"):
        MiniMaxH3AccelerationRouter().apply(object(), guide)

    assert guide["turbo_lora_applied"] is False
    assert guide["two_stage_enabled"] is False
    assert "未应用任何模型补丁" in guide["turbo_lora_error"]
```

- [ ] **Step 2: 运行测试并确认当前路由静默旁路**

Run:

```powershell
D:\ComfyUI_windows_portable-G313\python_embeded\python.exe -m pytest tests/test_performance.py -k "route_stops_when_official_lora" -v
```

Expected: FAIL；当前宽泛异常处理会返回基础模型而不是停止任务。

- [ ] **Step 3: 让专用错误记录状态后继续向上抛出**

在 `_apply_two_stage_models()` 的宽泛异常分支之前加入：

```python
    except H3LoRAApplicationError as exc:
        guide["turbo_lora_applied"] = False
        guide["head_chunking_applied"] = False
        guide["two_stage_enabled"] = False
        guide["turbo_lora_error"] = str(exc)
        LOGGER.error("[H3 two-stage models] official LoRA verification failed: %s", exc)
        raise
```

在 `_apply_acceleration()` 的 Turbo LoRA 宽泛异常分支之前加入：

```python
        except H3LoRAApplicationError as exc:
            guide["turbo_lora_applied"] = False
            guide["turbo_lora_error"] = str(exc)
            LOGGER.error("[H3 acceleration] official LoRA verification failed: %s", exc)
            raise
```

- [ ] **Step 4: 运行路由测试和性能模块测试**

Run:

```powershell
D:\ComfyUI_windows_portable-G313\python_embeded\python.exe -m pytest tests/test_performance.py -k "official_lora or route_stops_when_official_lora" -v
D:\ComfyUI_windows_portable-G313\python_embeded\python.exe -m pytest tests/test_performance.py -v
```

Expected: 新增 5 项契约测试全部 PASS，`tests/test_performance.py` 全部 PASS。

- [ ] **Step 5: 提交并推送失败关闭行为**

```powershell
git add nodes/performance.py tests/test_performance.py
git commit -m "fix: stop H3 sampling when official LoRA is inactive"
git push origin main
```

### Task 3: 中文说明与完整回归验证

**Files:**
- Modify: `docs/使用说明.md:25-41`
- Modify: `docs/API说明.md:17`

- [ ] **Step 1: 补充用户说明**

在 `docs/使用说明.md` 的性能模式说明后加入：

```markdown
官方 LightX2V LoRA 由 ComfyUI 内置加载器应用。启动采样前会核对实际新增的模型补丁数；成功日志包含 `patch_delta=<正整数>`。如果补丁数为零或加载器不兼容，任务会立即显示中文错误并停止，不会把基础模型按 4/8 步继续生成。
```

在 `docs/API说明.md` 的 `performance_preset` 说明后加入：

```markdown
API 字段没有变化。选择依赖官方 LightX2V LoRA 的性能档位时，后端只有在模型补丁增量大于零后才进入采样；否则请求在采样前失败，错误信息包含 LoRA 文件名和失败原因。
```

- [ ] **Step 2: 执行完整验证**

Run:

```powershell
D:\ComfyUI_windows_portable-G313\python_embeded\python.exe -m pytest -q
D:\ComfyUI_windows_portable-G313\python_embeded\python.exe -m compileall -q nodes tests
git diff --check
```

Expected: 全量 pytest PASS；`compileall` 和 `git diff --check` 退出码均为 0。

- [ ] **Step 3: 提交文档并推送主分支**

```powershell
git add docs/使用说明.md docs/API说明.md
git commit -m "docs: explain verified official H3 LoRA loading"
git push origin main
```

- [ ] **Step 4: 核对远程主分支与工作树**

Run:

```powershell
$local = git rev-parse HEAD
$remote = (git ls-remote origin refs/heads/main).Split("`t")[0]
Write-Output "LOCAL=$local"
Write-Output "REMOTE=$remote"
git status --short
```

Expected: `LOCAL` 与 `REMOTE` 完全一致，`git status --short` 无输出。

