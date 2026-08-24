# RTX VSR 真实单帧前置检查 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用 NVIDIA 官方建议范围内的真实 360p 单帧取代无效的 64×64 RTX VSR 探测，消除可用服务器上的 `code -14` 假失败。

**Architecture:** `probe_vsr_capability()` 继续复用正式输出的 `VsrFrameProcessor`，但会创建 `640×360` CUDA RGB 帧，执行一次 `1280×720` 的 `load + run`，验证输出形状并在 `finally` 中显式关闭效果、解除临时输入/输出引用。若单帧其余步骤已成功但关闭失败，仍必须阻止 H3；已有主错误时关闭失败只记录警告并保留主错误。不得调用全局 `torch.cuda.empty_cache()`，以便后续 H3 或多 GPU 任务复用分配器缓存。错误提示根据 Linux/Windows 平台给出不同依赖建议；导演台、正式输出与 API 字段保持不变。

**Tech Stack:** Python 3.12、PyTorch CUDA、NVIDIA `nvidia-vfx`/`nvvfx`、pytest、Git。

---

## 文件职责

- `nodes/rtx_vsr_stream.py`：真实单帧前置检查、资源清理及平台化错误提示。
- `tests/test_rtx_vsr.py`：探测尺寸、真实 `process()` 调用、质量透传、失败清理与平台提示契约。
- `docs/使用说明.md`：Windows/Linux RTX VSR 安装与前置检查说明。
- `docs/API说明.md`：API 服务器的 Linux/Windows 依赖与探测语义。

### Task 1: 用真实 360p 单帧替换 64×64 探测

**Files:**
- Modify: `tests/test_rtx_vsr.py:55-89`
- Modify: `nodes/rtx_vsr_stream.py:1-8,202-224`

- [ ] **Step 1: 写入真实单帧和失败清理测试**

在测试文件中增加模块导入：

```python
from nodes import rtx_vsr_stream
```

将现有两个 probe 测试替换为：

```python
@pytest.mark.parametrize("quality", ["HIGH", "ULTRA"])
def test_probe_vsr_capability_runs_real_360p_frame_and_cleans_up(monkeypatch, quality):
    calls = []
    probe_input = torch.zeros(3, 360, 640, dtype=torch.float32)
    probe_output = torch.zeros(720, 1280, 3, dtype=torch.float32)

    def fake_zeros(shape, *, device, dtype):
        calls.append(("zeros", tuple(shape), str(device), dtype))
        return probe_input

    class FakeProcessor:
        def __init__(self, api, selected_quality, device_id, width, height):
            calls.append(("init", api, selected_quality, device_id, width, height))

        def __enter__(self):
            calls.append("enter")
            return self

        def process(self, frame):
            calls.append(("process", frame))
            return probe_output

        def __exit__(self, exc_type, exc_value, traceback):
            calls.append("close")

    monkeypatch.setattr(rtx_vsr_stream, "load_vsr_api", lambda: "api")
    monkeypatch.setattr(rtx_vsr_stream, "VsrFrameProcessor", FakeProcessor)
    monkeypatch.setattr(rtx_vsr_stream.torch, "zeros", fake_zeros)
    monkeypatch.setattr(rtx_vsr_stream.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(rtx_vsr_stream.torch.cuda, "empty_cache", lambda: pytest.fail("probe must not call empty_cache"))

    assert probe_vsr_capability(quality, 0) is True
    assert ("zeros", (3, 360, 640), "cuda:0", torch.float32) in calls
    assert ("init", "api", quality, 0, 1280, 720) in calls
    assert ("process", probe_input) in calls
    assert calls[-1] == "close"


def test_probe_vsr_capability_reports_process_failure_and_still_cleans_up(monkeypatch):
    calls = []
    probe_input = torch.zeros(3, 360, 640, dtype=torch.float32)

    class FailingProcessor:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def process(self, frame):
            raise RuntimeError("NvVFX_Run failed")

        def __exit__(self, exc_type, exc_value, traceback):
            calls.append("close")

    monkeypatch.setattr(rtx_vsr_stream, "load_vsr_api", lambda: "api")
    monkeypatch.setattr(rtx_vsr_stream, "VsrFrameProcessor", FailingProcessor)
    monkeypatch.setattr(rtx_vsr_stream.torch, "zeros", lambda *args, **kwargs: probe_input)
    monkeypatch.setattr(rtx_vsr_stream.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(rtx_vsr_stream.torch.cuda, "empty_cache", lambda: pytest.fail("probe must not call empty_cache"))

    with pytest.raises(RuntimeError, match="NvVFX_Run failed"):
        probe_vsr_capability("ULTRA", 0)

    assert calls == ["close"]
```

- [ ] **Step 2: 运行测试并确认旧的 64×64 实现失败**

Run:

```powershell
D:\ComfyUI_windows_portable-G313\python_embeded\python.exe -m pytest tests/test_rtx_vsr.py -k "probe_vsr_capability" -v
```

Expected: FAIL；旧实现不会创建 `3×360×640` 输入，也不会调用 `process()`。

- [ ] **Step 3: 实现真实单帧探测与资源清理**

在模块顶部增加：

```python
import logging

LOGGER = logging.getLogger("MiniMaxH3.DirectorPlus")

PROBE_INPUT_WIDTH = 640
PROBE_INPUT_HEIGHT = 360
PROBE_OUTPUT_WIDTH = 1280
PROBE_OUTPUT_HEIGHT = 720
```

将 `probe_vsr_capability()` 的主体替换为：

```python
def probe_vsr_capability(quality: str = "HIGH", device_id: int = 0) -> bool:
    """Run one supported-size frame through the exact production VSR path."""
    probe_input = None
    probe_output = None
    cuda_device = torch.device("cuda", int(device_id))
    try:
        api = load_vsr_api()
        probe_input = torch.zeros(
            (3, PROBE_INPUT_HEIGHT, PROBE_INPUT_WIDTH),
            device=cuda_device,
            dtype=torch.float32,
        )
        with VsrFrameProcessor(
            api,
            quality,
            int(device_id),
            PROBE_OUTPUT_WIDTH,
            PROBE_OUTPUT_HEIGHT,
        ) as processor:
            probe_output = processor.process(probe_input)
        expected_shape = (PROBE_OUTPUT_HEIGHT, PROBE_OUTPUT_WIDTH, 3)
        if tuple(probe_output.shape) != expected_shape:
            raise RuntimeError(
                f"RTX VSR 探测输出尺寸异常：期望 {expected_shape}，实际 {tuple(probe_output.shape)}"
            )
        LOGGER.info(
            "[RTX VSR] 真实单帧前置检查通过 quality=%s gpu=%s input=%sx%s output=%sx%s",
            quality,
            int(device_id),
            PROBE_INPUT_WIDTH,
            PROBE_INPUT_HEIGHT,
            PROBE_OUTPUT_WIDTH,
            PROBE_OUTPUT_HEIGHT,
        )
        return True
    except Exception as exc:
        raise RuntimeError(
            "RTX VSR 前置检查失败，尚未开始 H3 视频生成。"
            f"\n质量：{quality}；GPU：{device_id}。"
            f"\n详细错误：{exc}"
            "\n请确认 NVIDIA 驱动与 nvidia-vfx 版本匹配。"
            "\n如果当前设备不支持该能力，请将‘最终输出’切换为‘原生尺寸直出’，不会影响 H3 生成。"
        ) from exc
    finally:
        if processor is not None:
            processor.close()
        probe_output = None
        probe_input = None
```

- [ ] **Step 4: 运行目标测试和 RTX VSR 全模块测试**

Run:

```powershell
D:\ComfyUI_windows_portable-G313\python_embeded\python.exe -m pytest tests/test_rtx_vsr.py -k "probe_vsr_capability" -v
D:\ComfyUI_windows_portable-G313\python_embeded\python.exe -m pytest tests/test_rtx_vsr.py -v
```

Expected: HIGH/ULTRA 真实单帧测试、失败清理测试和全部 RTX VSR 测试 PASS。

- [ ] **Step 5: 提交并推送真实单帧探测**

```powershell
git add nodes/rtx_vsr_stream.py tests/test_rtx_vsr.py
git commit -m "fix: validate RTX VSR with a real 360p frame"
git push origin main
```

### Task 2: 平台化依赖提示并完成回归

**Files:**
- Modify: `nodes/rtx_vsr_stream.py:15-31,216-224`
- Modify: `tests/test_rtx_vsr.py:22-45`
- Modify: `docs/使用说明.md:101-107`
- Modify: `docs/API说明.md:37-48`

- [ ] **Step 1: 写入 Linux/Windows 提示测试**

```python
def test_linux_dependency_error_prefers_driver_and_wheel_guidance(monkeypatch):
    monkeypatch.setattr(rtx_vsr_stream.sys, "platform", "linux")
    monkeypatch.setitem(sys.modules, "nvvfx", None)

    with pytest.raises(RuntimeError) as error:
        load_vsr_api()

    message = str(error.value)
    assert "570.190+" in message
    assert "580.82+" in message
    assert "590.44+" in message
    assert "nvidia-vfx" in message
    assert "Broadcast SDK" not in message


def test_windows_dependency_error_keeps_video_effects_runtime_guidance(monkeypatch):
    monkeypatch.setattr(rtx_vsr_stream.sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "nvvfx", None)

    with pytest.raises(RuntimeError) as error:
        load_vsr_api()

    assert "NVIDIA Broadcast SDK/Video Effects" in str(error.value)
```

- [ ] **Step 2: 运行测试并确认当前统一提示失败**

Run:

```powershell
D:\ComfyUI_windows_portable-G313\python_embeded\python.exe -m pytest tests/test_rtx_vsr.py -k "dependency_error" -v
```

Expected: FAIL；当前 Linux 提示仍要求 Broadcast SDK，且没有受支持驱动分支。

- [ ] **Step 3: 实现平台化运行库提示**

在 `_dependency_error()` 前增加：

```python
def _runtime_guidance() -> str:
    if sys.platform.startswith("linux"):
        return (
            "Linux 请使用 NVIDIA VSR 支持驱动分支：570.190+、580.82+ 或 590.44+，"
            "并在当前 ComfyUI Python 中安装官方 nvidia-vfx wheel。"
        )
    return (
        "Windows 请确认 NVIDIA 驱动、官方 nvidia-vfx wheel 与 "
        "NVIDIA Broadcast SDK/Video Effects 运行库兼容。"
    )
```

让 `_dependency_error()` 和 `probe_vsr_capability()` 的失败消息都拼接 `_runtime_guidance()`，删除无条件 Broadcast SDK 提示。

- [ ] **Step 4: 更新中文说明**

将两份文档的 RTX VSR 安装段落改为：

```markdown
RTX VSR 使用当前 ComfyUI Python 中的官方 `nvidia-vfx`。Linux 需要 NVIDIA VSR 支持驱动分支（570.190+、580.82+ 或 590.44+），不需要安装 Windows Broadcast SDK；Windows 需确认驱动、`nvidia-vfx` 与 Broadcast/Video Effects 运行库兼容。导演台会用 `640×360 → 1280×720` 的真实单帧执行 `load + run`，通过后才开始 H3 采样。
```

- [ ] **Step 5: 执行完整验证**

Run:

```powershell
D:\ComfyUI_windows_portable-G313\python_embeded\python.exe -m pytest -q
D:\ComfyUI_windows_portable-G313\python_embeded\python.exe -m compileall -q nodes tests
git diff --check
```

Expected: 全量 pytest PASS；编译与差异检查退出码为 0。

- [ ] **Step 6: 提交文档与平台提示并推送**

```powershell
git add nodes/rtx_vsr_stream.py tests/test_rtx_vsr.py docs/使用说明.md docs/API说明.md
git commit -m "fix: provide platform-specific RTX VSR guidance"
git push origin main
```

- [ ] **Step 7: 核对远程主分支**

```powershell
$local = git rev-parse HEAD
$remote = (git ls-remote origin refs/heads/main).Split("`t")[0]
Write-Output "LOCAL=$local"
Write-Output "REMOTE=$remote"
git status --short
```

Expected: `LOCAL` 与 `REMOTE` 一致，工作树无未提交内容。
