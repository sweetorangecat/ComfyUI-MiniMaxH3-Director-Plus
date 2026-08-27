# U19 工作台视觉升级 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将唯一 U11 Director Plus 工作流整理成 U19 风格的规整三列工作台，并在唯一 MP4 输出节点提供首帧/尾帧图片保存开关。

**Architecture:** 继续由 `tools/build_u11_workflow.py` 从 U10 派生工作流；只调整画布节点位置、分组、标题和输出节点的现有布尔输入，不改变 Director Plus 的自动模式路由、二采、性能和显存逻辑。输出节点仍是唯一的视频终点，首尾帧选项仅透传到 `MiniMaxH3StreamingVideoCombine`。

**Tech Stack:** Python 3.12、ComfyUI workflow JSON、pytest、Git。

---

### Task 1: 先为布局与首尾帧开关补回归测试

**Files:**
- Modify: `D:/ComfyUI_windows_portable-G313/ComfyUI/custom_nodes/ComfyUI-MiniMaxH3-Director-Plus/tests/test_workflow_tools.py`

- [x] **Step 1: 添加输出开关和布局契约测试**

在现有 `test_built_workflow_has_three_primary_sections` 后加入：

```python
def test_built_workflow_has_single_director_output_and_frame_export_toggles():
    source = {
        "last_node_id": 10, "last_link_id": 20,
        "nodes": [
            {"id": 1, "type": "Settings", "title": "Settings", "pos": [0, 0], "size": [300, 300], "mode": 0, "inputs": [], "outputs": []},
            {"id": 2, "type": "MiniMaxH3Director", "title": "", "pos": [400, 0], "size": [800, 500], "mode": 0, "inputs": [], "outputs": []},
            {"id": 3, "type": "DaSiWa_EnhancedVideoCombine", "title": "", "pos": [1250, 0], "size": [300, 300], "mode": 0, "inputs": [], "outputs": []},
        ], "links": [], "groups": [], "definitions": {"subgraphs": []},
    }
    built = build_workflow(source)
    assert sum(node["type"] == "MiniMaxH3DirectorPlus" for node in built["nodes"]) == 1
    outputs = [node for node in built["nodes"] if node["type"] == "MiniMaxH3StreamingVideoCombine"]
    assert len(outputs) == 1
    output = outputs[0]
    slots = {item["name"]: index for index, item in enumerate(output["inputs"])}
    values = output["widgets_values"]
    assert {"save_first_frame", "save_last_frame"} <= slots
    assert values[slots["save_first_frame"]] is False
    assert values[slots["save_last_frame"]] is False


def test_built_workflow_uses_u19_inspired_three_column_bounds_without_overlap():
    source = {
        "last_node_id": 10, "last_link_id": 20,
        "nodes": [
            {"id": 1, "type": "Settings", "title": "Settings", "pos": [0, 0], "size": [300, 300], "mode": 0, "inputs": [], "outputs": []},
            {"id": 2, "type": "MiniMaxH3Director", "title": "", "pos": [400, 0], "size": [800, 500], "mode": 0, "inputs": [], "outputs": []},
            {"id": 3, "type": "DaSiWa_EnhancedVideoCombine", "title": "", "pos": [1250, 0], "size": [300, 300], "mode": 0, "inputs": [], "outputs": []},
        ], "links": [], "groups": [], "definitions": {"subgraphs": []},
    }
    built = build_workflow(source)
    groups = {group["id"]: group for group in built["groups"]}
    assert {"u11-settings", "u11-director", "u11-output", "u11-assets"} <= groups.keys()
    assert groups["u11-settings"]["bounding"][0] < groups["u11-director"]["bounding"][0]
    assert groups["u11-director"]["bounding"][0] < groups["u11-output"]["bounding"][0]
    validate_workflow(built)
```

- [x] **Step 2: 运行新增测试确认当前布局契约尚未完全满足**

运行：`pytest tests/test_workflow_tools.py -q`

预期：新测试至少会因输出节点的 widgets/input 顺序或新布局断言失败；记录失败点后再实施。

### Task 2: 调整生成器的 U19 风格布局与输出默认值

**Files:**
- Modify: `D:/ComfyUI_windows_portable-G313/ComfyUI/custom_nodes/ComfyUI-MiniMaxH3-Director-Plus/tools/build_u11_workflow.py`

- [x] **Step 1: 在输出升级函数中按名称保留首尾帧开关**

检查 `_upgrade_stream_output_inputs()` 的输入重排逻辑，确保目标输入列表包含 `save_first_frame` 和 `save_last_frame`；对缺失旧输入补入 `False`，并按输入名称重建 `widgets_values`，不依赖旧槽位位置。

- [x] **Step 2: 在 `build_workflow()` 中设置单一输出的明确中文标题与编码默认值**

保留 `MiniMaxH3StreamingVideoCombine` 唯一节点，设置 `title = "最终视频输出（MP4 + 可选首尾帧）"`，并确保 `codec = "H.264"`、`container = "MP4"`、`save_first_frame = False`、`save_last_frame = False`。不创建第二个输出节点。

- [x] **Step 3: 重排主画布节点和分组边界**

在 `build_workflow()` 末尾统一设置：

```python
workflow["groups"] = [
    {"id": "u11-header", "title": "MiniMax H3 Director Plus", "bounding": [-430, 180, 2730, 150], "color": "#3f789e", "font_size": 26, "flags": {}},
    {"id": "u11-settings", "title": "能力与说明", "bounding": [-430, 400, 550, 1590], "color": "#3f789e", "font_size": 24, "flags": {}},
    {"id": "u11-director", "title": "导演控制台", "bounding": [120, 400, 1370, 1590], "color": "#4c806b", "font_size": 24, "flags": {}},
    {"id": "u11-output", "title": "预览与最终输出", "bounding": [1500, 400, 900, 1850], "color": "#5c7580", "font_size": 24, "flags": {}},
    {"id": "u11-assets", "title": "自动素材与兼容加速", "bounding": [-430, 2030, 2560, 620], "color": "#4c6f62", "font_size": 24, "flags": {}},
]
```

将顶部现有 Label 节点统一移到 `[-360, 220]`、`[-360, 280]`、`[-360, 330]`，宽度不超过 `2500`；将左列说明节点放在 `[-390, 470]` 起的垂直栅格，中列 Director Plus 固定在 `[140, 470]`，右列状态/输出固定在 `[1530, 470]` 和 `[1530, 720]`。所有节点边界必须通过 `validate_workflow()`。

- [x] **Step 4: 为右列增加中文输出说明节点**

新增一个 `MarkdownNote`，标题为 `输出说明`，放在右列输出节点下方，明确写出“最终保存为 H.264/MP4；首帧/尾帧图片开关只额外导出图片，不影响视频与音频”。位置与输出节点保持至少 30px 间距。

### Task 3: 重建唯一 U11 工作流并验证

**Files:**
- Modify: `D:/ComfyUI_windows_portable-G313/ComfyUI/user/default/workflows/minimaxH3/U11-MiniMaxH3-导演台Plus-中文增强版.json`

- [x] **Step 1: 运行生成器**

运行：

```powershell
python tools/build_u11_workflow.py --source "D:/ComfyUI_windows_portable-G313/ComfyUI/user/default/workflows/minimaxH3/U10-DaSiWa-MiniMaxH3-MythicAlchemy-v12导演台.json" --output "D:/ComfyUI_windows_portable-G313/ComfyUI/user/default/workflows/minimaxH3/U11-MiniMaxH3-导演台Plus-中文增强版.json"
```

- [x] **Step 2: 运行工作流结构校验**

运行：`python tools/validate_workflow.py "D:/ComfyUI_windows_portable-G313/ComfyUI/user/default/workflows/minimaxH3/U11-MiniMaxH3-导演台Plus-中文增强版.json"`

预期：报告唯一 Director Plus、唯一最终输出、无重复 ID、无断链、无可见节点重叠。

### Task 4: 全量验证、提交并推送

**Files:**
- Modify: 本任务中已列出的生成器、测试、说明文档和唯一 U11 JSON。

- [x] **Step 1: 运行全量测试与静态检查**

运行：`pytest -q; python -m compileall tools tests; git diff --check`

预期：全部测试通过，编译无错误，diff 无空白错误。

- [x] **Step 2: 检查最终契约**

确认 `git status --short` 只包含本次相关文件；解析 U11 JSON，确认 `MiniMaxH3DirectorPlus == 1`、`MiniMaxH3StreamingVideoCombine == 1`，输出节点的两个首尾帧布尔值均为 `false`。

- [x] **Step 3: 提交并推送远程主分支**

运行：

```powershell
git add tools/build_u11_workflow.py tests/test_workflow_tools.py docs/superpowers/specs/2026-08-27-u19-workbench-layout-design.md docs/superpowers/plans/2026-08-27-u19-workbench-layout.md
git commit -m "feat: refresh Director Plus workbench layout"
git push origin main
```

随后报告提交 SHA、远程分支状态和唯一 U11 JSON 路径。不要启动 ComfyUI。
