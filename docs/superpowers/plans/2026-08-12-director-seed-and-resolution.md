# U11 Director Seed And Resolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让导演台成为唯一规格入口，并把四种种子运行模式自动接入内部噪声节点。

**Architecture:** `MiniMaxH3DirectorPlus` 持有 seed 及 ComfyUI 原生生成后控制；工作流生成器把 seed 输出接到 Settings 子图。生成器同时移除 Settings 的三项旧规格输入并重排所有受影响链接。

**Tech Stack:** Python、ComfyUI 工作流 JSON、浏览器原生 JavaScript、pytest

---

### Task 1: 锁定节点与工作流契约

**Files:**
- Modify: `tests/test_director.py`
- Modify: `tests/test_workflow_tools.py`

- [ ] 添加失败测试，要求 seed 声明 `control_after_generate` 并从节点输出。
- [ ] 添加失败测试，要求旧规格输入从 Settings 与子图定义中移除，且 seed 自动连到 `noise_seed`。
- [ ] 单独运行这些测试，确认因功能尚未实现而失败。

### Task 2: 实现后端与工作流数据流

**Files:**
- Modify: `nodes/director.py`
- Modify: `tools/build_u11_workflow.py`

- [ ] 给导演台增加 seed 输入与输出，保持既有输出顺序不变。
- [ ] 在生成器中安全移除三个旧规格槽位并同步重排链接。
- [ ] 将 seed 输出自动连接到 Settings 的 `noise_seed`。
- [ ] 运行 Task 1 测试并确认通过。

### Task 3: 稳定界面并增加种子控件

**Files:**
- Rename: `js/minimax_h3_director_plus_v6.js` to `js/minimax_h3_director_plus_v7.js`
- Modify: `tests/test_frontend_source.py`

- [ ] 添加失败测试，要求中文种子数值和四种模式存在，且现有区块顺序保持不变。
- [ ] 在“生成规格”网格末尾追加两个控件，隐藏对应原生 widgets。
- [ ] 运行前端测试和 `node --check`。

### Task 4: 重建与验证单一工作流

**Files:**
- Modify: `templates/u11_api.json`
- Modify: `docs/使用说明.md`
- Modify: `docs/API说明.md`
- Modify: `D:/ComfyUI_windows_portable-G313/ComfyUI/user/default/workflows/minimaxH3/U11-MiniMaxH3-导演台Plus-中文增强版.json`

- [ ] 从原 U10 重建同一路径的 U11，并确认目录中只有一个 U11。
- [ ] 运行完整 pytest、工作流校验和前端语法检查。
- [ ] 检查节点无重叠、旧规格输入不再可见、seed 连线存在。
