# 文武双全作者配方 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** 在原 U11 导演台的动作增强选项中提供用户指定金鱼工作流的明确、可核验配方，不另造工作流、不静默回退成部分组合。

**Architecture:** 复用 `community_lora_profile=combat`，显示“文武双全（作者配方）”。单独的无 Torch 配方模块集中定义作者归档图参数和严格资源解析，原导演台、路由、调度、SelfLift 和长视频资源指纹共同消费它；未选该配置时保留现有行为。

**Tech Stack:** Python / pytest，ComfyUI 原生模型路由，SelfLift-Avatar，JavaScript / Node test runner。

## 已确定的设计和边界

用户在已解释当前仅部分接入后明确选择“作者的那种”；沿用此前原导演台、简化界面、自动长片、保留输入容量和声音映射的授权。无需再让用户选参数或批准同一方向。

证据是 `docs/research/2026-11-13-wenwu-workflow-comparison.md` 引用的作者公开页面和用户先前提供的归档图。归档图速度节点 343 的 `strength_model` 连到 366，366 保存值确认为整数 3。动作四节点在归档图中均 bypass；本次只在用户主动选“文武双全”时开启，不声称作者默认全开。

确定参数：simple / Euler / 12 间隔，transition=6，CFG=1，video/audio shift=12/3，lowres_scale=.6；DMAD FRO .8 → Mystic V1 .35 → Combat V2 .65 → Motion Repair V2 .35 → Fight Flow V1 .35 → canonical speed slider1.1 3.0；learned upscaler LMS v0.1。归档图中文速度文件与当前作者公开清单 canonical1.1 未证实字节一致，记录此差异，不假装等价。仅接受确定的文件名，不把旧 Repair/V5/Turbo/其他 upscaler 当成等价。

保留现代 Avatar global_stream、原生 Sol-Attn、双流 hard mask 和显存门槛；作者配方要求可用的全局路径，避免空间切块造成运动/白丝退化。该执行适配与旧图 full-frame 不同，必须在元数据和文档说明。8GB 不支持此 SelfLift 配方，原低显存路线不变。

适用 REF2VA + 质量优先二采 + 无音色/H3 原生参考/完整复用音频，选作者配置即自动选择 SelfLift；Fish、其他模式、低显存、非24FPS/RIFE 明确阻止，不静默换配置。原长片额外限制保持。附加 LoRA 中重复的作者资源和 Mystic V5 明确提示清空冲突项，不能悄悄叠加或静默撤掉用户配置。

所有七个必需资源缺失在导演台预检给出完整列表；模型路由的实际 LoRA 应用失败必须报错；用户显式选择的配方不受旧“全部关闭”或可选链环境变量悄悄删减。记录实际执行参数及已加载文件/强度，长片冻结相同资产。

## Task 1：后端配方与严格资源、采样集成

**Files:**
- Create `nodes/wenwu_recipe.py`, `tests/test_wenwu_recipe.py`
- Modify `nodes/sampling_experiments.py`, `nodes/two_stage_assets.py`, `nodes/performance.py`, `nodes/director.py`, `nodes/long_video_assets.py`, `nodes/long_video_entry.py`, `nodes/long_video_graph.py`, `nodes/two_stage.py`, `nodes/execution_report.py` and their relevant tests.

- [x] 先增加失败用例并实际运行：兼容请求 `validate_sampling_options({...,'sampling_method':'standard','community_lora_profile':'combat'})` 返回 `sampling_method='selflift'`；非兼容请求清晰报错。
- [x] 资源用 injected resolver 验证严格 FRO+V1+四动作+LMS；仅旧 Repair/V5/Turbo 均不能满足；收集全部缺失资源。
- [x] 以已有路由/调度测试边界验证 simple12、split6、Euler、12/3 shift、CFG1、.6 低网格及 `run_selflift` 接收13个 sigma；off 仍 beta10/6+4。
- [x] 验证真实加载调用顺序及强度；缺文件、失败应用、额外重复/V5 冲突均报错；无关附加适配器不被静默修改。
- [x] 原 SelfLift AV nested latent、全局执行、双方 masks 和保护历史检查完整保留；无全局能力时作者配置明确拒绝。
- [x] 作者配方、实际调度/步数/shift/asset chain 纳入 guide/report，长片冻结所有七个权重及真实强度；安装/替换资源改变指纹。
- [x] 运行相关 pytest 并自审；仅编辑所列生产模块/相关 Python 测试，不碰 JS/docs、不提交 Git、不运行 GPU、不部署服务。

Example contracts to encode in tests:

```python
normalized = validate_sampling_options(dict(mode='REF2VA', voice_mode='none',
    performance_preset='quality_two_stage', sampling_method='standard',
    community_lora_profile='combat', fps=24, motion_smoothing='off'))
assert normalized['sampling_method'] == 'selflift'
assert scheduler_plan(dict(normalized, performance_preset='quality_two_stage',
    resolved_backend='ref2va_model', mode='REF2VA', voice_mode='none')) == {
        'scheduler': 'simple', 'steps': 12, 'split_step': 6,
        'refine_reference_tail': False}
```

Run with portable Python: `python -m pytest tests/test_wenwu_recipe.py tests/test_sampling_experiments.py tests/test_sampling_integration.py tests/test_two_stage_assets.py tests/test_performance.py tests/test_director.py tests/test_long_video_assets.py -q`.

## Task 2：原导演台操作及说明

**Files:** Modify `js/minimax_h3_director_plus_v11.js`, `tests/js/director_restore.test.mjs`, `docs/使用说明.md`, `docs/SelfLift短视频实验.md`, existing research note; create a focused user-facing `docs/文武双全作者配方.md`.

- [x] JS 先失败测试：旧 `combat` 值仍能恢复、显示新名称；兼容选项自动 SelfLift；切换到不兼容模式不隐瞒错误且不改媒体/seed/时长。
- [x] 仅更名现有动作选项和简短说明，不增加参数控件或工作流；作者配置下方法控件不可悄悄降为 standard，仍可退出到 off。
- [x] 文档写清适用模式和显存、主动选择、动作链非原默认、12 vs10 计算差异、未 GPU 验收、现代执行适配、速度文件身份差异、下载来源未解决项；不得承诺去油腻/提速结果。
- [x] 调整研究 note：归档连线确认 slider3；对照表注明基线版本，不倒改研究事实。
- [x] Run `node --test tests/js/director_restore.test.mjs tests/js/long_video.test.mjs tests/js/media_slot_state.test.mjs` and `node --check js/minimax_h3_director_plus_v11.js`.

## Task 3：资源准备、验证和交付

**Files:** Researcher owns `docs/research/2026-11-13-wenwu-author-assets.md`; any install receipt stays outside Git.

- [x] 核对模型作者/授权发布者固定 URL、大小与 SHA256。查找已存在模型，优先复用，禁止替换现有文件。
- [x] 仅从已验证公开源下载缺少的资产至实际 ComfyUI 模型目录；临时文件完成后校验 SHA256 再原子落盘。Mystic V1 已经原作者平台元数据与同 SHA 镜像核验，未用 V5 替代。
- [x] 独立 spec 审查，再独立 quality 审查；修复所有实际 P1/P2 并重测。
- [x] 全量 pytest 前备份原 untracked `pyav-resource-test.mp4.execution.json`，结束后恢复；运行 Python 全套及 JS 全套，`git diff --check`。
- [x] 仅暂存该任务明确文件。用 `git -c maintenance.auto=false -c gc.auto=0 commit`，推送现有 feature branch 并 readback hash；保留源 main 及无关编辑。
- [x] 检查实际本地插件无活跃作业后做备份和范围更新，保留既有 upscale 修改和删除的示例；不主动运行 GPU。
- [x] 远程 90秒验收未完成时不重启、不提交 GPU、不部署，明确交付状态；绝不把源码通过当成服务器/GPU画质通过。

## 验证记录

实现、源码审查、代码推送与本地范围同步已完成；运行中远程服务保留原样。

- 最终全量 Python：**2122 passed**，179.78 秒；一个既有 pynvml 弃用警告。
- JavaScript：**110 passed**；入口语法和 `git diff --check` 通过。
- 后端相关集：694 passed；独立后端 spec 和最终 quality 各自复验103 passed。UI spec 独立复验110 passed。
- 独立 spec 找到并修复了错误 lifter 和相同 MODEL 连线跳过校验两处边界；新测试先重现10项失败，再绿。最终无未解决 P1/P2。
- 七个实际权重均重新计算 SHA-256 且 SafeTensors header/payload 校验通过，严格文件解析无缺失。未运行 GPU，未操作实际浏览器界面；内核/插件执行、速度、长片音画质仍需实测。
- SSH 只读队列确认一条运行、一条待跑；没有取消、重排、更新或重启远程服务。
- 本地及源目录25文件同步预检无冲突；保留原低显存路线、附加 LoRA 及工作流删除状态。模型和同步收据保存在仓库外。

代码发布提交 `8c23fa0b5dc1fda787776ca2becd5af4c40ea2ae` 已推送到 `origin/codex/u11-native-av-long-video` 并 readback 同哈希。实际本地插件已快进到该提交，源目录同步25个文件并逐个读回；源 main 的 HEAD 保持不动。原 upscale 修改、六个已删除示例和 untracked sidecar 保留；本地未启动/重启服务，远程仅只读查队列。本记录后续文档提交不改生产代码。
