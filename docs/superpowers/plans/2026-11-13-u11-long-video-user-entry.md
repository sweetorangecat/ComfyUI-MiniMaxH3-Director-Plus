# U11 长视频用户入口与实测方法接入

> **For agentic workers:** Use superpowers:subagent-driven-development to implement tasks sequentially, with independent specification and quality review. Existing user authorization covers this continuation; do not request another design approval or access the busy GPU server.

**Goal:** 在已有 U11 原生 AV 协调器上补齐实际用户入口、逐窗提示词与参考控制、可追溯双流返修和真实接缝审查，生成一个完整长视频文件。

**Architecture:** 沿用原 U11 图、有限时长规划、不可变 clean AV 检查点和标准 Comfy 队列。新的合同字段保持可选；按实际观测相位替换待演正文，参考筛选必须同步本地编号。常规返修使用秒数，专家条件只接受本项目已审来源并记录范围；不会自动批准内容或擅用未审生成余量。

**Tech Stack:** Python / Torch CPU tests / PyAV / ComfyUI graph and lazy nodes / JavaScript node tests。

## 已核对基线及范围

- 当前分支 `codex/u11-native-av-long-video`，起点 `1616e89`；源 checkout 和 GPU 实验目录不修改。
- 已有规划支持 30/60/90/97.125 秒及其他有限时长，没有 30/60/61 秒总时长上限；已有逐窗保存恢复、人工审查、原路径增强和完整装配，不重写这些能力。
- 90 秒已实测的是既有 loader/mask 隔离路径，不能据此前 1716 项离线测试宣称新生产入口已经 GPU 通过。
- 新发现的缺口：partial beat 复述完整旧动作；所有参考逐窗继承；quiet 窗仍绑音色；返修仅锁音频画面硬区间；UI 填 latent；缺少实际父末/子首预览；显式重生成种子不变；最终编码参数被硬编码覆盖。

## Task 1 — 合同、当前相位正文和参考绑定

Files: `nodes/long_video_plan.py`, `nodes/long_video_graph.py`, a small reference helper if needed; `tests/test_long_video_plan.py`, `tests/test_long_video_graph.py` and focused reference tests.

- [x] Add optional validated exact-window overrides keyed by `new_start` / `new_end`. Their `remaining_beats` replace matching clipped action text instead of appending it; optional six-section overrides provide current-scene prose. Original completed action and global summary remain audit metadata. Reject ambiguous/unknown overrides and references before queueing.
- [x] Add explicit timed-action/local-clock support. Convert embedded explicit English/Chinese event clocks, including sentence-final `65.400.` / `73.400.`, without rewriting dimensions; support structured frame-clock tokens for unambiguous contracts. Never rewrite spoken `<d>` content.
- [x] Add optional ordered `reference_selection` from the actual original flattened Picture/Audio slots. Compact selected links/file fields, remap directing markers, preserve Subject/Sx, record original bindings/maps/roles for Task2 to freeze in immutable requests. Reject dangling selections/markers and preprocessing-changing relocation. No policy keeps original reference selection.
- [x] Quiet owned-dialogue windows can omit local voice references while retaining inherited speaker identities. Remove stale file/link/name/transcript fields and voice-reference prose; never substitute an invented automatically generated speaker.
- [x] Return prompt policy/audit/selection/free-phase metadata from compilation for coordinator fingerprinting. Add optional `forward_frames` metadata (0 by default; multiples of17) without claiming the unused source tail is approved.
- [x] Red/green CPU tests exercise phase replacement, literal speech preservation, local event clocks and unchanged dimensions, noncontiguous media selection, quiet voice removal, and 30/60/90/custom duration behavior.

Task1 已通过独立规格与质量审查。最终独立质量回归 300 项通过；真实安装版技能 exporter 对 EN/ZH 各九窗的检查通过，保留 `not_ready_for_video: true` 且不写输出。生产入口使用 `long_video_graph.apply_window_reference_policy`；纯参考 helper 不导入 Director/Comfy/Torch，静态追踪不能冒充生产预处理校验。

## Task 2 — 双流返修和审过的前向条件

Files: `nodes/long_video_av.py`, `nodes/long_video_runtime.py`, a focused repair/conditioning helper if useful, store/service integration and matching tests.

已选择实现实际已审 parent-forward 协议，以兼容真实技能导出中的 17/34 帧条件；未来任意 AV 锚点协议仍须明确拒绝。待审尾部不会自动作为条件。正常秒数返修遇到已保护 forward 区域时明确拒绝；已接受的父窗可通过显式选定当前 take 的整窗重生成修正其未审尾部。

- [x] Canonical repair supports video-only, audio-only, joint AV and bounded fractional video weights. Validate finite [0,1] weights, actual float32 representation, integer ranges, protected history and at least one truly open region before invalidation. Unknown advanced fields fail explicitly. Existing locked-audio video requests remain compatible.
- [x] Optional parent-forward frames require actual accepted clean source, available surplus, matching codec phase, explicit source-tail review scope and observed phase. Derive stream ranges from native clocks; fingerprint all consumed video/audio values. Default0 does not reuse unreviewed surplus. Implemented `review_forward`; missing approval pauses before enhancement and continuation.
- [x] Arbitrary expert anchors, spatial masks, filesystem paths and future quiet-audio source protocols explicitly reject before invalidation. Their experimental evidence is not advertised as an implemented production capability.
- [x] `validate_result` checks every hard-zero element exactly, finite streams and actual changes in each requested positive stream; soft weights are not mistaken for hard-protected values. Preserve independent disk baselines and retry provenance.
- [x] Common user repair accepts film seconds, maps to native steps/ticks on the server, discloses actual snapped extent, and preserves history. Advanced API may expose native ranges; main UI does not.
- [x] Explicit rejected-content retry changes persisted sampling seed; automatic recovery retains original seed and bounded repair. Explicit accepted-parent replacement additionally binds the selected take. Preserve review/source invalidation and queue exclusivity.
- [x] Coordinator freezes all compiled policy/audit fields and the actual original/selected binding trace before `create_take`; graph execution validates this trace rather than mutating the immutable request. Fingerprints include model-affecting policies and the new reference helper's code identity.
- [x] CPU tests cover independent audio/video masks, soft weights, reviewed forward sources, unsupported future-anchor rejection, invalid request preserving accepted sources, failed repair recovery, stale parent history and retry seed distinctions.

Task2 已通过独立规格与质量审查。新增 kernel-lock CAS 将协调快照与模型指纹分离，保护配置、目标 take、重试次数、返修、父来源、停止和队列归属；真实 history-provider 的合法并发变更会重新读取，不把新控制状态覆盖成失败。解码后在队列对账前被拒绝的候选保留拒绝、文件和已生成进度，不自动重新采样。质量审查独立复核八条相关轨迹通过。最后整套回归留到 UI 集成冻结后执行，避免无必要的重复扩测。

## Task 3 — 用户入口、真实接缝和整片输出

Files: `nodes/long_video_entry.py`, `api/long_video.py`, `js/long_video.js`, `nodes/long_video_media.py`, workflow builder / existing U11 JSON / API template, appropriate tests.

用户主流程：开启 → 30/60/90 或自定义总秒数 → 导入 H3 技能的 U11 单语剧情 JSON → 检查剧情/素材接线 → 普通排队 → 逐窗必要内容与接缝审查 → 一个完整文件。导入不自动提交；当前总时长与合同不一致时清楚提示，不拉伸剧情或删台词。原 U11 默认 FL2VA/智能画质，长片可提供一个明确点击的“应用长片推荐设置”操作，设置既有 REF2VA/质量优先二采样/SelfLift/VOSR2 质量基线并关闭修脸/插帧，不默改音色、种子、画幅、素材或模型。关闭长视频仍走原短片入口。

- [x] Keep long video default off. Offer 30/60/90 convenience choices plus a finite seconds input. Provide contract import/example/preflight and readable errors; do not invent approved story/reference assets. Existing short output delegates unchanged.
- [x] Main panel shows target / contiguous approved / generated progress, compact window history, native review, resume/stop and one complete-film output. Preserve text drafts and playback across refresh. Accepted old windows remain selectable for correction.
- [x] Supply actual parent delivered-tail versus child first-new-frame evidence with timing and source identity. Generate a bounded CPU contact artifact from real previews; never infer visible continuity from equal latent prefixes.
- [x] 当项目要求借用父窗余量时，仅显示必要的“衔接素材审查”，用真实预览中的来源秒段填写实际自由起点状态并提交 `review_forward`。没有证明时保持 `awaiting_forward_review`；已经接受的父窗可明确整窗重生成以修正坏尾部。UI 不要求输入 forward 或 latent 数字。
- [x] UI repair uses seconds and simple labels; advanced technical fields remain outside the normal path. Approval remains explicit with scope and actual outgoing/remaining state.
- [x] Honor supported final encoder quality settings. Reject unsupported codec/container/bit-depth combinations during preflight rather than silently returning a different format. Do not alter existing short-film quality controls.
- [x] 完整成片支持 H.264/MP4/8-bit/AAC（Auto 显示实际解析结果），保留原二采 CRF 上限和音频码率；其他组合预检失败。RGB→YUV 使用实际 BT.709 full/limited 转换和元数据，真实 Y 平面、RGB 回传、CRF 质量/大小检查通过。
- [x] 增强/装配代码版本改变时要求重新核对导出指纹；PyAV 预检和可选依赖声明完成，不自动安装或部署。
- [x] Rebuild the original U11 JSON and API template idempotently; do not create another standalone production workflow. Add meaningful frontend runtime tests plus short-entry, workflow-structure, media and service regressions.
- [x] Publish the canonical machine-readable scene schema and actual EN/ZH H3 exports. Pending source facts stay unchanged; rich planning JSON is rejected with export guidance. Audio reference slots are statically 1..3, without capping Subject/Sx identities.

Task3 已通过独立规格与质量审查。普通秒数返修补齐浮点 ULP 边界兼容，真实重叠仍拒绝；API 首窗 `parent: null`、合法项目名与固定路径冲突、严格控制载荷及 malformed metadata/config 均有针对性回归。新前端直接运行真实扩展生命周期，包含同项目轮询乱序与跨项目回复隔离。

## Task 4 — 集成审查与交付

- [x] Final frozen implementation: 1957 Python tests and 97 JavaScript tests passed; original workflow/JSON/syntax checks passed; actual installed H3 exporter EN9/ZH9 passed with `not_ready_for_video: true` and no output writes. No GPU or deployment commands.
- [x] Independent specification review followed by quality review passed for all three tasks; substantive findings were fixed and targeted checks preceded the final whole-suite run.
- [x] Usage and capability table distinguish implemented normal/expert paths, explicitly unsupported conditions, CPU evidence, and pending new-entry browser/GPU validation.

交付约束：仅提交并推送本功能分支，保留工作树和未跟踪测试 sidecar。为 `E:/AIVideo/ComfyUI-MiniMaxH3-Director-Plus` 生成精确补丁与文件清单，仅在当前源文件匹配预期基线且补丁检查通过时应用；四处原有文件的 SHA256、所有研究、`examples/long_video_lab` 和 `outputs` 保持原样，不提交源 checkout。源项目更新、实际 ComfyUI 安装和服务器部署必须分开报告。最终 Git、补丁应用及保护校验结果由外部交付 manifest 记录；本阶段不重启、不动远程队列或部署服务。
