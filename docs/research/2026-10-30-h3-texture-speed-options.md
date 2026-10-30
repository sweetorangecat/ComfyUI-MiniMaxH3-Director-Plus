# H3 自然质感与速度方案核查

研究日期：2026-10-30。目标：RTX 5090 32GB 的 15 秒视频，以及低显存路线；同时考虑 REF2VA 克隆音色、不带音色和 FL2VA。以下只采用模型作者、项目维护者的代码、模型卡和版本 API。公开样片与作者测试不等于本项目的受控验证。

## 采用顺序

先修复现有路线中“选择动作增强也改变核心 LoRA 配方”的耦合，保持底模、加速 LoRA、核心质感 LoRA 和采样轨迹一致，再叠加可选动作 LoRA。Motion Repair 的作者明确说明高权重可能增加 CG/游戏感并影响声音；它不是通用去塑料模型，也没有证据证明它专为标准二采训练。当前样片的白色细纹仍需生成前后分阶段对照才能定位，不能从旧视频的报告推断本次原因。[^motion]

下一项最匹配自然质感目标的实验是 **Mystic V5 的 REF2VA 专版**。它已经发布，有明确作者与文件身份；作者描述的自然皮肤、灯光和音频训练比 Motion Repair V2 的翻滚/旋转目标更贴合当前需求。不过它与本项目 DMAD/SelfLift/音色保护组合的效果尚未验证，当前默认配方不应因“版本更新”就自动换成 V5。[^mystic]

结构性提速优先关注 FreeVideo / VDN 的独立对照，但不能直接替换当前 REF2VA 克隆路线：最新 FreeVideo 源码仍明确把 Ref2VA 标为实验，且发布的 VDN 权重没有做 Ref2VA 训练。[^free-code]

## 候选对比

| 方案 | 可解决的问题 | 音色和模式证据 | 速度与取舍 | 判断 |
| --- | --- | --- | --- | --- |
| 保留现有核心配方，动作 LoRA 独立叠加 | 避免动作选项隐式更换质感基线，便于定位回归 | 本轮修复作用于 SelfLift REF2VA，复用原有 H3 音频条件；仍需真实成片对照 | 不凭空增加采样步数；实际速度需运行报告 | 本轮先做的工程修复，不能据单测宣布雪花消失 |
| Mystic V5 REF2VA / FL2VA | 作者主张改善自然皮肤、光线、运动和音频；有按模式区分的权重 | REF 版仅训练 10–49 层 MLP，以减少参考/其他 LoRA 干扰；FL 版覆盖 MLP 和 attention，干扰范围更大 | 不是新的少步蒸馏器；作者没有本项目 5090 15 秒克隆测速 | 下一轮质感 A/B 首选；先替换同职责 LoRA，不叠加 V1+V5 [^mystic] |
| Motion Repair V2 | 空翻、旋转、倒置、恢复姿态和动作衔接 | 作者提到高权重会影响画风和音频；没有本项目 SelfLift 克隆保证 | 作者推荐单用约 0.9，搭配 Combat 首采 0.5–0.7，二采 0.2–0.3；不能直接套成 SelfLift 分辨率切换点的配方 | 动作专项；不作为去油腻或除雪花的默认药方 [^motion] |
| VDN-H3 ComfyUI 移植 | 8 步蒸馏，加局部精确注意力和长程线性分支 | 移植者测试 FL2VA/REF2VA，沿用原有音视频节点；但训练只覆盖 T2VA 类型布局，无克隆一致性的受控报告 | 5090 的公开例为 1280×736、145 帧、8 步，约 2:15 **采样**，含音频；不是 15 秒或 2K。作者明确提醒不是消费卡最省最快路线 | 可单独试验；不得与原加速 LoRA 或 Scheduled Sol Attention 直接叠加 [^vdn-port] |
| FreeVideo v0.4.25 / VDN Edge | 面向消费卡的流式权重、分块、内存调度和量化，含 ComfyUI 接口 | 实际代码接受图片/视频/音频参考，音频进入 Ref2VA 条件；`Reference` 与原样使用音轨是两种行为。Ref2VA 仍是未专门训练的实验兼容 | 作者收集到 5090、1344×768、10.8 秒、Light 档 249 秒端到端记录；没有相同种子/克隆/2K质量基准。表中 `After` 大多为估算，不能当该机器复测 | 新执行引擎中值得优先测的一项，但需要独立接入和音色验收 [^free] [^free-code] [^free-results] |
| LightX2V Turbo-SLA | 专门训练的 4 步、85% 稀疏注意力 FL2V 加速 | 已有 ComfyUI BF16 LoRA；模型卡只承诺 768p FL2V，不证明 REF2VA 原生克隆适用 | 作者在自有 LightX2V 5090 设置中报告约 2.5 倍；依赖训练匹配的 SLA、video/audio shift 和步进方式 | 可给 FL 路线单独研究；不能用同一个适配器覆盖全部模式 [^sla] |

**VDN 宣传的约 9 秒：** 官方对应 8×B200、768p、14.4 秒视频、预热后的 SGLang 路线。其参考表还把模型加载、预热、VAE 和编码排除在采样统计之外；不能转述成单张 5090 的 15 秒完整任务时间。[^vdn-official]

## SelfLift、动作重生成和超分的边界

- selflift-Avatar 当前开发分支支持双模型、全局上下文省显存，以及长片 learned upscaler 首帧边界修复。维护者仍标为实验：CPU/前端替身测试不能证明真实模型的质感、唇形或速度。`spatial` 空间切片可能引起背景鼓胀、漂移和接缝；`global_stream` 保留上下文但有运行环境要求，`global_ram` 更慢。音频 mask 传递修复也不等于克隆或唇形准确性保证。[^selflift]
- MAINodes Motion Lab 用时间拉伸、视频重新生成和恢复原时间轴处理快动作拖抹；公开例是 192 帧、约 1MP、两轮 12+6 步约 12 分钟，音频初始化仍被维护者标为窄范围验证。它适合特定快动作的专项修复，不是日常人物去塑料的廉价后处理。[^mainodes]
- SeedVR2 官方明确承认：对轻退化输入，例如 720p AIGC 视频，可能过度补细节、过锐和生成不讨喜细节。应先比较超分前后，再决定小倍率收尾；不能把生成阶段的结构错误交给超分修复。本次未核实 VOSR2 上游身份，不把 SeedVR2 的结论移植成 VOSR2 的确定缺陷。[^seedvr]

## 最小验证矩阵

作者 V5 REF 附带工作流（file `3311449`）里，`LoraLoaderModelOnly` 使用 V5 REF 强度 1、旧版 FL2V 4-step resized adapter 强度 0.75，采样器选择 `res_multistep`。这与本项目的 DMAD/v4 + Euler/beta SelfLift 配方不同；不能只把作者的强度 1 搬进当前链，就宣称复现了作者效果。该 JSON 的文件身份可由 Mystic 版本 API 核实。[^mystic]

1. 固定实际底模/量化版本、提示词、参考图、音频片段、FPS、sigma、分辨率与种子。当前问题样片种子可用 `9685887441`；缩成 4 秒只用于快速定位，不能代替 15 秒验收。
2. 先对照“核心配方固定、动作关闭”与“同核心配方、动作连续性开启”；分别测克隆音色与无音色。确认动作开关没有顺带替换底模、核心 LoRA、加速适配器或采样轨迹。
3. 再单独用 Mystic V5 REF 专版替换旧 Mystic，保留其余设置。FL2VA 使用 FL 专版单独测；不把两版混用，也不把作者推荐强度当成本项目已验证最优值。
4. 保存低分辨率采样末端、learned lift 后、最终采样后、超分前、超分后同时间帧。检查桌面/墙面的亮纹、皮肤和织物微纹理、头发边缘、身份/服装、运动连贯性、唇形与声音。用相同时间戳和像素区域比对。
5. 独立记录文本编码、首采、lift/精修、VAE、超分和编码的墙钟时间与峰值显存/内存。新引擎同时保留冷启动和预热后时间，报告实际生成尺寸与输出尺寸。
6. 通过短片后才做相同参考、音频的 15 秒 768p→1080p/2K；新增方案只有在真实画质、音色和总耗时同时满足目标后才进入默认路线。此次研究未执行这些 GPU 对照。

## 本轮已执行验证

修复前，实际加速路由测试中 SelfLift 的 `continuity` / `combat` 配置会缺失所选配方的 Mystic。修复后，48 组用例覆盖 DMAD/v4、无音色/H3 原生参考、四种动作选项与三种旧路由开关，验证核心配方、动作追加顺序、实际返回模型和已加载列表一致；两阶段仍共享同一模型。采样适配、模型路由、音频交接、二采、资产和文档相关检查共 442 项通过，另通过 Python 编译与 diff 空白检查。

这些是代码与调用契约验证。没有这条样片对应的完整执行报告，也未完成 RTX 5090 的同种子 GPU 对照，不能据此宣称雪花已消失、自然质感已提升或总耗时已下降。Mystic V5、Motion Repair V2、VDN 与 FreeVideo 均未在本轮自动安装或替换默认配方。

## 可复核版本与一手来源

所有时间为 UTC；版本创建时间不等于每个附加文件各自的上传时间。

| 项目 | 本次核实身份 |
| --- | --- |
| Mystic V5 | Civitai 作者 `alcaitiff`，model `2856467`，version `3416478`；版本发布 `2026-10-19T20:05:02Z`，更新 `2026-10-20T13:15:23Z` |
| Mystic V5 REF | `MysticXXX_MMH3-V5-ref2va.safetensors`，file `3311427`，SHA256 `F57C76E761F7A5113E541AD58E84C648A5FE4D3CFBC277D0C44AE9763DA29AFE` |
| Mystic V5 FL | `MysticXXX_MMH3-V5.safetensors`，file `3309543`，SHA256 `4168917DAEB6E570A004BC6994F73D091B9F3E887F7E946BD00C0F2F7573E5F3` |
| Motion Repair | HF revision `1676b352bb9f62407f76aee2ad55db2f36d04471`，模型卡更新 2026-10-25；V2 权重提交 `7ed4f7ad3d5a4b7cc27d3ff0a6dd7c219064bbfb`，2026-09-28 |
| VDN 官方 | GitHub `e262cb5770b3dc992d4f44f251cd17bbfe0c4d8b`，2026-10-05；HF `2ce18deb1d1c6b115d6bb3732f3750fdbac95173` |
| VDN ComfyUI 移植 | `Saganaki22/ComfyUI-VDN-H3` v1.5.2，发布 2026-09-12；SHA `3eb63496c24ca70faaf8a14b6c75fcb480e34bf1` |
| FreeVideo | v0.4.25，发布 `2026-10-29T17:43:06Z`；SHA `8d5e0272b7c0fe7672739803cd3179ceb3b9a316` |
| VDN Edge | HF revision `94f5ccc7bb78917e1eed47de4fc04a61465b8673`，2026-10-13；是 FreeVideo 引擎包，不是额外训练的画质模型 |
| Turbo-SLA | HF revision `b4cd41131be06909110464ec1be3b30aa6068e8e`，更新 2026-10-16 |
| selflift-Avatar | SHA `8e01c093f027c0281025741b025f117475ea044b`，2026-10-25；发布包 v0.1.4 不包含所有 main 分支修复 |
| MAINodes | 实际维护仓库 `matlowai/ComfyUI-MAINodes`，SHA `f4868b4a08e8a504ce86db54a17961d399ffa2bc`，2026-08-29 |

[^mystic]: 作者模型 API：[model 2856467](https://civitai.com/api/v1/models/2856467)；[version 3416478](https://civitai.com/api/v1/model-versions/3416478)。作者的自然质感、测试胜率和音频描述是自报结果，未在本项目复现。
[^motion]: [作者固定版本模型卡](https://huggingface.co/JOKER141/MiniMax-H3-General-Motion-Continuity-Repair/blob/1676b352bb9f62407f76aee2ad55db2f36d04471/README.md)；[提交历史 API](https://huggingface.co/api/models/JOKER141/MiniMax-H3-General-Motion-Continuity-Repair/commits/main)。
[^vdn-port]: [固定版本 README](https://github.com/Saganaki22/ComfyUI-VDN-H3/blob/3eb63496c24ca70faaf8a14b6c75fcb480e34bf1/README.md)；[v1.5.2 发布](https://github.com/Saganaki22/ComfyUI-VDN-H3/releases/tag/v1.5.2)。注意 `merge` 与 `bypass`、`fast_kernels` 的蒸馏模型漂移说明，以及与 Scheduled Sol Attention 的互斥。
[^vdn-official]: [官方固定版本 README](https://github.com/OpenVDN/vdn-minimax-h3/blob/e262cb5770b3dc992d4f44f251cd17bbfe0c4d8b/README.md)；[官方权重固定版本](https://huggingface.co/OpenVDN/vdn-minimax-h3/blob/2ce18deb1d1c6b115d6bb3732f3750fdbac95173/README.md)。
[^free]: [FreeVideo 固定版本 README](https://github.com/FlashML-org/FreeVideo/blob/8d5e0272b7c0fe7672739803cd3179ceb3b9a316/README.md)；[v0.4.25 发布](https://github.com/FlashML-org/FreeVideo/releases/tag/v0.4.25)；[VDN Edge 固定版本](https://huggingface.co/OpenVDN/vdn-minimax-h3-edge/blob/94f5ccc7bb78917e1eed47de4fc04a61465b8673/README.md)。
[^free-code]: [Comfy 节点模式声明与参考音频接口](https://github.com/FlashML-org/FreeVideo/blob/8d5e0272b7c0fe7672739803cd3179ceb3b9a316/freevideo_engine/comfy_nodes.py)；[Ref2VA 采样器及未做参考训练的声明](https://github.com/FlashML-org/FreeVideo/blob/8d5e0272b7c0fe7672739803cd3179ceb3b9a316/freevideo_engine/reference_sampler.py)；[图像/音频参考布局与条件校验](https://github.com/FlashML-org/FreeVideo/blob/8d5e0272b7c0fe7672739803cd3179ceb3b9a316/freevideo_engine/media_conditioning.py)。
[^free-results]: [维护者公布的社区执行报告汇总及估算口径](https://github.com/FlashML-org/FreeVideo/blob/8d5e0272b7c0fe7672739803cd3179ceb3b9a316/docs/community-results.md)。不是克隆音色或去塑料受控质量基准。
[^sla]: [LightX2V 作者固定版本模型卡与完整运行配置链接](https://huggingface.co/lightx2v/Minimax-h3-Turbo-SLA/blob/b4cd41131be06909110464ec1be3b30aa6068e8e/README.md)。
[^selflift]: [维护者固定版本 README](https://github.com/slmonker/selflift-Avatar/blob/8e01c093f027c0281025741b025f117475ea044b/README.md)。
[^mainodes]: [维护者固定版本 README](https://github.com/matlowai/ComfyUI-MAINodes/blob/f4868b4a08e8a504ce86db54a17961d399ffa2bc/README.md)。
[^seedvr]: [SeedVR2 官方模型卡的 Notice / Limitations](https://huggingface.co/ByteDance-Seed/SeedVR2-7B)。
