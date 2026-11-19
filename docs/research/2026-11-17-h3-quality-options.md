# H3 真实质感、长视频与速度：公开方案核查

后续实测、恢复端与色彩矩阵修正、各候选失败边界见[生成与交付实测](2026-11-17-h3-generation-trials.md)。OmniSR采用及45秒画质通过结论均已撤回；本文保留调研阶段的候选排序。

资料核查日期：2026-11-17。目标是保持原导演工作台与上传数量，改善皮肤、发丝、颜色和动作，同时覆盖原生克隆音色与不克隆两条路径。本次只研究与归档，没有改变运行默认值、下载大权重或新增 GPU 测试。

## 判断

有值得试的新路线，但没有证据支持“再加一个质感 LoRA，所有模式就同时变快、变真实”。优先级应是：

1. 复用完全相同的草稿，比较 **H3 Regenerate-2K-IC**、**FlashVSR+ Turbo**、现 VOSR2 与纯缩放。IC 属于 H3 条件重生成，FlashVSR+ 属于时序视频恢复；二者解决的问题和代价不同。
2. 如果进入恢复前的画面仍缺少质感，再比较完整、匹配的生成配方：官方 REF8 v1.1、SMD REF2VA 专训 8 步、Mystic V5 REF 作者单采。不要把这些全部叠入当前 SelfLift。
3. 长剧情继续使用原始参考固定、短镜头独立初始化、审核后拼接。单镜头延长单独处理。长视频架构和单镜头肤质是两项验收，前者通过不代表后者通过。

这是候选排序，不是本机质量或速度胜出的结论。

## 已有本地证据及边界

比较证据：`outputs/diagnostics/clarity_comparison_20261117/comparison-report.json`；原像素图：同目录 `old15_vs_long45_original_pixel_crops.png`、`long45_before_sr_after_sr_assembly.png`。

- 留存的独立 15 秒样片和新 45 秒长片都从 1344×768 生成，再重建到 1920×1098，约 1.43 倍；后者没有可确认的整体细节提升。旧片是同参考资产的留存诊断，不能冒充用户最满意的历史成片。
- 进入 VOSR2 前的无损帧已可见肤质不足；这些帧位于 ColorGuard 后。由此能排除“问题只在最终 MP4”，不能仅凭它锁定采样器、VAE 或颜色处理中的某一个环节。
- VOSR2 增加头发边缘、衣服纹理的清晰感，也使部分面部表面更平滑。三个增强片与拼接成片的对应帧平均绝对 RGB 差为 1.586、1.715、1.628 /255；现有证据不支持把拼接当作主要肤质损失来源。
- 新独立镜头样片未在已审采样密度下显示旧式逐段加暖、加深趋势，但镜头间表情、手、道具和肤色仍有差异；没有获得完整动作、唇形、音色和写实质感验收。
- 15 秒片段的 VOSR2 重建约 1023–1029 秒，即约 17 分钟；单段队列约 1731–1734 秒。计时来自已有执行记录，嵌套阶段不可相加，也不应外推为所有 5090 的速度。参见 `2026-11-17-long-video-independent-shots.md`。
- 旧片与新片的 seed、提示词、表演不同，不能据此量化“长视频路线导致退化”。后续必须使用同源草稿或严格控制生成输入。

负面对照必须保留：纯 REF 无加速 20 步、官方 REF8 **v1.0**、增大 SelfLift 首轮比例、去 Mystic、最后两步完整注意力、AddGuide 帧 0 都没有被验证能恢复中景/全身原参考肤质。新 2D lifter 减轻了斜纹，皮肤依然平滑。详见 `2026-11-14-h3-photorealism-controls.md` 和 `2026-11-16-native-texture-probes.md`。

## 恢复端：优先隔离，不再只看输出尺寸

| 路线 | 已核实的事实 | 本项目判断与风险 |
| --- | --- | --- |
| **Open-H3-Regenerate-2K-IC** | 2026-11-09 发布。IC 是 In-Context：草稿视频作为 REF2VA 的 `<Video 1>` 条件；作者意图保持脸部结构、布局和唇形。up3 是后段 3 次 transformer 推理，须配 HyperFlow、LMS Sharp、对应 sigmas/VAE。原草稿音轨直接保留。[1–4] | 最贴近 H3 的新候选。不是把原参考照片的毛孔复制回视频，也不是现 DMAD 上追加普通 LoRA。作者明确非 IC 版会画更多纹理但可能改变眼睛、皱纹、嘴形，IC 优先保留它们。没有我们 32GB 的完整 15 秒测速或质量验收。 |
| **FlashVSR+ Turbo / Base** | 2026-10 的新时序视频恢复模型，公开权重、消费卡支持、因果时空建模。[5–6] | 优先试 Turbo，使用同一草稿替换 VOSR2；原音轨保留，不改变生成端克隆方式。原始社区测试也有变暗、发丝闪烁、运动拖抹反馈，不能凭新模型名称切换默认。[7] |
| 现 VOSR2 | 官方是一阶段 **图像恢复** 模型；推理源码先 bicubic 到目标尺寸，再恢复；`upscale=1` 本来就是官方默认之一。[8–9] | 不是原生时序视频网络。实际保存的 Linux 兼容层按帧推理；DINO 特征缓存不等于跨帧纹理建模。`target_canvas_1x` 本身不是已证实错误，也不能承诺存在未发现的 fidelity/denoise 开关。 |
| SeedVR2 | 官方时序扩散恢复；官方卡主动提示轻退化 720p AIGC 可能被过度增强、过锐和补入不受欢迎细节。[10] | 可作第二轮对照，不能当成必然去油腻的替换。7B、长窗口与 CPU offload 的完整时间及内存要另测。 |
| SwinIR classical / BasicVSR++ / RealBasicVSR | 有公开重建或多帧恢复实现；SwinIR 区分 classical 与 GAN，RealBasicVSR 默认示例使用感知/GAN 损失。[11–12] | 可以做更克制的速度/保真对照，但也不会恢复原帧没有的人脸细节；不能因非扩散就称默认权重无生成倾向。 |
| 纯缩放 / VOSR2 残差混合 | 纯 Lanczos/bicubic 是必要无 AI 基线；固定混合原帧与生成结果是工程实验，不是官方 VOSR2 保真参数。 | 混合不会省掉已运行的 VOSR2 耗时；特征位置变化时还可能双边、虚影。只用离线诊断筛选，不直接作为修复默认。 |

### IC 已有代码的实际边界

截至核查，作者 HF revision `54a7104b`，Comfy 转换 revision `7d6e5e28`；社区节点 commit `a85bd966`。这是 Open-H3 作者经 NVLabs/Sana 公布的路线，不是 MiniMax 官方发布的恢复模型。作者原始配置与转换配方应分别锁定，不能根据 README 名称猜帧数。[1–4]

- README/节点标题还保留 124f，但当前 `nodes.py` 已允许 `frame_count=124..362`、步长 17，并验证 `5+17k`。至少要有 113 个真实输入帧。8 秒 192 帧可作首测；现有 107 帧小探针不适用。360 帧的 15 秒草稿可补到 362 的模型网格，交付时须精确恢复原帧数与时长。
- 输入预处理会 center-crop。接入我们的画幅时必须审查尺寸；如需对齐，应 pad 后精确去 pad，不能隐式裁掉画面。不能因为有 2K 权重就默认支持任意输出画布。
- 内部音频被静音 latent/mask 固定，最终复用草稿音轨。该接口可用于含克隆或普通原生音频的既有草稿；音轨保留不证明再生成的嘴形保持正确。
- 362 帧是当前社区接口边界。长片继续按已审短镜头处理，不能称一请求无限长。独立处理仍须检查切点、颜色和动作。
- 2K/3 步不代表低成本。高分辨率 transformer、额外参考 token、offload 都有代价；旧非 IC 版的 B200 数字不能转称 IC 在本机的速度。8GB 不应默认启用这条未经验证的路径。

### FlashVSR+ 的验收重点

作者模型和论文支持将它列为真正视频恢复候选，但论文 H200 或作者 4090 的数字不能用作我们的端到端承诺。需要检查 128 对齐、裁切/缩放、全部入 RAM、temporal compression、分块边界和音轨 mux；保持原画幅、fps、全部内容与首尾帧。[5–6]

社区原始反例是单人自测，不是受控横评；仍足以要求对发丝闪烁、脸部运动拖抹及整体变暗设明确失败门槛。[7]

## 生成端：按完整训练配方比较

| 路线 | 相比我们已测内容的新东西 | 应如何测试 |
| --- | --- | --- |
| **LightX2V REF8 v1.1 / 768p** | 2026-10-20 发布的实际不同权重；已测失败的是 v1.0。Comfy BF16 LoRA 的 v1.1 SHA256 为 `e56863a7524e0300729cbb7b64cc8b548d90698c1abbf7d83898b0122fbe1e81`。[13] | 锁定对应 REF 调度、时移、底模，不搬用 FL v1.3 的 shift6。先不叠 Mystic/People/MotionRepair/额外稀疏；克隆和无克隆分别测。作者讨论已有 REF8 语音及多适配器失败报告。[14] |
| **SMD REF2VA 专训 8 步** | 与 4 步是分别发布的学生；原生 Comfy 完整工作流、专用 ManualSigmas；不是把旧加速器多跑几步。[15] | 按作者 Euler、BasicGuider 无 CFG、strength1、12/3 shifts、final interval scale0.5 整套复现；不能再叠 Lightning/Turbo。公开 Compact 样例是 A100 限制预算、5 秒约24.3GiB，不能宣称物理24/32GB卡的15秒已经通过。克隆需独立验证。 |
| **Mystic V5 REF 作者单采** | 作者真实 JSON 使用纯 REF INT8_convrot、V5-ref1.0、旧 FL LightX4 v0.1 resized rank21 .75、res_multistep/simple/12 步（外层输入覆盖内部残留8），单采无 SelfLift；与我们 DMAD+V1+Euler/beta/SelfLift 不同。[16] | 完整复现作者路线再与基线比，不把不同模式 V5 混用。目前 `nodes/performance.py` 的 SelfLift 策略会把上游 V5 换回 U22 V1，直接追加 V5 不能证明已经测试了 V5。作者质量主张不等于我们动作/音色通过。 |
| fal People / Facial Realism CloseUp | 公开可下载的专用人物/近景适配器；People 最新可核实版本仍为8月，CloseUp 明确 experimental。[17–19] | People 在 pruned/INT8+Turbo 上存在作者承认的兼容问题及道具消失、色彩异常自报。CloseUp 不代表中景全身可用。作为替换式单 LoRA 对照，不能默认叠满。 |
| 官方 DMAD 专训8步 / Larry EMA600 | 有独立训练的工作点、专用调度和建议步数。官方 DMAD8 要 LCM/simple/CFG1，不能用当前混合 dareties 名称推定等价。[20–21] | 与当前适配器区分；先核实 REF、底模、原生参考音色覆盖，再做路线替换。更多步本身没有新的保真证据。 |
| HyperFlow H3 Flash | 已发布可运行引擎，专用分解层/双时间调度/Loader，存在完整4/6/8步方案。[22] | 官方 Comfy 当前 REF 接图像参考，视频/音频参考在 Python 路线；原生声音输出不等于现 Comfy 工作流已接通参考音色。15 秒要用含 INT8 row-chunk 溢出修复的 Loader；普通 LoRA 的剪枝索引不能盲套。 |

SMD 原生参考条件的接口、训练覆盖和实际克隆稳定性要分开记录；“同步音频”不是音色克隆保真证明。所有低步数方案都必须先检查无超分输出，不能用 SR 掩盖原生画面失败。

SMD 的4步 Comfy 仓库创建于2026-10-13、更新于10-29，不是本次调研当天的新发布。示例为两图、狗叫/笑声，未展示参考音色；其音频解码检查不等于试听或克隆验收。转换文件含208个LoRA目标，已读文件头的当前4步原始/Comfy检查点没有额外 replacement tensors；README通用警告不能被误读成它们必然漏载。转换含QKV与SwiGLU布局处理，不能拿通用PEFT重命名代替。详见分项研究中的实际文件头记录。

## 长视频：分镜剧情和持续长镜头分开判断

我们当前独立短镜头路线已经落实了社区最直接的止损方向：每镜回到原始参考，避免将上一镜的新颜色和纹理无限递归成下一镜条件。不能把这项已做内容包装成尚未实施的新修复。清晰度仍受每段原生质量与恢复端约束。

- tritant H3 Extender 的独立 Ref2VA 路线同样提供 Motion Context OFF、无前后生成帧条件和固定全局原始参考；其逐片颜色匹配在解码后进行，不改变 latent。新 latent refine 是另一次细化采样，不是已证实的去塑料捷径。[23]
- 原生 latent continuation、Motion Context/AV masks 适合需要承接的长镜头；留在 latent 域只能避免重复有损编码，不能保证模型自身不累计颜色、形体或身份误差。[24–25]
- Herrgott Infinite Continuation Suite 可周期性插入干净首尾帧并独立控制原生视频/音频mask，但其Qwen引用不是持续的原生Ref2VA条件；作者VALIDATION仍将真实模型下运动、颜色、音频检查列为待验收，不能当成已验证替代。[32–33]
- FramePack、RIFLEx、SkyReels/LongCat 等具体训练/架构不能当成给当前 H3 附加一个节点即可生效。若另建后端，要单独验证参考条件、音频、身份与原工作流约束；不能以它们的视频延长能力推定 H3 克隆能力。
- **LongCat-Video-Avatar 1.6** 是2026-11-15真正发布的新候选，作者称改善长时误差累积、嘴形和身体动作，有DMD/OPSD少步体系。它属于另一套生成后端，音频驱动既有讲话录音不等于H3用短音色参考生成新对白。若H3候选仍失败，可对长期对话类任务另行评估，不能默认代替复杂动作剧情。[34]

这项架构选择允许总剧情继续变长，不代表单个持续动作无限长且永不失真，也不能取消镜头切点的动作/道具审核。

## 哪些“新方案”暂时不能当现成修复

- **pi-H3**：作者公告为2026-11-10，计划11月16–22日发布代码/权重；核查时仍有 coming soon，DWD 自称实验。B200上5秒/3.8秒不是32GB卡承诺，也没有本项目REF克隆验收。[26]
- **Turbo-SLA**：当前官方是4步 FL2V/768p/85%稀疏，不能推广成所有REF+音色模式方案。[27]
- **VDN/FreeVideo**：FreeVideo v0.6.0确有11月16日更新，但当前源码仍声明 released VDN checkpoint 未经Ref2VA训练，Edge也不是新Ref训练；接口接受参考不等于新的保真能力。[28–29]
- **叠 LoRA**：People、Mystic、MotionRepair、蒸馏器、稀疏各自有训练假设。当前没有一手受控证据证明全部相加会去塑料并保持动作。MotionRepair作者也提示强权重可能增加CG/游戏感、影响音频。[30]
- **训练角色 LoRA**：社区有作者公开同种子对照及学习脸/音色的案例，但这是针对角色的数据/训练工程，不是免费通用修复；可作长期路线，不能拿单人演示代替我们中景动作验证。[31]
- **Realism Slider**在11月17日仅查到页面线索，对应API404。11月19日确认服务器已有2,700,080字节的文件，SHA256为`96a7bed24f7aa1351c6066666495ee64046faf130f53224f4e4a44b3b49b038d`，实际为50个MLP rank1模块。该文件与FastH3的层尺寸匹配，正反1.0均已完成同条件5秒测试，肤质目标仍未通过。原作者与推荐配方信息尚未核实，具体测试见生成实测报告。

## 最省算力的可判失败实验

1. **固定一段8秒、192帧草稿**，含说话、转头、手部和头发运动；保存采样后解码、ColorGuard后、恢复后的无损同帧。用现有草稿，不重跑生成来污染SR对照。该长度兼容IC最少113真实帧的要求。
2. 同源先比纯缩放、现VOSR2、IC up3、FlashVSR+ Turbo；统一画布、裁切、色彩空间、fps、音轨、编码和展示比例。不能拿不同尺寸缩略图或单张最好看帧评胜负。
3. 原像素检查皮肤和发丝；播放检查闪烁、虚影、纹路、手/道具、动作轨迹、嘴形与色彩随时间变化。IC或SR将脸改漂亮但失去原貌也判失败。
4. 记录模型加载/编译/预热、采样、解码、恢复、编码的分项与完整wall time，并记录峰值显存/RAM。既要比较热跑成本，也要交代冷启动。作者单模块倍数不能充当整片加速。
5. 恢复端没有合格者，再逐条比较完整生成候选；固定原参考、提示词、seed和时长，分别测克隆/无克隆。不同调度同seed仍不构成相同噪声轨迹，但应保留相同初始条件与多样本复验。
6. 8秒筛选通过才测试15秒；15秒通过才跑多镜头长片与接缝。用户UI不新增试验开关；只有质量/动作/音频/时间共同达标的路线，才进入原来的自动策略。

此阶段不承诺完全复制原照片中每一根发丝和毛孔。要达成的可检验目标是：比当前基线更自然的细节、更少美颜平滑/伪纹理，颜色不递增漂移，动作与音频不退化，完整运行时间可接受。

## 来源与可追溯记录

优先原作者代码、模型卡、实际工作流；社区评论仅表示原帖作者经验，不代表普遍基准。公开渠道包括GitHub、Hugging Face、Civitai及作者工作流和社区原帖；不声称覆盖所有私群或未公开方案。

1. [Open-H3-Regenerate-2K-IC模型卡](https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K-IC)
2. [NVLabs项目页，IC章节与非IC取舍](https://nvlabs.github.io/Sana/Open-H3-Regenerate-2K/#ic)
3. [Comfy转换模型与完整配方](https://huggingface.co/slmonker/Open-H3-Regenerate-2K-IC-ComfyUI)
4. [当前Comfy节点源码](https://github.com/slmonker/ComfyUI-OpenH3Regenerate2KIC/blob/a85bd966/nodes.py)
5. [FlashVSR+官方仓库](https://github.com/OpenImagingLab/FlashVSR_Plus)
6. [FlashVSR+论文](https://arxiv.org/abs/2610.17552)
7. [超分测试者原帖及后续反馈](https://www.reddit.com/r/StableDiffusion/comments/1xha6jy/ive_tested_11_video_upscaling_methods/)
8. [VOSR官方仓库](https://github.com/cswry/VOSR)
9. [VOSR一步推理源码](https://github.com/cswry/VOSR/blob/main/inference_vosr_onestep.py)
10. [SeedVR2官方模型卡，Limitations](https://huggingface.co/ByteDance-Seed/SeedVR2-7B)
11. [SwinIR官方仓库](https://github.com/JingyunLiang/SwinIR)
12. [RealBasicVSR官方仓库与训练配置](https://github.com/ckkelvinchan/RealBasicVSR)
13. [LightX2V实际权重文件树](https://huggingface.co/lightx2v/Minimax-h3-Turbo/tree/main)
14. [LightX2V作者发布与兼容讨论57](https://huggingface.co/lightx2v/Minimax-h3-Turbo/discussions/57)
15. [SMD REF2VA-8step原生Comfy工作流](https://huggingface.co/zjwfufu/SMD-MiniMax-H3-Ref2VA-8step-ComfyUI)
16. [Mystic V5 REF作者版本API](https://civitai.com/api/v1/model-versions/3416478)
17. [People原作者模型卡](https://huggingface.co/fal/MiniMax-H3-Realism-People-LoRA)
18. [People作者兼容讨论4](https://huggingface.co/fal/MiniMax-H3-Realism-People-LoRA/discussions/4)
19. [Facial Realism CloseUp原作者模型卡](https://huggingface.co/prithivMLmods/MiniMax-H3-Facial-Realism-CloseUp)
20. [官方DMAD模型卡](https://huggingface.co/ZhengmingYu/DMAD)
21. [Larry原作者模型卡](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora)
22. [HyperFlow官方集成文档](https://github.com/Video-Rebirth/hyperflow-h3-flash/blob/main/integrations/README.md)
23. [H3 Extender原作者代码](https://github.com/tritant/ComfyUI_MiniMax_H3_Extender)
24. [H3 Motion Context MultiRef](https://github.com/seitanism/ComfyUI-H3-Motion-Context-MultiRef)
25. [H3 Latent Continuation](https://github.com/SatoDive/Minimax-H3-Latent-Continuation)
26. [pi-H3作者公告](https://hanshengchen.com/blog/pi-h3/)
27. [Turbo-SLA官方模型卡](https://huggingface.co/lightx2v/Minimax-h3-Turbo-SLA)
28. [FreeVideo v0.6.0发布](https://github.com/FlashML-org/FreeVideo/releases/tag/v0.6.0)
29. [FreeVideo原始reference_sampler](https://github.com/FlashML-org/FreeVideo/blob/main/freevideo_engine/reference_sampler.py)
30. [Motion Continuity Repair作者说明](https://huggingface.co/JOKER141/MiniMax-H3-General-Motion-Continuity-Repair)
31. [角色LoRA作者实作与同种子对照](https://www.reddit.com/r/StableDiffusion/comments/1xbobrr/original_character_lora_for_minimax_h3_learnings/)
32. [Herrgott Suite作者README](https://github.com/HerrgottMargott/Herrgotts-H3-Infinite-Continuation-Suite)
33. [Herrgott作者VALIDATION](https://github.com/HerrgottMargott/Herrgotts-H3-Infinite-Continuation-Suite/blob/main/VALIDATION.md)
34. [LongCat-Video官方仓库与Avatar1.6发布说明](https://github.com/meituan-longcat/LongCat-Video)

分项研究与原始元数据：`outputs/diagnostics/quality_research_20261117/` 中三个 `*_sources.md`、模型API JSON、Mystic作者原始工作流JSON。作者宣称、社区自报、本机观察在本文中分别标记；没有把下载成功或流程跑通算作画质通过。


生成端配方复核补充：Mystic V5 作者图实际使用12步，外层连线覆盖内部残留8；详见 `2026-11-17-h3-generation-recipes.md`。LightX REF8 v1.1 的完整对应官方配置仍待补证，不以FL或旧REF4配置填空。

## 2026-11-19实际复验补充

FastH3 OmniRef原生1920×1088的15秒任务已完整执行两组，每组约30分41至43秒，另有条件准备和服务启动成本。发丝、衣料较清楚，面部仍平滑，两组声音辅助转写分别出现额外词句或重复台词，均验收失败。诊断保存器的旧色彩矩阵标记问题已核实并修正，原始PNG中的肤质问题继续存在。

进一步对照冻结的作者契约及本机源码，发现作者VSA采用128-token tile，本机Comfy采用64-token cube。新增隔离128行BF16适配经过CPU几何、GPU细算、原生QKV/RoPE及压缩分支数学差分检查，并实际生成同条件克隆音色5秒视频。完整任务约11分15秒，原像素对照仍未建立肤质提升证据。该样本维持拒绝采用状态，详细配方、边界、误差和媒体SHA见`2026-11-17-h3-generation-trials.md`。
