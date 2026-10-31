# 八个 H3 社区工作流的借鉴评估

研究日期：2026-10-14。对照代码：`6119d50`。用户已授权采用方案 B；本报告保留研究阶段的证据与取舍，实际接入范围和试用方式见 [SelfLift 短视频实验](../SelfLift短视频实验.md)。

结论：优先借鉴 U32 的渐进分辨率采样思路，U34/U36 的两阶段模型与资源交接，以及 U40 的音视频 latent 接续。不要把八套工作流的 LoRA、注意力补丁、二采和超分全部叠加。运动清晰度要在 H3 生成阶段验证，不能靠输出 FPS 或超分名称判断。

## 证据范围

- 读取八份用户提供的原始 JSON，合计 607 个根图节点，并检查 U36、U40 的子图、节点 mode、Set/Get 虚拟连线、链接输入和控件值。
- 八份根图均未发现指向不存在节点的 link，或输入槽与 link 目标不一致的情况。这是静态结构检查，不是 ComfyUI 运行验证。
- 按 JSON 的版本字段查阅 SelfLift、YCNodes、T8、Audio Guide Feather、ClipStream、FeiHou 和 ComfyUI 的公开源码。笔记节点和文件名仅作为资料，不作为有效运行行为的依据。
- 没有在 AutoDL 上运行这八套图；没有同模型、同提示词、同种子的完整视频对照。因此不承诺运动不失真、提速比例或 8GB 不爆显存。
- 原始文件指纹、节点参数和版本记录见同目录 `2026-10-14-community-workflows-evidence.json`。

## 逐套结果

| 工作流 | 实际路线与关键证据 | 借鉴价值 | 采用判断 |
| --- | --- | --- | --- |
| U30 Singularity | 社区 Singularity REF 底模；Euler；simple 6 经全区间 sigma 插值；2 步切分；训练型 latent 1.5x；末段加载 LMS 0.3；独立处理视频与音频 | sigma 切分、视频放大而保留音频、精修 LoRA 只在后段应用 | 后两项有价值；当前项目已有 AV 分离和二采。整套不是轻量提速配方 |
| U31 动作替换迁移 | Viggle 专用底模、DMD LoRA、固定文本 embedding、输入动作视频、窗口采样 | 有明确动作视频时提供更强动作约束 | 独立动作迁移功能，不是普通 T2VA/REF2VA 的参数优化 |
| U32 BUNNY SelfLift | hybrid b25-49 INT8；turbo v4 0.7；simple 8 + refiner；SelfLift 低分辨率前段 6 步、空间倍率 0.5；TST 0.2；SolAttn；动作 LoRA | 用渐进分辨率减少昂贵的高分辨率采样；可单独验证 TST 的时序稳定性 | 最值得建立实验对照，但 H3 适配仍是实验性。不能照搬所有武戏 LoRA |
| U34 WORK-FISHER | LOW/HIGH 两套 conditioning 与 LoRA；双时钟 Euler；实际计划 base=9、coarse=4、refine=5；latent 放大 1.5x；HIGH 尺寸来自放大器；显式卸载 | 两阶段网格契约、按阶段配置动作 LoRA、显式音视频时间参数和资源释放 | 工程方法值得参考；不是仅复制一个 scheduler 就能获得相同效果 |
| U35 “官流” | 社区 Cobijada hybrid 底模；er_sde；手写 11 个 sigma 值；ref2v 8-step LoRA 0.75；fafafast 6、Motion Repair 0.3、Combat 0.65 | 手写时间表及运动节奏 LoRA，可作小范围对照 | 文件名不能证明官方最优；不作为默认通用配方 |
| U36 FeiHou v6 | 实际链接控制总步数 9、一采 4、二采 5；1.5x latent；第一阶段产物先预览；第二阶段等待预览输出；Semantic Bridge 0.1；面部精修旁路 | 首采预览、阶段模型交接、按音频真实时长裁切、将后处理拆开验证 | UI/交接可借鉴；所标公开版本缺少关键节点注册，暂不能按完整可复现方案直接移植 |
| U37 SelfLift 接续 | 默认启用上下文；前段尾部 22 帧经 AddGuide 作为新段开头；音频羽化 12 ticks；默认上下文分支 SelfLift 4 步/0.65；输出裁去头部 22 帧；LAB 色彩迁移 0.75 | 接续画面和音频同时锚定、裁掉重复上下文、接缝色彩与声音过渡 | 适合长视频。RGB 解码再编码有损；色彩迁移需限定在同场景 |
| U40 万物接续 | 分别保存一采、二采完整 AV latent；两路 MotionContext；22 帧上下文；一采 6 + 二采 4，再经 refiner 修改尾段；实际 latent 放大控件 2x | 完整 latent 接续、双阶段分别保留上下文、断点复用、音画同步去重 | 长视频优先参考的路线；不能把它直接当成低显存短视频提速方案 |

### 容易误判的连线与参数

**U30：三个采样节点不等于三次有效去噪。** 节点 71 的 simple 6 经节点 94 的 `ExtendIntermediateSigmas(2, 1, 0, linear)`，对范围内每个间隔细分；对 H3 0–1 轨迹是 12 个间隔，不是简单加两步。节点 99 切出前两步。节点 100 再以 step=0 切分，它的 high_sigmas 只有一个值；喂给节点 128 的这一支没有去噪间隔。真正要核查的是前两步和放大后的剩余轨迹。节点 102 取首采 noisy output 的音频，104 取 denoised output 的视频，两者角色不同，不能随意互换。

**U31：当前用的是标为“8 step”的第三套 CustomSigmas。** 节点 55 的 sigmas 经 Reroute 44 接节点 43，而不是 BasicScheduler 37。这份字符串只有 8 个 sigma 值，即 7 个积分间隔；最终窗口次数还取决于 ViggleChunkedSampler。不能按控件标题报实际计算步数。

**U32：不是完整 SelfLift-zero。** 节点 79 的 `rho=0` 且使用外部 H3 learned upscaler，作者称此为 learned latent-only lift；不走像素 VAE anchor 修正。8 步经 YC refiner `extra_steps=1` 重排低 sigma 尾段，通常为 9 个间隔；具体仍需打印实际 sigmas。`lowres_scale=0.5` 是宽高各减半，像素面积约为四分之一，不是减半。

节点 82 虽开启 TST 0.2，但节点 79 同时开启高分辨率空间分块。指定版本源码会在空间分块使视频行数不匹配时跳过 TST 并发出日志。因此不能称整段高分辨率视频都同时获得 TST 修正。该分块没有跨 tile 注意力，只保留第一块的音频预测；省显存与全局动作一致性要分别测。

**U34：节点标题是旧参数。** 节点 9 标题写 simple8/split4/refine3，实际控件是 9/4/5：首采取 simple9 的前四个间隔，再使用公开的 5 步精修 sigma 配方。节点 13 标题写 2x，实际 scale_by=1.5。节点 29 输出决定 LOW 尺寸，其面积控件为 0.5MP；不能拿 conditioning 控件中被链接覆盖的 736×416 当作运行尺寸。节点 16 的 Tail、Bias、STG、Restart 均关闭，不能把结果归功于这些功能。

Semantic Bridge 配置84启用0.1，并经83、85分别应用到LOW/HIGH conditioning。这是独立的语义条件适配变量；不能把与动作LoRA、采样、底模同时变化的结果单独归因于Bridge。首次对照建议关闭它，再独立测试。

**U35：beta 30 节点没有喂给采样器。** 节点 125 直接接 ManualSigmas 147 的 10 个积分间隔；实际计算次数还取决于 er_sde 实现。强度 6 的 fafafast 是模型特定配方，不能把强度值理解成通用“6 倍运动速度”。

**U36：不能按显示值推断输出是 8FPS。** 三个 VideoCombineV2 的 frame_rate 输入都被链接覆盖。普通音频使用 Setup 的 24FPS 流程，DurationCrop 只在自动音频时长控制启用时裁切。FaceRefine 304 及其输出 303 均 mode=4，不在此次有效精修中。节点 343 的 `wait_1st_video` 会等待首采解码/预览，适合排查，最终速度模式可考虑去掉这段额外工作。

在 JSON 标注的 FeiHou commit `a62b8544…`，公开 `__init__.py` 仅注册 Loader、FeiHouEasyH3、Output 等八类节点，没有 `FeiHouEasyH3Setup`、`SampleEnhancer1st/2nd`、`FaceRefine`。公开 nodes.py 也未包含这些类。只能核实图的连接意图以及公开代码中的卸载/裁切方法，不能声称全部核心算法已核实，也不能仅据这份图给出 AutoDL 完整可运行保证。

**U37：sigma 加密未生效。** 节点 232 的输出 links 为空，SelfLift 250 实际直接接 beta8 节点 124。上下文 PrimitiveBoolean 267 为 true，覆盖各 switch 本地显示的 false：transition_step=4，lowres_scale=0.65，使用 AddGuide+AudioGuideFeather，并进行裁头与 ColorTransfer。没有上下文时才用 6/0.5 分支。

**U40：上下文加噪默认未生效。** 一采/二采强制跳过开关 5139、5148 都为 true，两个噪声节点不是这次结果的来源。FloatConstant 5421 的 1.5 只接到 SetNode 411，后者输出为空；实际放大节点 5783 的倍率控件为 2。不能按画布上的 1.5 标签规划显存。

## 与我们当前代码的差异

| 能力 | 当前状态 | 应采取的动作 |
| --- | --- | --- |
| latent 二采、视频/音频分离、二采噪声 | 已有；beta12，默认 8+4，1.5x；音色保护时可能完整首采12再视频二采4 | 保留已有契约；日志显示实际执行次数，避免“12步”与实际16次混淆 |
| 放大后的 conditioning 网格校验 | 已有 `prepare_second_stage_guider` 和 shape 校验 | 不重复实现；新增算法接入时复用相同检查 |
| 注意力、投影、MLP 分块与动态加载 | 已有精确计算的复用/分块路线 | 低显存继续采用；不把 SolAttn、TST、KJ、现有补丁全部叠在同一层 |
| 动作 LoRA | 二采链已支持 Motion Repair、Combat 等；有文件且开关默认时自动叠加，缺少时跳过 | 这是可直接优化的默认策略：通用动作与武戏分开，实际生效项写进运行记录 |
| SelfLift 渐进采样 | 未接入 | 建立明确标为实验的可选路线，不静默替换现有二采 |
| TST | 未接入 | 独立变量测试，关闭空间分块时再验证是否实际生效 |
| 完整 AV latent 接续 | 有61秒规划/磁盘检查点代码，但缺少 `nodes/longfilm_backend.py`，不能实际完成长片生成 | U40 可作为补齐接续后端的参考；应独立立项 |
| 色彩保护 | 有硬首帧曝光保护；REF2VA 无硬首帧时只按普通参考图的全片色度统计做保守保护，T2VA 无参考仍保持原始色彩 | 与 U37 的 LAB 颜色分布迁移不同；只做一次全片色度系数，避免逐帧调色和时序闪烁 |
| VOSR2 | 已有显式分支，但当前二采菜单/校验只允许 SeedVR2 或 RTX VSR；回退后仍有路线限制 | 若接入新二采，要同时处理最终超分选择，确保所选 VOSR2 真正执行并留证 |

代码依据：`nodes/performance.py` 的 preset、`scheduler_plan`、`_apply_ref_detail_loras`；`nodes/two_stage.py` 的音频保护/网格适配/二采执行；`nodes/two_stage_assets.py` 的可选 LoRA 和模型契约；`nodes/h3_reuse_attention.py`；`nodes/color_guard.py`；`nodes/schema.py` 的 `POSTPROCESS_MODES_BY_PERFORMANCE`；`nodes/longfilm.py` 的后端导入。

### 对“动作糊、像掉帧、动作慢”的意义

这三种现象不能混成一个问题：生成帧中人物变形/拖影属于生成与时序一致性；播放不匀要查帧时间、重复帧和编码；动作进展慢要查实际生成动作与叙事时长。先看超分前的原始视频，再看超分后，同一动作同一时间点对照。

U32/U34/U35 通过采样轨迹和动作 LoRA 改变生成，可能对第一种和第三种有帮助，但可能同时影响身份、服装、镜头和声音。U37/U40 主要处理跨片段接续，不能据此保证单段武戏不变形。RIFE 可以增加中间帧，不能恢复模型没生成的动作细节；把24FPS写成48FPS也不是时序修复。

此前用户样片是24FPS、约15秒，日志中 VOSR2 将1344×768输入处理成2688×1536。原生生成、latent精修、VOSR2三个阶段都要独立计时。若大部分时间花在逐帧超分，单纯换采样器不会按同等比例缩短总耗时。

## 可选采用方案

**方案 A：只改现有路线。** 将社区 LoRA 改为可控的“关闭 / 动作连续性 / 武戏”配置；通用生活动作优先不加 Combat/GunFu/fafafast，连续性试验从 Motion Repair 0.25 开始。完善实际步数、阶段尺寸、注意力方法、耗时、实际超分器记录。复用已有内存管理。范围小，但不引入新的采样算法。

**方案 B（推荐）：方案 A + 一个 SelfLift 可选实验。** 在现有二采的高级配置中提供“标准二采 / 渐进采样（SelfLift，实验）”，保留目前稳定和低显存路线。先支持参考生成的4–5秒、24FPS、1080p输出对照，不默认开启TST、Semantic Bridge或武戏LoRA，不自动转48FPS。SelfLift先测8个间隔、transition=4、scale=0.65、rho=0、learned upscaler卸载，再测U32的6/0.5与尾段加一步；记录实际sigmas而不是依赖标题。8GB继续提供原有低显存模式，SelfLift是否能在8GB启用须独立验证。

新的高级配置需有明确依赖检查：SelfLift节点实际存在、标准Euler、正确AV格式、放大器文件已安装、关键帧与噪声mask兼容、分块与TST互斥。缺依赖提前说明，不悄悄切换算法。最终超分与采样方法分别记录；选择VOSR2就执行VOSR2，选择其他就走对应路线。未通过资源规划的组合提前拒绝，避免显示选项与实际路线不符。

**方案 C：同时重做长视频与动作迁移。** 在方案B基础上增加U40式双阶段AV归档/续接、U37式音频羽化与同场景色彩过渡，以及U31专用动作迁移。能力最多，但新模型、新依赖与测试矩阵显著增加；不建议与第一次运动清晰度优化一起上线。

推荐先按方案B做有开关的实验，验证后再决定是否改变默认；U40接续作为下一阶段。当前没有足够证据宣布这八套中任何一套是对所有模型、动作、显卡都“最靠谱”的默认配方。

## 验证与验收

1. 固定底模、编码器、VAE、参考图、提示词、24FPS、目标尺寸以及种子。选至少三类场景：普通起身行走、快速转身/挥手、双人快速交互。先用4–5秒短片，在实际AutoDL环境生成至少两个种子。
2. 基线：当前稳定20步原生输出与当前标准二采。候选逐个增加：SelfLift、Motion Repair 0.25、不开空间分块的TST 0.2。先使用相同底模与关闭其他社区LoRA的控制组，不将多个变化混成一次试验。
3. 原生/二采结果单独保存，再应用同一所选超分器；完整记录首采、lift、二采、decode、SR、encode耗时和峰值显存。不为了截图再次对首尾帧做整套超分。
4. 自动检查视频帧数、逐帧时间、实际FPS、音频长度与同步。逐段正常播放和逐帧检查脸、手、肢体、衣服纹理、背景弯曲、拖影及动作完成时间。光流差值或单帧锐度只用于定位，不能代替视觉判断。
5. 8GB验收使用实际3070或等价资源：至少4秒参考生成、原有低显存模式、一个轻量最终放大路线。没有真实执行结果就不标“8GB稳定”。高显存整帧与低显存分块要分别留视频，尤其检查跨tile动作和音频。
6. 接续阶段另测两段：上下文22/39帧分别实验，切除同样时长的图像与音频；检查重复动作、音频接缝、同场景色差、切镜/曝光变化被误校正、以及断点重跑只复用匹配模型/尺寸/配方的AV latent。

验收方向：相对同条件基线，用户看到更少的动作变形/拖影，同时总耗时和峰值显存可接受。若无可见收益、时序破坏或收益主要来自更强武戏风格，保留候选为实验，不替换默认。

## 已查阅的关键上游来源

- [SelfLift：U32指定版本说明](https://github.com/facok/comfyui-SelfLift/blob/835c3cf7919410f71f414688f049c6ca497789d0/README.md)：H3实验性适配、仅Euler、rho=0外部lifter、空间分块限制与TST互斥。
- [SelfLift：采样实现](https://github.com/facok/comfyui-SelfLift/blob/835c3cf7919410f71f414688f049c6ca497789d0/nodes.py)：按实际sigmas切分，分辨率转换后恢复噪声状态，不增加denoiser调用。
- [YC Sigma Refiner](https://github.com/yichengup/ComfyUI-YCNodes-MiniMax-H3/blob/146207295c1d2bd6b64a6552484cb6ed8be1c9ef/py/h3_sigma_refiner.py)：重排低sigma尾段并增加指定间隔数，与Core ExtendIntermediateSigmas算法不同。
- [Core sigma节点](https://github.com/Comfy-Org/ComfyUI/blob/v0.35.0/comfy_extras/nodes_custom_sampler.py)：SplitSigmas共享边界，Extend对每个符合区间的间隔插值。
- [T8二采轨迹](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/blob/8b7408e4b927b07b076b48b26f43adeb0a22613a/h3_t8/learned_latent_upscale_advanced.py)：simple前段与公开3/4/5步尾段，视频与音频时钟匹配。
- [T8双时钟](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/blob/8b7408e4b927b07b076b48b26f43adeb0a22613a/h3_t8/sampling.py)：处理旧版缩放audio velocity与新版raw velocity。当前Core已支持AV，不能据此认定我们的普通Euler有双时钟错误。
- [ClipStream说明](https://github.com/AraneaQwQ/ComfyUI-H3-ClipStream/blob/277715cedfea771a9714995e07763cebe022c2df/README.md)与[上下文代码](https://github.com/AraneaQwQ/ComfyUI-H3-ClipStream/blob/277715cedfea771a9714995e07763cebe022c2df/motion_context/nodes.py)：直接取已有AV latent，同步裁剪音画；拒绝分辨率不匹配。
- [Audio Guide Feather](https://github.com/wjc573/ComfyUI-MiniMax-H3-Audio-Guide-Feather/blob/043c631d6817e60a0437f1ab4c98b0c923e0e4da/nodes.py)：给音频guide最后的40Hz tokens增加半余弦释放曲线；U37的12 ticks对应约0.3秒。
- [FeiHou指定公开版本注册](https://github.com/FX-FeiHou/ComfyUI-FeiHou-Easy-H3/blob/a62b854489ee0388725ff6500f1a25b2908ea25a/__init__.py)与[实现](https://github.com/FX-FeiHou/ComfyUI-FeiHou-Easy-H3/blob/a62b854489ee0388725ff6500f1a25b2908ea25a/nodes.py)：公开可核实的卸载/裁切行为及关键缺失节点。
