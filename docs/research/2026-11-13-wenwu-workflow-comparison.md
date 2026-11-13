# 视频“文武双全”工作流：第一手来源核验

核验日期：2026-11-13。本文记录用户指定视频及其作者工作流的证据，不代替本项目代码、运行时模型链或画面质量的验收。

## 精确视频与作者工作流

| 项目 | 核验结果 |
| --- | --- |
| 视频 | [MiniMax H3终极篇、新二采、新加速、文武双全又快又好](https://www.bilibili.com/video/BV1LKJm6VELc/) |
| 作者 | YZ_金鱼，Bilibili UID `14843708` |
| 发布时间 | 北京时间 `2026-10-15 20:42:48`，API `pubdate=1792068168` |
| 时长 | `829` 秒，即 `13:49` |
| 视频标识 | `aid=117444974805808`，`cid=42768140874` |
| 简介指定的工作流 | YZ金鱼-MiniMax H3 终极篇-文武双全 |

以上由 [Bilibili 官方公开视频信息 API](https://api.bilibili.com/x/web-interface/view?bvid=BV1LKJm6VELc) 直接返回，成功状态为 `code=0`；搜索引擎收录的原视频页亦与标题、时间和简介相符。

视频简介指向[国内作者工作流 `2109907990638850050`](https://www.runninghub.cn/post/2109907990638850050)、[海外作者工作流 `2110354436651560962`](https://www.runninghub.ai/zh-cn/post/2110354436651560962/)，并另列[国内 AI 应用](https://www.runninghub.cn/ai-detail/2110357037929558018)、[模型及工作流整合包](https://pan.quark.cn/s/6af352275c72)、[官方提示词 skills](https://github.com/MiniMax-AI/MiniMax-H3/tree/main/skills)。简介明确致谢四只兔子的武打 LoRA 组合；这些地址是简介中的链接，本次没有下载整合包或核验其完整内容。[来源：视频官方 API](https://api.bilibili.com/x/web-interface/view?bvid=BV1LKJm6VELc)

国内 RunningHub 页面署名同为 YZ_金鱼，标题相同，并反向链接该 Bilibili 视频，形成双向关联。其公开页面 SSR 数据中的 `publishTime` 为 `2026-10-14T20:57:45+08:00`，这是工作流页面时间，不是视频发布时间。[来源：作者工作流](https://www.runninghub.cn/post/2109907990638850050)

## 原作者页面公开的具体组成

直接读取上述国内 RunningHub 页面的公开 HTML `__NUXT_DATA__`，得到以下 `customNodes` 和 `usedModels`。这不是根据视频标题猜测。

| 方面 | 页面公开清单中的精确项目 | 该证据能说明的范围 |
| --- | --- | --- |
| 二采路线 | `SelfLiftH3Sampler`、`h3_upscaler_lms_v0.1.safetensors` | 作者流包含 SelfLift 及其 latent 放大模型；不能仅由名称确定完整步数和强度 |
| 加速/采样配方 | `minimax_h3_dmad_full_dareties_v4_step600_comfy_fro.safetensors`、`MiniMaxH3SigmaShift` | 包含 DMAD v4 和视频/音频 sigma shift 节点 |
| 细节 LoRA | `MysticXXX_MMH3-V1.safetensors` | 包含 Mystic V1；不是 V5 |
| 文武/动作组合 | `H3_Combat_V2.safetensors`、`Motion_Repair_V2.safetensors`、`H3_Fight_Flow_V1.safetensors`、`H3_speed_slider_1.1.safetensors` | 支持核对 Combat V2、动作修复 V2、Fight Flow V1 和速度滑块的文件身份；清单本身不能证明全部同时启用 |
| 注意力相关 | `BlockSparseAttention`、`ModelAttentionBackend` | 包含相关节点；清单不能证明目标机器已经安装或实际应用某个内核 |
| 参考和音画 | `MiniMaxH3ReferenceToVideo`、`LoadAudio`、`VHS_LoadVideo`、`VAEDecodeAudio`、`VHS_VideoCombine` | 包含多模态参考和音画输出所需节点；不能因此推定无限时长或原生双音色已验收 |
| 底模候选 | `minimax_h3_hybrid_fl2va_ref2va_b25-49-int8.safetensors`，以及 `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | 混合底模等候选确实在作者公开依赖清单中；实际选择还需工作流连线/运行信息 |

来源为[原作者工作流公开页](https://www.runninghub.cn/post/2109907990638850050)内嵌的页面数据。本次该数据的 `workflowContent=null`，因此没有从公开 HTML 获得完整可执行图；没有绕过登录或权限去获取下载内容。

## 本地已归档副本：可用于参数核对，但不是视频逐字证据

本次只读检查了先前保存的[同名作者工作流 JSON](E:/AIVideo/ComfyUI-MiniMaxH3-Director-Plus/outputs/diagnostics/quality_research_round4_20261109/community/YZ金鱼-MiniMax+H3+终极篇-文武双全.json)。文件含 101 个节点，SHA-256：

```text
2fd8ba1bb77d29a5ed2348127982d32ffc3ac2c1db46a105fa0128cfd35535ed
```

该哈希与同目录既有 `downloaded_workflows_audit.json` 相符；其主要节点、LoRA 名称与当前作者页面公开清单相互印证。不过，副本已有本地医生/患者参考提示词，速度滑块名称也与当前在线清单不同。因此下面只称为“归档副本保存值”，不称为视频中演示的原始默认参数，不声称与当前线上下载逐字节一致。

| 节点 | 归档副本保存值 |
| --- | --- |
| `192 UNETLoader` | 混合 `fl2va_ref2va_b25-49-int8` 底模 |
| `53 LoraLoaderModelOnly` | DMAD v4 `0.8`，`mode=0` |
| `400 LoraLoaderModelOnly` | Mystic V1 `0.35`，`mode=0` |
| `255 KSamplerSelect` | `euler` |
| `261 BasicScheduler` | `simple`、`12` 步、`denoise=1` |
| `333 MiniMaxH3SigmaShift` | `shift_video=12`、`shift_audio=3` |
| `397 SelfLiftH3Sampler` | `cfg=1`、`transition_step=6`、`lowres_scale=0.6`、`rho=0`、`highres_tiling=false`，放大模型为 `h3_upscaler_lms_v0.1.safetensors` |
| `342/344/345/343` 动作 LoRA | Combat V2 `0.65`、Motion Repair V2 `0.35`、Fight Flow V1 `0.35`、中文命名速度滑块 `3`；343 的强度实际连到 366，其保存值也是整数 `3`；四个节点保存状态均为 `mode=4`，不能当成已经启用的链 |
| `332 BlockSparseAttention` | 保存的后端选项为 `sol-attn`；不是本机内核生效证明 |

特别需要区别：这份归档值是 **simple 12 步**，不是凭“U22”名字就能归为 beta 10 步；Mystic 保存强度为 **0.35**。是否有同等功能、是否精确复现这份图，以及实际画质是否通过，是三个需要分别验证的问题。

副本节点 `402` 的作者备注还区分 simple/beta 的高动态取舍，并提示高分辨率分块可能影响运动；这是保存于工作流的使用建议，不是本次实测结论，也不是要求更改本项目设置的指令。[本地来源：归档 JSON](E:/AIVideo/ComfyUI-MiniMaxH3-Director-Plus/outputs/diagnostics/quality_research_round4_20261109/community/YZ金鱼-MiniMax+H3+终极篇-文武双全.json)

## “文武”含义的直接上游说明

金鱼简介致谢的四只兔子，在 [Combat Base V2 发布视频](https://www.bilibili.com/video/BV1rTbf6CEtj/)简介中，把该模型的作用同时覆盖到对白表演的身体动作、声音动态、镜头/状态连续性，以及持续攻防、受击状态和多人物战斗调度。因此这里不能把 Combat V2 简化成只对武戏有用的开关；这些都是作者对训练目标和效果的说明，不能替代本项目验收。[来源：该视频官方 API，作者 UID `49344623`](https://api.bilibili.com/x/web-interface/view?bvid=BV1rTbf6CEtj)

同一发布简介建议动作描述放在对白前面，先采用无触发词方式；需要更强攻击或终结动作时再使用 `prfight2` / `prfin1`。其量化效果数字是作者自述，本次未做独立复现，不将其作为本项目的收益承诺。[来源：Combat V2 官方视频 API](https://api.bilibili.com/x/web-interface/view?bvid=BV1rTbf6CEtj)

四只兔子的[另一份公开多参工作流说明](https://www.runninghub.cn/post/2094373732667187201)区分 Combat 的动作表现与动作修复的连续性/稳定性职责，并建议二采使用较弱的动作修复。这是被致谢模型作者的上游使用建议；它不是金鱼这份 SelfLift 工作流的逐项参数定义，不能把两份图的配置直接混为一谈。

## 与当前导演台代码的对照

对照版本为 `1efc599`，是本次作者配方集成**之前**的基线；表内“当前”均指该基线，不代表后续版本。后续明确选用的完整配置见[文武双全作者配方](../文武双全作者配方.md)，本次文件来源核验见[权重记录](2026-11-13-wenwu-author-assets.md)。以下是基线源码可证实的能力和默认配置，不是实际模型加载或成片质量验收。

| 能力 | 当前代码 | 与作者配置的关系 |
| --- | --- | --- |
| SelfLift、learned latent upscaler、DMAD/Mystic | 已有选择和调用路线；DMAD 未注册时可回退 Turbo v4；Mystic V1 配方强度为 0.45，附加 V5 会走避免重复叠加的分支 | 已借鉴核心路线；不等于逐项复现归档参数 |
| 动作增强 `combat` | `Motion_Repair.safetensors` 或原中文旧别名 0.25，加 `H3_Combat_V2.safetensors` 0.5 | 只接入部分动作组合 |
| Motion Repair V2 / Fight Flow V1 / 速度滑块的自动组合 | 当前动作候选没有 Motion Repair V2；没有 Fight Flow V1 或速度滑块的自动配方 | 不能把现有“武戏”选项叫作原作者全套文武配方 |
| 原附加 LoRA 节点 | 可手工选择已注册 LoRA，追踪已连接的 DaSiWa / 原生 LoRA loader | 可选通用入口不代表三项缺失组合已经配置并验收 |
| 启用状态 | `community_lora_profile` 默认 `off`；原 U11 路由器默认“全部关闭”。SelfLift 自身会请求紧凑 DMAD/Mystic 配方，但不会自动开启全套武打 LoRA | 不是任何普通生成都会加载武打组合 |
| 缺失权重 | 可选动作/细节 LoRA 未注册或应用失败时，记录跳过；需查看 `community_loras_applied` / `detail_loras_applied` 及实际日志 | 界面可选不构成权重已经实际应用的证据 |
| 图像及音色输入 | 保留原九个普通图槽和三路 H3 原生音色；SelfLift 不支持 Fish 克隆分支 | 当前已有图像/音色能力不等于支持全部声音工作方式 |
| 用户参考视频输入 | Director 写入 `ref_videos={}`、`ref_video_audios={}`，没有对应上传控件；Guide 仅透传这些字典 | 作者流的 VHS 视频参考模块尚未接入主导演台；原生 AV 历史续拍另有用途 |
| 采样参数 | 当前 SelfLift beta 10 间隔、6+4 拆分；调用 lowres_scale=0.6，Mystic 配方 0.45 | 归档图是 simple 12、transition6、Mystic0.35；本次没有对两组配方做同源 GPU 对照 |

源码位置：[采样与动作组合](../../nodes/sampling_experiments.py)、[模型资源及 DMAD/Mystic 配方](../../nodes/two_stage_assets.py)、[实际 LoRA 链选择和缺失跳过](../../nodes/performance.py)、[导演台输入与空视频字典](../../nodes/director.py)、[输入指南透传](../../nodes/guide.py)、[原附加 LoRA 追踪](../../nodes/additional_loras.py)、[默认输入定义](../../nodes/schema.py)。

因此能确认的是：核心采样路线和部分文戏/武戏能力已接入，完整武打自动组合及参考视频入口尚未对齐。不能从选项名称推出已经获得作者演示的质感、速度、复杂攻防连续性或多人身份稳定性。

## 本次核验限制

- 本次核到的是官方视频元数据、简介、原作者工作流的公开页面数据、已归档本地图和被致谢模型作者的说明；**没有观看完整视频，也没有取得完整字幕/转写**。
- 普通网页抓取原视频返回 HTTP 412，但官方 view API 成功。[官方 player API](https://api.bilibili.com/x/player/v2?bvid=BV1LKJm6VELc&cid=42768140874)返回 `need_login_subtitle=true`、`subtitle.subtitles=[]`，只能说明本次公开请求没有字幕，不能说视频一定没有字幕。
- 公开视频标题含速度/质量评价，不能作为本机耗时、自然质感、双人身份稳定性或武打连续性的验收证据。
- 本次不运行 GPU、不下载模型、不修改源目录的工作流或研究记录、不部署服务。
