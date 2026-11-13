# “文武双全”作者配方：模型来源与文件身份

核验日期：2026-11-13。范围仅为来源、公开访问和文件身份；未下载模型正文，未运行 GPU、访问 ComfyUI 服务、SSH、部署或重启。原作者工作流与归档参数背景见 [工作流比较记录](2026-11-13-wenwu-workflow-comparison.md)。

## 结果

七个目标现均已核到文件身份：六个来自模型发布者自己的 Hugging Face 仓库，或 RunningHub 明示代表作者发布的仓库；Mystic V1 则由原平台 Civitai 的官方 API 确认作者、版本、文件 ID 和 SHA-256，再核对 Sentinel7 第三方镜像的大小与哈希。七个固定 revision 的 HF 地址均以无认证 `HEAD` 请求获得 HTTP 200，`Content-Length` 与相应元数据一致。**中文速度滑块与英文 1.1 的字节同一性仍未证实。**公开可读不等于所有用途都获许可，下面分别记录许可信息，不把缺少许可字段解释成任意使用许可。

来源结论以模型发布者的 Bilibili、RunningHub、HF、GitHub 页面和公开 API 为主；Mystic V1 经本任务明确追加的 Civitai 官方 API 核验。第三方镜像只用于身份经原平台哈希确认后的下载定位，不作为作者或许可事实的来源。搜索归档、转载模型站和第三方教程不作为结论依据。作者 [RunningHub 工作流](https://www.runninghub.cn/post/2109907990638850050)当前公开 `usedModels` 列表确认了七个目标文件名，但列表是字符串数组，没有提供逐模型下载 ID，也不能证明归档中文滑块的内容。

## 可直接定位的文件

前六项字节数与 SHA-256 来自下方列出的 HF 官方仓库 API 的 `siblings[].size`、`siblings[].lfs.sha256`；Mystic V1 来自 Civitai 官方 API，并与 HF 镜像 tree API 的 `size`、`lfs.oid` 核对。这些不是本次对模型正文重新计算的哈希。下载地址固定到本次查询到的仓库 revision；后续实际下载仍应核对 SHA-256。

| 目标文件名 | 发布者与来源 | 字节数 | SHA-256 |
| --- | --- | ---: | --- |
| `H3_Combat_V2.safetensors` | [JOKER141 / MiniMax-H3-Combat-Base-V2](https://huggingface.co/JOKER141/MiniMax-H3-Combat-Base-V2) | 155,110,280 | `5b3edb09e9d6029439badf9bc9db4ba4a355ae679b0810888de82c4793c0bac8` |
| `Motion_Repair_V2.safetensors` | [JOKER141 / MiniMax-H3-General-Motion-Continuity-Repair](https://huggingface.co/JOKER141/MiniMax-H3-General-Motion-Continuity-Repair) | 155,110,272 | `edfa0e2858caeba7be9108883f61da8525d3c58af998c94ac7a449d9603d7574` |
| `H3_Fight_Flow_V1.safetensors` | [JOKER141 / H3-Fight-Flow-Fight-Interaction-LoRA](https://huggingface.co/JOKER141/H3-Fight-Flow-Fight-Interaction-LoRA) | 155,110,296 | `579e8eca67b6d70f4db94b2db67cd938751d576ee3d72aba875dae3d911f7398` |
| `H3_speed_slider_1.1.safetensors` | [RunningHubAI / rh-h3-speed-slider-1.1-lora](https://huggingface.co/RunningHubAI/rh-h3-speed-slider-1.1-lora) | 10,093,356 | `593b074e8eb17bf9f858c3586fc12f53038dbfdd9634e146fe34a17e2661ca40` |
| `minimax_h3_dmad_full_dareties_v4_step600_comfy_fro.safetensors` | [tsolful / minimaxh3_tests](https://huggingface.co/tsolful/minimaxh3_tests) | 956,743,688 | `4d4ea608992d72303e4971f78728f587f43ecb72490f365a042ed82e37a907ce` |
| `h3_upscaler_lms_v0.1.safetensors` | [Alissonerdx / Minimax-H3-ComfyUI](https://huggingface.co/Alissonerdx/Minimax-H3-ComfyUI)，仓库路径 `latent_upscaler/` | 690,593,048 | `40d4228227146245ef2b65480b647c50222465ca125de5880f192e3259e40d59` |
| `MysticXXX_MMH3-V1.safetensors` | 原作者 alcaitiff，[Civitai 版本 3226233 官方元数据](https://civitai.com/api/v1/model-versions/3226233)；下载定位为 Sentinel7 第三方镜像 | 596,451,528 | `7dd837f4c5420d4ec59f3ff1757b3f9d264a6c0f120d65d9c4f59c77584dd0be` |

| 文件 | 固定 revision 下载 URL | 本次公开访问 |
| --- | --- | --- |
| Combat V2 | [下载](https://huggingface.co/JOKER141/MiniMax-H3-Combat-Base-V2/resolve/7b3c5b7157d66517f5bd8f5d3fa8fd2c66bc4d11/H3_Combat_V2.safetensors) | `private=false`、`gated=false`；HEAD 200 |
| Motion Repair V2 | [下载](https://huggingface.co/JOKER141/MiniMax-H3-General-Motion-Continuity-Repair/resolve/1676b352bb9f62407f76aee2ad55db2f36d04471/Motion_Repair_V2.safetensors) | `private=false`、`gated=false`；HEAD 200 |
| Fight Flow V1 | [下载](https://huggingface.co/JOKER141/H3-Fight-Flow-Fight-Interaction-LoRA/resolve/da31fb85e3142c6ae0974002e6f880df6e6095c7/H3_Fight_Flow_V1.safetensors) | `private=false`、`gated=false`；HEAD 200 |
| Speed Slider 1.1 | [下载](https://huggingface.co/RunningHubAI/rh-h3-speed-slider-1.1-lora/resolve/e1af402c4483fbf3c03894364a2403cbe316a584/H3_speed_slider_1.1.safetensors) | `private=false`、`gated=false`；HEAD 200 |
| DMAD v4 FRO | [下载](https://huggingface.co/tsolful/minimaxh3_tests/resolve/38c6ad48a4f39f7fcbfcdfa03cbc491cd62beadb/minimax_h3_dmad_full_dareties_v4_step600_comfy_fro.safetensors) | `private=false`、`gated=false`；HEAD 200 |
| LMS upscaler FP16 | [下载](https://huggingface.co/Alissonerdx/Minimax-H3-ComfyUI/resolve/db85bdf94ef5ba2467787f1e4704588fce3cabbd/latent_upscaler/h3_upscaler_lms_v0.1.safetensors) | `private=false`、`gated=false`；HEAD 200 |
| Mystic V1，已核验原平台哈希的第三方镜像 | [Sentinel7 镜像下载](https://huggingface.co/Sentinel7/h3/resolve/97eca5b9d50abbb10089b783c5845921cbc84cc5/2856467/3226233/MysticXXX_MMH3-V1.safetensors) | 镜像 `private=false`、`gated=false`；HEAD 200、`application/octet-stream` |

对应注册表证据：[Combat API](https://huggingface.co/api/models/JOKER141/MiniMax-H3-Combat-Base-V2?blobs=true)、[Motion Repair API](https://huggingface.co/api/models/JOKER141/MiniMax-H3-General-Motion-Continuity-Repair?blobs=true)、[Fight Flow API](https://huggingface.co/api/models/JOKER141/H3-Fight-Flow-Fight-Interaction-LoRA?blobs=true)、[Speed Slider API](https://huggingface.co/api/models/RunningHubAI/rh-h3-speed-slider-1.1-lora?blobs=true)、[DMAD API](https://huggingface.co/api/models/tsolful/minimaxh3_tests?blobs=true)、[LMS API](https://huggingface.co/api/models/Alissonerdx/Minimax-H3-ComfyUI?blobs=true)。

Mystic 注册表证据：[Civitai 模型 API](https://civitai.com/api/v1/models/2856467)、[Civitai V1 版本 API](https://civitai.com/api/v1/model-versions/3226233)、[Sentinel7 镜像目录 API](https://huggingface.co/api/models/Sentinel7/h3/tree/main/2856467/3226233)。

## 作者与许可边界

| 模型 | 第一手来源能确认的作者／发布身份 | 公开许可信息与限制 |
| --- | --- | --- |
| Combat V2 | JOKER141 发布；[RunningHub 代表发布的模型卡](https://huggingface.co/RunningHubAI/rh-v2-lora/blob/main/README.md)另署作者“四只兔子”，原 RH 模型 `2094270584455782402` | JOKER141 当前模型卡声明 `license=other`、`license_name=minimax-h3-community-license`，指向 [MiniMaxAI 上游 LICENSE](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)。不能称为 Apache/MIT 或无条件许可。 |
| Motion Repair V2 | JOKER141 发布；[同系列 RH 代表发布卡](https://huggingface.co/RunningHubAI/rh-bunny-general-motion-continuity-repair-lora/blob/main/README.md)署 FOURBUNNY，但该 RH 卡列的是 V1 的 `Motion_Repair.safetensors`，不能拿它替代 V2 文件证据 | [V2 所在仓库模型卡](https://huggingface.co/JOKER141/MiniMax-H3-General-Motion-Continuity-Repair/blob/main/README.md)声明同一 MiniMax H3 Community License，并链接上游 LICENSE。 |
| Fight Flow V1 | JOKER141 发布；[RH 代表发布卡](https://huggingface.co/RunningHubAI/rh-h3-fight-flow-lora/blob/main/README.md)署 FOURBUNNY，原 RH 模型 `2104849019909824514` | [JOKER141 模型卡](https://huggingface.co/JOKER141/H3-Fight-Flow-Fight-Interaction-LoRA/blob/main/README.md)声明同一 MiniMax H3 Community License。当前卡已出现 V2 更新说明，V1 仍独立公开。 |
| Speed Slider 1.1 | [HF 卡](https://huggingface.co/RunningHubAI/rh-h3-speed-slider-1.1-lora/blob/main/README.md)明示 RunningHub 代表作者发布、作者字段为 RH 用户 T8star-Aix；[RH 原页](https://www.runninghub.cn/model/public/2102025667339702273)也署 T8star-Aix | HF 卡没有独立 SPDX 许可，要求遵循原项目／上游条款；并链接 FoS 的 Motion Speed Slider 原项目 `2951265`、版本 `3342524`。这里只确认链接关系，未据此认定 T8star-Aix 是原始训练作者，也未核验该外部平台条款。 |
| DMAD v4 FRO | tsolful 自己的 HF 仓库中公开上传 | 该仓库当前没有模型卡，API `cardData=null`，文件清单没有 LICENSE；本次不能证明独立的模型许可或所有合并组成的授权条款。其他 tsolful 仓库的许可不能自动移用。 |
| LMS upscaler FP16 | Alissonerdx / alisson-anjos 自己的模型发布仓库；[作者说明](https://huggingface.co/Alissonerdx/Minimax-H3-ComfyUI/blob/main/docs/latent-upscaler.md)明确是从 LBH-123-AI 底座继续训练 2,000 步的 FP16 推理导出 | 当前仓库卡是 `mixed-asset-licenses`；[LICENSES.md](https://huggingface.co/Alissonerdx/Minimax-H3-ComfyUI/blob/main/LICENSES.md)明确专用 Character Swap / Motion Transfer 许可证不适用于 latent upscaler。当前说明只称其保留自身／上游条款，未明确列出该文件独立许可。上游 [LBH-123-AI 注册表](https://huggingface.co/api/models/LBH-123-AI/Minimax_h3_latent_Upscaler?blobs=true)标 Apache-2.0，但这不能单独证明下游微调文件的完整授权。 |
| Mystic V1 | [Civitai 原模型 API](https://civitai.com/api/v1/models/2856467)直接返回 `creator.username=alcaitiff`；Sentinel7 只是下载镜像发布者 | 同一官方 API 返回 `allowNoCredit=true`、`allowDerivatives=true`、`allowDifferentLicense=true`，`allowCommercialUse=[Image,RentCivit,Rent,Sell,SellMerge]`；未返回独立 `license` / `licenses` 字段。这里保留原始许可字段值，不把镜像仓库的其他声明移用于该模型。 |

以上为来源事实记录，不对缺失许可作推定。没有测试实际下载的模型是否包含额外内嵌许可元数据。

## Mystic V1：原平台身份与镜像核对

[Civitai 模型 API](https://civitai.com/api/v1/models/2856467)与 [V1 版本 API](https://civitai.com/api/v1/model-versions/3226233)均实际 GET 成功，HTTP 200，确认：

- 模型 `2856467` 为 `[MMH3] Mystic XXX`，作者 `alcaitiff`。
- 版本 `3226233` 为 `v1.0`、`baseModel=MiniMax H3`、`status=Published`；模型 API 中该版本 `availability=Public`。
- V1 发布时间 `2026-08-13T23:49:00.572Z`。主文件 ID `3108441`，精确名称 `MysticXXX_MMH3-V1.safetensors`，格式 SafeTensor、FP32。
- 官方 `sizeKB=582472.1953125`，按该 API 的 KiB 数值换算为 `596451528` 字节；完整 SHA-256 与上表一致。

官方文件专属地址为 [Civitai V1 文件 3108441](https://civitai.com/api/download/models/3226233?fileId=3108441)。本次无认证 `HEAD` 跟随跳转后到达 `auth.civitai.com`，最终是 HTTP 200 的 `text/html` 登录页，**不能把这个 200 当成权重可直接下载**。没有登录、绕过访问控制或下载模型正文。

[Sentinel7/h3 官方 HF 目录 API](https://huggingface.co/api/models/Sentinel7/h3/tree/main/2856467/3226233)给出的目标文件路径为 `2856467/3226233/MysticXXX_MMH3-V1.safetensors`，大小 `596451528` 字节、LFS SHA-256 `7dd837f4c5420d4ec59f3ff1757b3f9d264a6c0f120d65d9c4f59c77584dd0be`，两项均与原平台吻合。镜像 [仓库 API](https://huggingface.co/api/models/Sentinel7/h3)当前 revision 为 `97eca5b9d50abbb10089b783c5845921cbc84cc5`，公开且未设置 gate；上表固定镜像链接的无认证 HEAD 实际返回 HTTP 200、`application/octet-stream`、`Content-Length=596451528`。因此本次个人安装可定位到与原平台已发布身份匹配的镜像文件；实际下载后仍须重算正文 SHA-256。镜像发布者与原模型作者分开署名。

[lynaNSFW/mysticxxx_MM_H3](https://huggingface.co/lynaNSFW/mysticxxx_MM_H3)仍不是本次 V1 的定位依据：该卡自行写 Remix，并指向 V2；其当前 API 清单没有目标 V1 文件名。金鱼视频简介中的整合包也未访问或下载，不拿它代替上述已核验元数据。

## 文件别名：已证实与未证实

| 名称关系 | 本次结论 |
| --- | --- |
| 归档 `H3 速度滑块.safetensors` ↔ 当前公开 `H3_speed_slider_1.1.safetensors` | **未知。**只有工作流里的两个名称，没有中文文件本体或哈希，没有原作者提供的重命名记录。不得把中文旧名加入“确定同一权重”的自动回退候选。 |
| `H3_speed_slider_1.1.safetensors` ↔ 同仓库 `_r` 变体 | **不同文件。**API 中 `_r` 为 10,093,479 字节、SHA-256 `84f94b49d9f6b778cdc17fdc37b71a5f8f400c3475d94cb5ffce547756a53b99`；不能替换 1.1 的身份校验。 |
| `Motion_Repair.safetensors` ↔ `Motion_Repair_V2.safetensors` | **不同文件。**两者都是 155,110,272 字节，但 V1 SHA-256 是 `a678a3657cefd5c2c2712f892da2f71be53c3f252e335557e918854f8abc579f`，V2 见上表。文件大小相同不能证明同版。 |
| DMAD `...comfy.safetensors` ↔ `...comfy_fro.safetensors` | **不同文件。**原仓库 full 版本为 3,913,285,640 字节、SHA-256 `ca34641d06c4e714d27a8540fd20721ef5daf54e9d58fa8be169d33b2d2699a9`，FRO 见上表；本次目标为作者公开清单中的 FRO。 |
| LMS `...v0.1.safetensors` ↔ `...v0.1_fp32.safetensors` | **不同文件。**作者说明区分 FP16 推理导出和 FP32 训练归档；FP32 为 1,381,153,616 字节、SHA-256 `7b70711ebf13051bac4b637247f2e6f0db5e1c6da88af3f28ad0165c6483c8c4`。 |
| LMS FP32 `experiments/h3_upscaler_sharpness_2000steps_v0.1_fp32.safetensors` ↔ `latent_upscaler/h3_upscaler_lms_v0.1_fp32.safetensors` | **已证实当前仓库两个路径字节相同。**二者 API 大小和 SHA-256 均为上一行 FP32 值；这不证明 FP16 与 FP32 相同。 |

上述差异由相应 [Speed Slider API](https://huggingface.co/api/models/RunningHubAI/rh-h3-speed-slider-1.1-lora?blobs=true)、[Motion Repair API](https://huggingface.co/api/models/JOKER141/MiniMax-H3-General-Motion-Continuity-Repair?blobs=true)、[DMAD API](https://huggingface.co/api/models/tsolful/minimaxh3_tests?blobs=true)、[LMS API](https://huggingface.co/api/models/Alissonerdx/Minimax-H3-ComfyUI?blobs=true)直接返回。

## 集成前的本机只读观察与实施边界

只读列出 `D:/ComfyUI_windows_portable-G313/ComfyUI/models/loras/` 的直接子文件时，没有七个目标文件；该目录只有既有图像 LoRA 及其 info/预览文件。`D:/ComfyUI_windows_portable-G313/ComfyUI/extra_model_paths.yaml` 本次检查不存在。这不是运行时“未注册”的证明：没有查询正在运行的服务，也没有读取其启动参数或递归扫描其他模型库。

追加有界本地检查：

- 在 `E:/AIVideo/ComfyUI-MiniMaxH3-Director-Plus/outputs/` 中按文件名扫描，只有 `quality_research_round6_20261109/assets/weights/` 的无关 Wushu v7 权重，没有本次七个目标。项目根目录本身没有 `models/`、`models_cache/`、`model_cache/`、`.cache/`、`cache/`。
- `E:/models_cache`、`E:/model_cache`、`E:/AI/models_cache`、`E:/AIVideo/models_cache`、`E:/AIVideo/model_cache`、`E:/ComfyUI/models`、`E:/ComfyUI_windows_portable/ComfyUI/models` 均不存在。
- 当前环境指定 `HF_HOME=F:/AI/huggingface`、`HF_HUB_CACHE=F:/AI/huggingface/hub`。该 hub 的九个缓存目录没有本次 H3、Mystic、tsolful 或 Alissonerdx 相关仓库条目。默认 `C:/Users/sugar/.cache/huggingface/hub` 也没有相关仓库条目。没有扫描整块 E/F 盘。
- 既有 `outputs/diagnostics/lms_trial_20261101/remote_inventory.json` 和 `outputs/diagnostics/033143/remote_inventory.json` 记录过远端相关模型名称。这些是过去保存的证据，不能证明当前远端文件存在，更不能当成本机缓存文件；本次没有探测远端。

因此本次有界本地检查没有找到可复用的目标权重正文，没有可以报告的“本机实际文件哈希”。这不排除其他未检查目录存有模型。

本次为七个文件提供了固定下载身份，其中 Mystic V1 的文件身份由原平台确认、下载地址为哈希已匹配的第三方镜像。不能据此宣称完整作者配方已安装、中文滑块已同版替换或配方画质已验收。归档四个动作节点仍是 `mode=4` 的旁路保存状态；如何启用作者权重组合应保留明确的用户选择和实际加载记录，不能由下载成功推导成片合格。

## 后续模型准备（2026-11-13）

随后实施阶段已将上述七个固定版本文件下载到实际本机 ComfyUI 的 `models/loras/minimax/` 与 `models/latent_upscale_models/minimax/`，逐个重新计算 SHA-256，全部与表中原始元数据一致。只读 SafeTensors header 检查也确认了每个文件的张量索引与实际 payload 长度相符。严格配方解析器对实际文件检查结果为六项 LoRA 加一项 LMS，无缺失。

这项检查只证明本机权重文件已准备并校验，不证明插件或 Sol-Attn 内核已注册、实际 GPU 应用成功或成片质量合格；没有运行 GPU，也没有更新或重启远程服务。后续仅用 SSH 只读查询了远端队列，未干预作业。安装收据和本机路径留在仓库外，模型正文不随代码提交。
