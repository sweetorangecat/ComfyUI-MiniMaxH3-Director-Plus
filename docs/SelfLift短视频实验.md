# SelfLift 短视频实验

本次借鉴 U32 的渐进分辨率采样，作为可选实验接入。标准采样仍是默认值；低显存显卡继续使用“低显存”档。八套工作流的具体比较见 [研究报告](research/2026-10-14-community-workflows-review.md)。本地验证覆盖参数、调用、连线和输出路由，尚未在 AutoDL 做实际视频画质、耗时和峰值显存对照。

## 建议先这样试

1. 在现有导演台选择 **REF2VA → 质量优先二采样**。SelfLift 支持无音色、H3 原生参考和完整复用音频；Fish 克隆音色仍走标准路线。旧自建工作流试用 SelfLift 前需连接二采节点的 `video_vae` 输入；已经连接时无需更换工作流。
2. 先用 **5 秒、1080p、固定种子**和一个连续动作；放大方式选你已经安装的 VOSR2、SeedVR2 或通用 AI。
3. 展开“二采高级设置”，动作增强保持“关闭”。先生成“标准二采”作为对照，再只把二采方法改为 **SelfLift 渐进采样（实验）**。
4. 两者比较运动中的脸、手、衣服边缘和背景稳定性，同时记录总耗时。再单独试“动作连续性”，不要第一次就叠加武戏配置。

SelfLift 使用 REF2VA、24 FPS，时长沿用导演台的 **4–15 秒**，支持横屏、竖屏和所选超分输出 **2K/4K**，不再额外限制为 4–5 秒或 FHD。启动门槛为总显存至少 20GB、空闲至少 18GB；长视频的显存需求仍需实测。Fish 克隆音色继续使用标准采样。“二采高级设置”会保留展开或收起状态，修改选项后不会自动收起。

## 实際执行配方

SelfLift 的用途是把前半程计算放在较小网格上，再放大继续采样，以尝试减少计算量。它不会自动增加画质，也不是最终视频超分。U22 配方使用混合 INT8 底模、DMAD v4（0.8）和 Mystic（0.45），调度为 beta 10 个积分间隔、标准 Euler：前 6 步在宽高约 0.6 倍网格采样，经 learned latent upscaler 放大后，在目标网格完成后 4 步。目标 H3 网格按比例限制在 1344×768 像素面积以内；最终尺寸由所选超分完成，2K/4K 不代表 H3 原生以该分辨率采样。

`rho=0.6, w_min=0.5, w_max=1.0`，使用 learned latent upscaler 与像素/VAE 锚点的局部一致性修正，并在放大后卸载 learned 模型。2026-10-26 的同素材对照中，原先 `rho=0` 的纯 learned 过渡在超分前就会产生白色细丝和划痕；局部修正消除了检查帧中的明显伪影，比纯像素锚点保留更多细节，但增加一次 VAE 解码/编码，不能作为提速承诺。

初始 latent 必须是目标网格，低分辨率前段由 SelfLift 内部建立。高分辨率阶段优先启用 U22 的 Sol-Attn 与 MiniMax MLP 分块；新版 SelfLift 节点同时使用 `global_stream` 全局上下文分块、4096 token 批处理和 4GiB 权重缓存，缺少这些可选参数时自动兼容旧节点。Sol-Attn 内核不可用时回退现有 Sage/精确注意力路线，不旁路训练型二采。它是外部 learned lift 与像素锚点的混合配方，不是论文的完整 SelfLift-zero。

日志、指南和输出元数据记录 `sampling_actual_method=selflift`、`sampling_executed_steps=10`、实际节点路线与采样耗时、6+4 步拆分、高分辨率分块参数，以及实际 `selflift_rho`、`selflift_w_min`、`selflift_w_max` 和放大模型。旧图从路由器选择匹配模型时继承 BasicGuider 的完整 AV 采样对象和视频/音频 shift，避免模型恢复为 12/3 而调度器仍为 10/5。标准二采也记录实际执行步数，便于同种子对照。

## 安装依赖

使用 [facok/comfyui-SelfLift](https://github.com/facok/comfyui-SelfLift)，本次核对的源码版本为 `835c3cf7919410f71f414688f049c6ca497789d0`。需要包含 `SelfLiftH3Sampler` 和 `upscaler_unload` / `highres_tiling` 参数的版本；缺失或过旧会在生成前报错。

在 ComfyUI 根目录、运行 ComfyUI 的同一个 Python 环境执行。以下适用于尚未安装该插件的环境：

```bash
git clone https://github.com/facok/comfyui-SelfLift.git custom_nodes/comfyui-SelfLift
git -C custom_nodes/comfyui-SelfLift checkout 835c3cf7919410f71f414688f049c6ca497789d0
if [ -f custom_nodes/comfyui-SelfLift/requirements.txt ]; then
  python -m pip install -r custom_nodes/comfyui-SelfLift/requirements.txt
fi
```

已安装时先检查该目录的版本和本地改动，避免重复安装两个副本。安装后重启 ComfyUI 并刷新页面。现有工作流连接完整时继续使用即可。

模型沿用现有二采资产：

| 资产 | 放置位置 |
| --- | --- |
| `minimax_h3_dmad_full_dareties_v4_step600_comfy.safetensors`（U22 优先；也接受带 `minimax/` 前缀的注册路径） | `models/loras/` |
| `minimax_h3_dmad_full_dareties_v4_step600_comfy_fro.safetensors` 或 `minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors`（兼容回退） | `models/loras/` |
| `MysticXXX_MMH3-V1.safetensors` | `models/loras/` |
| `h3_upscaler_lms_v0.1.safetensors`（U22 优先）或 `minimax_h3_latent_upscaler_3d_fp16/bf16*.safetensors` | `models/latent_upscale_models/` |
| `ComfyUI-SolAttn_triton` 的 `SolAttnPatch`（可选）与 KJNodes 的 `MiniMaxChunkFeedForward` | `custom_nodes/` |
| 所选最终超分的节点与模型 | 按 VOSR2 / SeedVR2 / 通用 AI 的安装说明 |

现有 BF16 latent upscaler 也可使用，无需为了试验重复下载 FP16。DMAD 缺失时会选择已安装的 v4 兼容 LoRA；Sol-Attn 或新版 MLP 分块缺失时保留训练型二采并回退到现有兼容注意力，不会静默改成普通单采。当前 Sol-Attn 插件如果没有 `extra_tokens` 输入，报告会把 `sol_attn_extra_tokens_applied` 标为 `false`，不会伪报 U22 的 256 已生效。模型下载说明见 [使用说明](使用说明.md)。旧自建工作流还需要把视频 VAE 接到 `MiniMaxH3TwoStageSampler` 的 `video_vae` 输入；新版官方示例两条分支已接好。标准二采可不接这条线，SelfLift 必须接。

## 动作增强配置

| 选择 | 实际应用 |
| --- | --- |
| 关闭 | 不加可选社区 LoRA |
| 动作连续性 | Motion Repair，强度 0.25 |
| 武戏 | Motion Repair 0.25 + H3 Combat V2 0.5 |
| 旧版社区配置 | 沿用兼容加速路由和环境开关控制的旧链 |

这些配置只作用于质量优先二采样。新导演节点默认关闭，不会因模型目录里有文件就自动堆叠。实际加载项目写入指南和日志。普通走路、起身先试动作连续性；武戏配置用于打斗对照。模型来源和现有下载工具见使用说明的社区 LoRA 部分。

## 低显存和超分

8GB 显卡先选择 **低显存 + 768p + 4 秒 + 通用 AI 超分**，使用已安装的 X2 模型。该配置沿用现有低显存采样路线，SelfLift 暂不对 8GB 开放；本地 CPU 测试不能保证实际显存峰值。

放大方式与采样方法分别选择。选 VOSR2 就调用 VOSR2；选 SeedVR2 就调用 SeedVR2；选通用 AI 就调用通用 AI。所选依赖缺失会报错。采样输出等于或大于目标尺寸时，也执行所选超分，再适配最终尺寸，因此额外超分可能增加耗时；画质收益需要同种子对照确认。

## 本次验证记录

2026-10-24：便携版 ComfyUI Python 运行时通过 1420 项 Python 测试、Node.js 前端 80 项测试和 Python 编译检查。测试覆盖 U22 路由、DMAD/Mystic 选择、Sol-Attn 失败回退、SelfLift 10 步 6+4 调用、高分辨率分块参数和执行报告证据。测试验证参数与调用契约，不能替代实际 GPU 视频、画质、耗时和峰值显存验收；AutoDL 当前未运行，因此本轮没有新增 5090 A/B 数据。

2026-10-26：修正旧图的 AV shift 交接，启用局部像素/VAE 锚点校正。399 项相关回归与 10 项文档测试通过；5090 上通过原节点生成了带 H3 克隆音色的 2560×1464 样片，检查帧中未再发现原对照的明显白丝。4 秒设置实际输出 107 帧、4.458 秒，不代表完成 15 秒多空间块验收。对照与运行证据见[排查记录](research/2026-10-26-selflift-glass-artifacts.md)。
