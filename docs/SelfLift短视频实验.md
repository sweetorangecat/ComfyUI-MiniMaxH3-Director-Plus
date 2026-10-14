# SelfLift 短视频实验

本次借鉴 U32 的渐进分辨率采样，作为可选实验接入。标准采样仍是默认值；低显存显卡继续使用“低显存”档。八套工作流的具体比较见 [研究报告](research/2026-10-14-community-workflows-review.md)。本地验证覆盖参数、调用、连线和输出路由，尚未在 AutoDL 做实际视频画质、耗时和峰值显存对照。

## 建议先这样试

1. 在现有导演台选择 **REF2VA → 不使用音色 → 质量优先二采样**。旧自建工作流试用 SelfLift 前需连接二采节点的 `video_vae` 输入；已经连接时无需更换工作流。
2. 先用 **5 秒、1080p、固定种子**和一个连续动作；放大方式选你已经安装的 VOSR2、SeedVR2 或通用 AI。
3. 展开“二采高级设置”，动作增强保持“关闭”。先生成“标准二采”作为对照，再只把二采方法改为 **SelfLift 渐进采样（实验）**。
4. 两者比较运动中的脸、手、衣服边缘和背景稳定性，同时记录总耗时。再单独试“动作连续性”，不要第一次就叠加武戏配置。

SelfLift 使用 REF2VA、无音色条件、24 FPS，时长沿用导演台的 **4–15 秒**，支持横屏、竖屏和所选超分输出 **2K/4K**，不再额外限制为 4–5 秒或 FHD。启动门槛为总显存至少 20GB、空闲至少 18GB；长视频的显存需求仍需实测。音色参考继续使用标准采样。“二采高级设置”会保留展开或收起状态，修改选项后不会自动收起。

## 实際执行配方

SelfLift 的用途是把前半程计算放在较小网格上，再放大继续采样，以尝试减少计算量。它不会自动增加画质，也不是最终视频超分。使用匹配的 turbo v4 模型，两阶段复用同一模型。调度固定为 simple 8 个积分间隔、标准 Euler：前 4 步在宽高约 0.65 倍网格采样，经训练型 H3 latent 放大后，在目标网格完成后 4 步。目标 H3 网格按比例限制在 1344×768 像素面积以内；最终尺寸由所选超分完成，2K/4K 不代表 H3 原生以该分辨率采样。

`rho=0`，使用 learned latent upscaler，并在放大后卸载。初始 latent 必须是目标网格，低分辨率前段由 SelfLift 内部建立。本阶段强制使用模型克隆上的 PyTorch 精确注意力，并跳过继承的 Sage/TST 注意力替换；不启用空间分块或额外 sigma refiner。运行记录的 `sampling_attention_kernel=pytorch` 可用于核对。它是外部 learned lift 配方，不是论文的完整 SelfLift-zero。日志、指南和输出元数据记录 `sampling_actual_method=selflift`、`sampling_executed_steps=8`、实际节点路线与采样耗时，以及两阶段各4步。标准二采也记录实际执行步数，便于同种子对照。

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
| `minimax_h3_turbo_v4_step600_ema.safetensors` 或 `minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors` | `models/loras/` |
| `minimax_h3_latent_upscaler_3d_fp16.safetensors`（优先）或 `minimax_h3_latent_upscaler_3d_bf16.safetensors` | `models/latent_upscale_models/` |
| 所选最终超分的节点与模型 | 按 VOSR2 / SeedVR2 / 通用 AI 的安装说明 |

现有 BF16 latent upscaler 也可使用，无需为了试验重复下载 FP16。模型下载说明见 [使用说明](使用说明.md)。旧自建工作流还需要把视频 VAE 接到 `MiniMaxH3TwoStageSampler` 新增的 `video_vae` 输入；新版官方示例两条分支已接好。标准二采不要求补这条线。

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

2026-10-14：853项相关Python检查、43项前端检查通过，示例与API模板连线校验、Python编译和JS语法检查通过，并完成独立规格与代码审查。Python检查使用真实CPU Torch、固定版本的ComfyUI AV容器/拼接实现，并对模型加载、GPU采样等外部边界提供测试替身；这些结果验证参数和调用契约，不能替代实际ComfyUI/AutoDL视频、画质、耗时与峰值显存验收。
