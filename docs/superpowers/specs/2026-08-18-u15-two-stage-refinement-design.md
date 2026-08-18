# U15 二阶段质量采样设计

## 目标

将 U15 的核心能力收进 `ComfyUI-MiniMaxH3-Director-Plus`，让 U11 在不增加用户手动连线的前提下支持全片二阶段细节重建。

## 方案

新增 `MiniMaxH3TwoStageSampler` 自定义节点，替代 U11 的单一 `MiniMaxH3MemoryAwareSampler`。节点始终执行第一采样；仅当导演指南的性能预设为 `quality_two_stage` 时，执行以下第二阶段：

1. 使用 ComfyUI 原生 `LTXVSeparateAVLatent` 分离视频和音频 latent。
2. 只对视频 latent 进行 1.5 倍空间放大，保持时间维和音频 latent 不变。
3. 使用第一阶段 sigma 轨迹的低噪声尾段执行 6 步二采，避免重新走完整高噪声轨迹。
4. 使用相同的 H3 guider、sampler 和随机种子派生噪声，重新合并视频与音频 latent。

二采采样失败时必须返回明确错误，不静默降级为伪清晰结果。非 `quality_two_stage` 预设必须原样旁路，保证现有低显存、极速和稳定质量行为不变。

## 路由与显存

- 新预设允许 T2VA/I2VA/FL2VA/L2VA/REF2VA 的高显存质量路线使用；I2VA/FL2VA/L2VA/T2VA 走 H3 原生音色参考时也可以使用。
- `低显存`、`极速4步`、`参考图加速` 和 Fish S2 锁定音色路线不启用二采，避免把对白生成和 latent 细化叠加到同一次任务中。
- 二采目标由原生采样尺寸乘以 1.5 得到；最终 2K/4K 仍由现有最终输出后处理路线编码，避免把 2K/4K 整段 latent 常驻显存。
- 音频 latent 不放大、不重采样，确保原生 H3 音频时间轴和同步不被改变。

## 可观察性

指南中写入 `two_stage_enabled`、`two_stage_scale`、`two_stage_steps` 和 `two_stage_status`。输出说明明确显示“U15 二阶段 latent 细化”或“二阶段已旁路”。

## 验证

- 单元测试验证预设路由、非目标预设旁路、AV latent 分离/合并、视频空间放大不改变时间和音频形状。
- 工作流验证确保 U11 仍为单一导演台、没有可见重叠和手动连接要求。
- 不启动 ComfyUI；在当前内置 Python 中运行完整 pytest。
