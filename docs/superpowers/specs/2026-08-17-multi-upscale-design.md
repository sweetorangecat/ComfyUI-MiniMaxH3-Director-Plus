# U11 多路线最终放大设计

## Goal

让 U11 导演台在不增加手动连线的前提下，支持原生直出、Lanczos 快速放大、通用 AI 超分和 NVIDIA RTX VSR 四种最终输出路线，并能在 H3 生成前报告缺失的后处理依赖。

## Scope

- 保留现有导演台布局、素材上传方式、模式自动路由和 MP4 音频输出。
- 将 `postprocess_mode` 扩展为 `native`、`lanczos`、`ai_upscale`、`rtx_vsr`。
- 新增 `ai_upscale_model`，默认值为 `auto`；高级用户可指定已安装的放大模型文件。
- API 模板公开同名参数，默认仍为 `native` 和 `auto`。
- 不把 AI 超分或 RTX VSR 接入 H3 采样链路，避免改变生成内容和显存峰值。

## Routing

导演节点根据最终目标尺寸与 H3 原生尺寸生成 `postprocess_path`：

| 条件 | 路线 |
| --- | --- |
| 目标尺寸等于原生尺寸 | `native_bypass` |
| 目标尺寸小于原生尺寸 | `downscale`，使用 CPU 分块缩放 |
| 目标尺寸更大且模式为 `native` | `native_bypass`，加入“2K/4K 不会放大”的中文警告 |
| 目标尺寸更大且模式为 `lanczos` | `lanczos`，CPU 分块 Lanczos |
| 目标尺寸更大且模式为 `ai_upscale` | `ai_upscale`，逐帧使用 ComfyUI 通用放大模型 |
| 目标尺寸更大且模式为 `rtx_vsr` | `rtx_vsr`，保留现有 NVIDIA 前置能力探测 |

所有路线都保持帧数、帧率、音频和最终目标宽高不变。

## AI model selection

`ai_upscale_model=auto` 按放大倍率选择模型：

- 放大倍率不超过 2 倍：优先 `RealESRGAN_x2plus.pth`，其次匹配 `OmniSR_X2*`。
- 放大倍率超过 2 倍：优先匹配 `OmniSR_X4*`，其次 `RealESRGAN_x4plus.pth`。
- 若首选不存在，按同倍率候选排序回退；没有任何可用模型时，在导演节点阶段抛出安装提示。
- 手动指定的模型必须位于 ComfyUI `models/upscale_models`，不存在或不可加载时不得静默改用其他模型。

AI 模型在输出阶段才加载，使用后释放；每帧以小批次处理，并将模型输出再缩放到精确目标尺寸，避免 X2/X4 模型倍率与 2K/4K目标不完全匹配。

## Error handling

- 导演节点在 `ai_upscale` 模式下先扫描并解析模型路径；缺失模型在 H3 采样前报错。
- RTX VSR 继续执行真实 `NvVFX_Load` 前置探测；失败时在 H3 采样前报出驱动、SDK、GPU 和降级建议。
- 输出节点保留二次校验，防止 API 或旧工作流绕过导演节点。
- AI 超分处理异常不得自动切换为 Lanczos，避免用户得到未预期的低质量视频。

## UI and API

- 现有“最终输出”控件改为四项，保持同一规格区和现有 DOM 尺寸。
- 仅当选择 `ai_upscale` 时显示模型覆盖下拉框；默认显示“自动选择”。
- 说明文字明确区分“原生尺寸直出不会放大”和“AI/RTX 后处理会输出 2K/4K”。
- API `MiniMaxH3DirectorPlus` 增加 `ai_upscale_model` 入参；工作流 API 模板与可视化工作流使用相同默认值。

## Testing

- 单元测试覆盖四路线选择、原生大目标警告、自动模型选择、手动模型缺失、模型加载失败和精确目标尺寸。
- 流式输出测试覆盖 Lanczos 分块、AI 模型逐帧处理、帧数/音频保持和异常不降级。
- 工作流测试确认没有新增手动连接节点，输出结构和无重叠布局保持不变。
- 完成后运行完整 `pytest` 和工作流校验，并推送 `origin/main`。
