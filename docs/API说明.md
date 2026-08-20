# API 说明

## 简化接口

接口前缀为 `/h3-director-plus`，返回 JSON，错误统一为 `{ "ok": false, "error": { "code": "...", "message": "中文说明" } }`。

### `GET /h3-director-plus/schema`

返回版本化字段 schema 和中文名称。当前公开字段为：

`mode`、`prompt`、`duration`、`aspect_ratio`、`resolution_preset`、`custom_width`、`custom_height`、`seed`、`first_image`、`last_image`、`references`、`voice_mode`、`voice_reference_audio`、`voice_reference_audios`、`voice_reference_names`、`target_dialogue`、`reference_transcript`、`fish_model_path`、`ref_image_size`、`performance_preset`、`postprocess_mode`、`rtx_quality`、`ai_upscale_model`。

`duration` 为 4-15 秒整数。`references` 最多 9 张；`voice_reference_audios` 最多 3 路，`voice_reference_names` 按相同下标绑定角色。旧字段 `voice_reference_audio` 继续兼容，等价于音色参考 1。

API 的 `seed` 是本次请求使用的明确整数。画布中的固定、递增、递减、随机属于 ComfyUI 客户端的连续运行状态，不作为无状态 API 入参公开；需要批量生成时，由调用方为每个请求明确传入 seed。

`performance_preset` 支持 `稳定质量`、`质量优先加速`、`质量优先二采样`、`极速4步`、`参考图加速`、`低显存` 和 `自定义`，实际可用项会按模式及音色路由过滤。`质量优先加速` 固定 20 步、只启用 SageAttention、使用 ComfyUI 动态分层加载、关闭 Turbo LoRA 与 EasyCache；Sage 不可用时保持原生 20 步并返回回退说明。`质量优先二采样` 固定 U09 图像空间二采路线：首阶段 3 步，VideoVAE 解码后按 1.5 倍放大并重新编码，再以 2 步、denoise 0.2 低噪声重绘。该档位强制 `postprocess_mode=rtx_vsr`，只允许 HIGH/ULTRA；若传入 native、lanczos 或 ai_upscale，API 会在 H3 采样前拒绝请求。首阶段画布会根据最终目标自动选择，按二采后到 RTX VSR 的线性放大比例约 1.9 倍控制，不再固定用 0.20 MP 硬放大到 2K/4K；15 秒 2K 通常约 0.50 MP 首阶段，4K 会接近 H3 原生画布上限。该档位不提供给低显存或 Fish S2 锁定路线，显存不足时应改用“质量优先加速”或“低显存”。

`postprocess_mode` 控制最终视频输出：`native`（原生尺寸直出）、`lanczos`（CPU Lanczos 快速放大）、`ai_upscale`（ComfyUI 通用 AI 超分）或 `rtx_vsr`（RTX VSR AI 细节重建）。但 `performance_preset=质量优先二采样` 时只允许 `rtx_vsr`，不能把二采重绘与 OmniSR/Lanczos 串联；其他预设仍一次只执行选中的一种最终路线。`ai_upscale_model` 为 `auto` 或 `models/upscale_models` 中已安装的模型名；`auto` 会按目标倍率优先选择 X2/X4 模型。`rtx_quality` 可选 `HIGH` 或 `ULTRA`，仅在 `rtx_vsr` 时生效。目标尺寸与 H3 原生尺寸相同会自动旁路，目标更小走高质量缩小。最终只保存一个视频；AI 模型或 RTX VSR 依赖缺失时会在生成前明确报错，不会静默退回另一种算法。

API 中的 `resolution_preset` 表示请求的最终输出目标，支持 `2K QHD`（2560×1440）和 `4K UHD`（3840×2160）。选择 `低显存` 后，H3 原生采样尺寸会按时长降低；四种 `postprocess_mode` 都由同一个流式 MP4 输出节点保存，不需要修改 API 模板或连接第二个输出节点。`native` 输出实际原生尺寸，其余路线在目标更大时按对应算法逐帧放大。

`voice_mode = "fish_lock"` 时只使用 `voice_reference_audios` 的第 1 路作为 Fish 音色样本，`target_dialogue` 是要新生成的对白，`reference_transcript` 是样本音频原文（建议填写），`fish_model_path` 默认使用 `s2-pro-w4a16 (auto download)`。Fish 失败会返回明确错误，不会静默回退到 H3 原生音色。

### `GET /h3-director-plus/status`

探测本机模型和节点能力。Fish S2 分成 `node_available`、`model_available`、`available` 三个状态，不会把空模型目录报成可用。`postprocess.rtx_vsr` 会显示 `node_available`、`dependency_available` 和中文安装提示；状态探测不会下载模型或初始化 VSR。

启用 RTX VSR 前，在 ComfyUI 自带 Python 中安装 DaSiWa 依赖，然后重启 ComfyUI：

```powershell
D:\ComfyUI_windows_portable-G313\python_embeded\python.exe -m pip install -r D:\ComfyUI_windows_portable-G313\ComfyUI\custom_nodes\ComfyUI-DaSiWa-Nodes\requirements.txt
```

还需要 NVIDIA Broadcast SDK 提供的 `nvvfx`/`VideoSuperRes` 运行库。验证命令：

```powershell
D:\ComfyUI_windows_portable-G313\python_embeded\python.exe -c "import nvvfx; print(nvvfx.VideoSuperRes)"
```

### `POST /h3-director-plus/assets`

使用 multipart 字段 `asset` 上传图片、视频或音频。文件只会写入 ComfyUI `input/h3-director-plus/`，拒绝 `../`、目录分隔符和未知扩展名。响应中的 `asset` 值可直接填入 `first_image`、`last_image`、`voice_reference_audio` 或 `references`。

### `POST /h3-director-plus/generate`

请求会先归一化、计算 32 对齐分辨率、补丁化 U11 API 模板，再进入 ComfyUI 标准 prompt 队列。关键返回值包括 `prompt_id`、`number` 和 `resolved_backend`：无音色的 I2VA/FL2VA 为 `fl2va_model`，带 H3 音色 reference 或 REF2VA 为 `ref2va_model`。

示例：

```powershell
$body = @{
  mode = "FL2VA"
  prompt = "角色从左向右走到门口，在 00:03 说：我们回家。"
  duration = 5
  aspect_ratio = "16:9"
  resolution_preset = "0.83 MP"
  first_image = "h3-director-plus/first.png"
  last_image = "h3-director-plus/last.png"
  voice_mode = "h3_reference"
  voice_reference_audios = @("h3-director-plus/voice-a.wav", "h3-director-plus/voice-b.wav")
  voice_reference_names = @("角色甲", "角色乙")
  performance_preset = "参考图加速"
  postprocess_mode = "rtx_vsr"
  rtx_quality = "HIGH"
  seed = 123
} | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8188/h3-director-plus/generate -Method Post -ContentType application/json -Body $body
```

### `POST /h3-director-plus/validate`

使用与 `generate` 相同的 JSON 入参，完成请求归一化、模板补丁和 ComfyUI 节点校验，但不进入执行队列、也不加载模型。部署 API 前建议用它检查本机节点、模型文件名和输入素材路径。

### `GET /h3-director-plus/jobs/{prompt_id}`

查询任务状态。`state` 为 `pending`、`running`、`completed` 或 `not_found`；完成后 `history` 与标准 ComfyUI `/history/{prompt_id}` 使用同一结果结构，视频文件位置在输出节点记录中。

## 标准 `/prompt` API

插件内部的 `templates/u11_api.json` 是标准 API prompt 模板，不需要单独导入第二个工作流。模板中的节点标题是稳定查找键：`快速设置 / API 入参`、`API 首帧图片`、`API 尾帧图片`、`API 音色参考音频1` 到 `3`、`API 参考图1` 到 `9`、`预览与输出`。程序化调用时优先使用简化 `generate`，需要标准队列控制时再把模板补丁后的 `prompt` 原样 POST 到 `/prompt`。

标准 API 的关键字段仍是英文节点输入名，中文名称只放在 schema、节点标题和文档中，便于 API 稳定化和中文用户排错。
