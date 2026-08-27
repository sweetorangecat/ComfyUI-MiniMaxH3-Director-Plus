# API 说明

## 简化接口

接口前缀为 `/h3-director-plus`，返回 JSON，错误统一为 `{ "ok": false, "error": { "code": "...", "message": "中文说明" } }`。

### `GET /h3-director-plus/schema`

返回版本化字段 schema 和中文名称。当前公开字段为：

`mode`、`prompt`、`duration`、`aspect_ratio`、`resolution_preset`、`custom_width`、`custom_height`、`seed`、`first_image`、`last_image`、`references`、`voice_mode`、`voice_reference_audio`、`voice_reference_audios`、`voice_reference_names`、`target_dialogue`、`reference_transcript`、`fish_model_path`、`ref_image_size`、`performance_preset`、`postprocess_mode`、`rtx_quality`、`ai_upscale_model`、`motion_smoothing`、`audio_loudness`。

`duration` 为 4-15 秒整数。`references` 最多 9 张；`voice_reference_audios` 最多 3 路，`voice_reference_names` 按相同下标绑定角色。旧字段 `voice_reference_audio` 继续兼容，等价于音色参考 1。

API 的 `seed` 是本次请求使用的明确整数。画布中的固定、递增、递减、随机属于 ComfyUI 客户端的连续运行状态，不作为无状态 API 入参公开；需要批量生成时，由调用方为每个请求明确传入 seed。

性能预设还支持 `高清快速（v4 8步）`、`参考高清（原生20步）` 和 `参考极速（官方4步）` 三个显式路由档位。前者仅出现在 FL/T2V 后端，使用 `minimax_h3_turbo_v4_step600_ema.safetensors`、LoRA strength 1.0、8 步 simple/Euler 单采；后两者仅出现在 REF2VA/H3 音色参考后端，分别是原生 20 步 + SageAttention，以及官方 Ref2VA Turbo 4 步。Fish S2 继续排除训练型 latent 二采，旧预设的默认关系保持不变。

`performance_preset` 支持 `稳定质量`、`质量优先加速`、`质量优先二采样`、`极速4步`、`参考图加速`、`低显存`、`低显存二采` 和 `自定义`，实际可用项会按模式及音色路由过滤。`质量优先加速` 固定 20 步、只启用 SageAttention、使用 ComfyUI 动态分层加载、关闭 Turbo LoRA 与 EasyCache；Sage 不可用时保持原生 20 步并返回回退说明。两个二采预设自动解析训练型路线：FL/T2VA/硬首尾帧后端使用 `trained_latent_fl`，依次应用 `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors` 与 `minimax_h3_fl2v_turbo_4step_v1.1_768p_comfyui_bf16.safetensors`；REF2VA 或 H3 原生音色参考后端使用 `trained_latent_ref`，两阶段应用 `minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors`，并通过 `H3SigmaRefiner` 强化尾段 sigma。两条路线都先完成匹配 LoRA 首采，再把 AV latent 分为视频/音频，仅用 `minimax_h3_latent_upscaler_3d_bf16.safetensors` 对 24 通道视频 latent 做约 1.5 倍训练型放大，保留原音频后执行匹配低 sigma 二采。`低显存二采` 额外限制为 4–6 秒和 FHD 像素预算：1080p 约 0.46MP/0.37MP/0.31MP 首采（4/5/6 秒），再由 `RealESRGAN_x2plus.pth` 单路重建到 FHD；时长越长，神经细节基准越低。启动前至少需要 6.0GB 空闲显存，采样期间使用 LOW_VRAM。Fish S2 与训练型二采互斥；Sage、EasyCache、RIFE 不叠加到该路线。

`performance_preset` 的 API 字段保持不变。依赖官方 LightX2V LoRA 的性能档位，只有在 ComfyUI 内置加载器实际新增模型补丁且 `patch_delta>0` 后才会进入采样。`POST /generate` 可能已经成功返回 `prompt_id` 并入队；官方 LoRA 会在任务执行到加速节点时验证，若验证不通过，任务会在首个 H3 采样节点前终止，调用方应从任务历史或执行错误读取原因。常见的缺文件、损坏文件和零补丁错误会包含解析后的 LoRA 文件名。上次加载失败后重新执行加速节点会再次验证，不会因旧的 `turbo_lora_applied=false` 绕过检查。

`postprocess_mode` 控制最终视频输出：`native`（原生尺寸直出）、`lanczos`（CPU Lanczos 快速放大）、`ai_upscale`（ComfyUI 通用 AI 超分）或 `rtx_vsr`（RTX VSR AI 细节重建）。无新增必填入参：质量二采仍传 `performance_preset=quality_two_stage`、`postprocess_mode=rtx_vsr`。2K/4K 控制器会固定 `HIGHBITRATE_ULTRA` 并只执行一次 RTX VSR；不再启用会产生彩条/花屏风险的 `DEBLUR_LOW` 双效果链。精确 `1080p FHD` 是内部特例，无新增 API 入参：28GB+ 且至少 24GB 空闲显存时规划 `1344×768 -> 2016×1152 -> 1920×1080`；20–24GB 级显卡或空闲显存不足 24GB 的更高档显卡，在至少保留 18GB 空闲显存时规划 `1280×704 -> 1920×1056 -> 1920×1080`，返回 `vram_safety_tier=16_24gb_fhd`。两种 FHD 路线都返回 `postprocess_path=balanced_fhd_downscale`、`upscale_method=aspect_lanczos_downscale`，中心等比裁切并 Lanczos 对齐，不探测或执行 RTX VSR。`低显存二采` 只允许 `ai_upscale`，并按二采后的实际倍率自动优先选择 X2 模型，当前推荐和默认命中 `RealESRGAN_x2plus.pth`。除 FHD 最终缩小特例外，两种二采都不会叠加另一种放大器。`ai_upscale_model` 为 `auto` 或 `models/upscale_models` 中已安装的模型名；自动解析使用神经二采尺寸而不是首采尺寸，避免错误选择 X4。`rtx_quality` 的公共枚举为 `HIGH`、`ULTRA`、`HIGHBITRATE_ULTRA`，质量优先二采样的公共请求仍规范化为 `HIGHBITRATE_ULTRA`，但 FHD 特例不会实际加载 VSR。最终只保存一个视频；实际需要的 AI 模型或 RTX VSR 依赖缺失时会在生成前明确报错，不会静默回退。

两个训练型二采预设的最终 H.264 编码值都固定为 `min(请求 quality, 16)`；U11 和 API 模板继续显示兼容旧路线的默认值 20，执行二采时由输出节点自动解析为 16。编码结束后通过 H.264 stream-copy 写入 BT.709 limited-range VUI，不重新编码视频帧或音频；任务 UI 元数据会返回 `encode_quality` 与 `color_metadata`。其他性能预设保持原编码值 20。

`motion_smoothing` 控制最终运动平滑，可选 `off`、`rife_x2`，默认 `off`。后端继续接受旧值 `auto`，但会按关闭处理。`rife_x2` 只允许与 RTX VSR 搭配；两个训练型二采预设和普通低显存模式都固定关闭。其他兼容预设中，RIFE 按相邻帧逐对处理，不会在内存中构造完整双倍帧视频。

`audio_loudness` 控制最终编码前的音频处理，默认 `auto`：将过小的 H3 音轨峰值安全提升到 -1.5 dBFS，最大增益 30 dB；静音保持静音。设为 `original` 时保持原始波形。该处理只影响最终 MP4 音轨，不改变 H3 参考音色或生成内容。

API 中的 `resolution_preset` 表示请求的最终输出目标，支持精确 `1080p FHD`（1920×1080）、`2K QHD`（2560×1440）和 `4K UHD`（3840×2160）。4K 不是 H3 原生采样。普通低显存路线支持 4–15 秒并把最终目标限制在 1080p；`低显存二采` 支持 4–6 秒和 FHD 像素预算，时长越长会自动降低首采网格；20–24GB 级显卡在至少有 18GB 空闲显存时可执行 4–15 秒精确 FHD 保守二采，但 2K 仍只开放 8 秒以内；28GB 以上才允许通过前置检查后请求长视频 2K/4K。所有路线仍由同一个流式 MP4 输出节点保存。

`GET /schema` 的 `resolved_outputs` 描述运行前可观察结果：`resolved_two_stage_route`、`first_stage_width/height`、`second_stage_width/height`、`final_upscale_scale_x/y`、`final_upscale_scale`、`max_final_vsr_scale`、`vram_safety_tier`、`quality_basis` 与 `required_assets`。实际 `generate` 响应和任务元数据还可查看 guide/status 的 `rtx_deblur_mode`、`postprocess_path`、`source` 与 final尺寸，展示“首采 -> 神经二采 -> 最终输出”，不要只显示最终分辨率标签。

`voice_mode = "fish_lock"` 时只使用 `voice_reference_audios` 的第 1 路作为 Fish 音色样本，`target_dialogue` 是要新生成的对白，`reference_transcript` 是样本音频原文（建议填写），`fish_model_path` 默认使用 `s2-pro-w4a16 (auto download)`。Fish 失败会返回明确错误，不会静默回退到 H3 原生音色。

### `GET /h3-director-plus/status`

探测本机模型和节点能力。Fish S2 分成 `node_available`、`model_available`、`available` 三个状态，不会把空模型目录报成可用。`postprocess.rtx_vsr` 会显示 `node_available`、`dependency_available` 和中文安装提示；状态探测不会下载模型或初始化 VSR。

启用 RTX VSR 前，必须在当前 ComfyUI Python 中安装官方 `nvidia-vfx` wheel。以下 DaSiWa 依赖安装命令可作为安装入口，然后重启 ComfyUI：

```powershell
D:\ComfyUI_windows_portable-G313\python_embeded\python.exe -m pip install -r D:\ComfyUI_windows_portable-G313\ComfyUI\custom_nodes\ComfyUI-DaSiWa-Nodes\requirements.txt
```

Linux 请使用 NVIDIA VSR 支持驱动分支 `570.190+`、`580.82+` 或 `590.44+`，不需要 Windows NVIDIA Broadcast SDK。Windows 请检查 NVIDIA 驱动、官方 `nvidia-vfx` wheel 与 NVIDIA Broadcast SDK/Video Effects 运行库兼容。导演台会在 H3 开始前，用真实 `640×360 → 1280×720` 单帧执行 RTX VSR `load + run`；失败会提前停止。验证命令：

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
  motion_smoothing = "rife_x2"
  audio_loudness = "auto"
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
