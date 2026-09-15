# H3 动作替换（Viggle）

主工作流文件是 `U11-导演台Plus-全功能-动作替换模式-高显存.json`。它以原有高显存导演台为主，保留原有 H3 模式、性能路由、音色、二采、超分和输出节点，并额外合并 U31 的 Viggle 动作替换分支。

`U11-动作替换-Viggle-H3.json` 仅作为原始 U31 节点基线保留；日常使用请打开全功能合并版。

1. 用 `VHS_LoadVideo` 读取原视频，连接 `IMAGE` 到 `MiniMaxH3DirectorPlus.action_video` 和 `MiniMaxH3ActionReplacementConditioning.cond_video`。
2. 用 `LoadImage` 读取目标人物图，连接到 `action_reference_image` 和适配节点的 `ref_image`。
3. 用 `ViggleTextCondLoader` 输出 `TEXT_COND`，连接适配节点。
4. 连接 H3 视频 VAE 到适配节点的 `vae`。
5. 导演台切换为 `ACTION_REPLACE`（界面显示“动作替换（Viggle H3）”），把导演指南连接到适配节点的 `director_guide`。
6. 适配节点的 `Viggle动作条件` 连接到 `ViggleChunkedSampler.cond_set`。采样器、Viggle 专用 UNET/LoRA、BlockSparseAttention 和合成节点沿用 U31 工作流。
7. 原视频音轨连接到适配节点的 `source_audio`，再将 `原视频音轨` 连接到最终视频合成节点的 `audio`。未连接时只生成画面，不会凭空生成音频。

动作替换的语义是：原视频提供动作、节奏、姿态和镜头运动，目标人物图提供身份与外观；输出是目标人物重新生成并执行原动作的视频，不是简单抠图贴图。音色替换属于后期可选步骤，默认不改变原视频音轨。

依赖：`ComfyUI-Viggle-Animate-H3`（节点包 `comfyui-viggle-animate-h3`），以及 U31 使用的 Viggle H3 底模、LoRA 和 `fixed_embed_fwd_anyframe.safetensors`。
