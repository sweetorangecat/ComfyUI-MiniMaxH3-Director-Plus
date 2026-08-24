# RTX VSR 真实单帧前置检查设计

## 问题与证据

当前 `probe_vsr_capability()` 使用 `64×64` 输出尺寸，只创建并加载 `VideoSuperRes` 效果，不提交输入帧。服务器因此在正式 H3 采样前返回 `NvVFX_Load ... code -14`。

同一服务器使用 NVIDIA 官方调用方式实测：

- GPU：NVIDIA GeForce RTX 4090 D
- 驱动：580.105.08
- `nvidia-vfx`：0.1.0.1
- `640×360 → 1280×720`：HIGH 与 ULTRA 均完成 `load + run`

因此故障是无效的 `64×64` 探测造成的假失败，不是 GPU、驱动或 ULTRA 质量不受支持。

## 修改范围

- 前置检查固定创建一张 CUDA `float32` RGB 测试帧，尺寸为 `3×360×640`，像素范围为 `[0, 1]`。
- 使用用户实际选择的 HIGH 或 ULTRA 创建现有 `VsrFrameProcessor`。
- 目标尺寸固定为 `1280×720`，执行一次与正式输出相同的 `load + run + DLPack clone` 路径。
- 校验输出形状为 `720×1280×3`。
- 无论成功或失败，都关闭 VSR 效果、删除测试输入/输出并释放 PyTorch 未占用的 CUDA 缓存。
- 成功日志记录质量、GPU 和探测尺寸。
- 失败仍在 H3 采样前抛出，不允许静默退回 Lanczos 或基础缩放。

本次不修改：导演台布局、HIGH/ULTRA 选项、正式逐帧 RTX VSR 实现、H3 采样、二采倍率、Sigma、工作流 JSON、API 字段及最终输出尺寸。

## 数据流

```text
用户选择 RTX VSR + HIGH/ULTRA
  → 创建 640×360 CUDA RGB 测试帧
  → 创建 VsrFrameProcessor（目标 1280×720）
  → load NVIDIA VSR 效果
  → run 一帧并立即复制 DLPack 输出
  → 验证 720×1280×3
  → 关闭效果并释放测试显存
  → 通过后才允许 H3 开始采样
```

## 错误提示

- Linux：提示检查 NVIDIA Linux 驱动支持分支、`nvidia-vfx` 版本及实际 GPU；不再把 Windows Broadcast SDK 作为 Linux 首要安装建议。
- Windows：保留驱动、`nvidia-vfx` 与 NVIDIA Broadcast/Video Effects 运行库兼容提示。
- 错误中继续包含质量、GPU编号和 NVIDIA 原始错误。

## 测试与验收

- 单元测试必须证明前置检查使用 `3×360×640` 输入和 `1280×720` 输出，而不是 `64×64`。
- 必须证明前置检查真实调用 `process()`，并校验输出形状。
- HIGH 与 ULTRA 都沿用用户选择，不自动降级。
- `process()` 或效果创建失败时，错误继续在 H3 采样前向上抛出。
- 成功和失败均关闭效果并清理临时张量。
- 运行 RTX VSR、导演台、流式输出相关测试及全量测试。
- 不启动本地 ComfyUI；服务器更新后以相同 RTX 4090 D 环境重新运行 ULTRA，前置检查应通过并进入 H3 采样。

