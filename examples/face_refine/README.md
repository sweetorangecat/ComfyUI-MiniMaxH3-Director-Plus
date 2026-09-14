# Director Plus 内置人脸修复

人脸跟踪裁剪、原帧 latent 注入、原音轨锁定、逐帧重绘强度与合成节点现由本项目提供。
不需要另装 `ComfyUI-H3-FaceRefine` 或 `ComfyUI-H3-NativeAudioLock`，也不需要上传独立测试包。

## 安装

更新当前自定义节点项目，在启动 ComfyUI 的 Python 环境中运行：

```bash
python custom_nodes/ComfyUI-MiniMaxH3-Director-Plus/tools/install_face_refine.py --comfy /root/ComfyUI
```

把 `/root/ComfyUI` 换成实际目录；Windows 便携版使用 `python_embeded/python.exe`。
脚本只安装本项目可选依赖和两个模型，不克隆独立插件。
安装后重启 ComfyUI。原有 U11 使用的 H3 混合底模、文本编码器、视频/音频 VAE 仍需可用。
修复分支另外需要 VideoHelperSuite 和提供 `MiniMaxH3MemoryEfficientSageAttentionPatch` 的已有节点包。
检测器为 `models/ultralytics/bbox/face_yolov8m.pt`，修复 LoRA 为
`models/loras/minimax/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors`。

## 主工作流

`examples/U11-…清晰度增强版.json` 已接入右侧人物修复分支：

1. U11 先保存原版视频。
2. 原版输出的 `filename` 自动接入修复分支的视频路径。
3. 跟踪右侧人物，768×768 脸部重绘，8 步，固定种子 42，`denoise=0.4`。
4. 修复脸部合成回原分辨率，输出到 `output/video/FaceRefine/right…mp4`。

默认分支处于启用状态，会增加一次脸部采样。可将整个人脸修复分组设为 Never/禁用，只生成原版。
原版输出请使用视频容器（MP4/MKV/WebM），不要改成 Animated WebP/AVIF。
右侧是首个有效画面中的位置，之后按连续性跟踪；左侧人物在跟踪节点的 `select` 改为 `left_most`。
两个人依次修复时，用右侧修复结果作为左侧流程的输入，不能只把两份都指向原片。

## 已有视频与跟踪预览

- `FaceRefine-right-tracking-only.json` / `FaceRefine-left-tracking-only.json`：只检测和预览，不进行 H3 采样。
- `FaceRefine-right-1080p.json` / `FaceRefine-left-1080p.json`：完整后处理流程，修改读取节点的视频路径即可。
- 文件名中的 1080p 指目标使用场景；实际保留原片分辨率，低分辨率输入不会在此强制升成 1080p。

默认关闭 `identity_track`，使用位置选择与连续跟踪，减少额外模型依赖。
多人交叉时先检查预览；需要人脸身份辅助时另装 `insightface` 和兼容的 ONNX Runtime，
开启 `identity_track` 并连接身份参考图。它可能下载额外模型。
高清角色图也可替换 ImageFromBatch 到 H3 参考图的连线，提高身份参考质量。

最终保存直接使用输入视频的音频和帧率，修复不会重新生成最终音轨；MP4 编码会重新编码音频，
不承诺压缩码流逐字节一致。音频锁定参与重绘条件，但口型、身份稳定和画质提升仍需实际对照。
裁剪重绘会额外占用内存与显存；这条分支加载整段原分辨率视频，不属于流式低内存处理。
长视频建议先截短验证，不能将原输出节点的低显存承诺直接套用到此分支。

目前已做代码单元测试与 JSON 静态校验，尚未在远程 GPU 实测画质和耗时。
脸型变化或接缝明显时，可固定其他参数把 denoise 从 0.4 降到 0.3 比较。

重新生成项目示例与主流程接线：

```bash
python tools/integrate_face_refine.py
```
