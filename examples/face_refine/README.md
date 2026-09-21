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

`examples/U11-…清晰度增强版.json` 使用导演台统一开关，默认 `off`。
关闭时懒加载选择器直接输出原始解码帧，不运行修复模型或检测器。无需更改整个分组的运行模式。

1. 在“选择修复人物”节点填写原始角色图的 Picture 编号，一次只修复一个人物。
2. 三视图或多人图另接同一人的清晰单人图到 `identity_override`，仅供身份跟踪；原有条件编号保持不变。
3. `auto` 启用身份跟踪及 PySceneDetect 自动切镜。原始裁剪只作重绘底图，条件沿用导演台完整提示词、图片与音频编号。
4. 768×768、8 步、`denoise=0.4` 重绘后合成回原始解码帧，由原输出节点编码，保留原音轨来源。

主流程需要 `insightface`、`onnxruntime`、`scenedetect`，身份模型首次使用可能下载权重。
没有普通参考图的模式必须提供 `identity_override` 并选 Picture 1；该图作为新增身份参考，原提示词应与此编号一致。
InsightFace 面向人脸，不保证能识别拟人动物；检测与身份匹配可能出错，必须先查看跟踪预览。
预览默认禁用以避免开关关闭时仍执行检测；手动启用预览会独立运行跟踪，检查后应恢复禁用。
修复强度和原图质量都会影响脸型与细节；本次新接线尚未进行 GPU 画质对照，也未验证 8G 长视频可用性。

## 已有视频与跟踪预览

- `FaceRefine-right-tracking-only.json` / `FaceRefine-left-tracking-only.json`：只检测和预览，不进行 H3 采样。
- `FaceRefine-right-1080p.json` / `FaceRefine-left-1080p.json`：修改视频路径，并在 LoadImage 选择原始单人角色图。
- 文件名中的 1080p 指目标使用场景；实际保留原片分辨率，低分辨率输入不会在此强制升成 1080p。

独立示例默认关闭 `identity_track`，使用位置选择与连续跟踪；与主 U11 的身份跟踪不同。
多人交叉时先检查预览；需要人脸身份辅助时另装 `insightface` 和兼容的 ONNX Runtime，
开启 `identity_track` 并连接身份参考图。它可能下载额外模型。
独立示例已改用原始角色图，不再使用 ImageFromBatch 截取模糊原帧作身份参考。
独立示例保留单人物修复提示词；完整原始提示词与多参考编号由主 U11 分支传递。

最终保存直接使用输入视频的音频和帧率，修复不会重新生成最终音轨；MP4 编码会重新编码音频，
不承诺压缩码流逐字节一致。音频锁定参与重绘条件，但口型、身份稳定和画质提升仍需实际对照。
裁剪重绘会额外占用内存与显存；这条分支加载整段原分辨率视频，不属于流式低内存处理。
长视频建议先截短验证，不能将原输出节点的低显存承诺直接套用到此分支。

历史旧接线完成过 RTX 5090 实测：158 帧、1080p、24fps 跟踪全部成功；首次修复约 207 秒，含模型加载。
当前配方在该近景原片上更软，未验证到画质提升，详见 `docs/FaceRefine验证.md`。
脸型变化或接缝明显时，可固定其他参数把 denoise 从 0.4 降到 0.3 比较。

重新生成项目示例与主流程接线：

```bash
python tools/integrate_face_refine.py
```
