# 第三方声明

MiniMax H3 Director Plus 是独立实现的 ComfyUI 插件，不修改也不替代
`ComfyUI-DaSiWa-Nodes`。

前端交互和视觉风格参考并改编自 DaSiWa 的 `MiniMaxH3Director`，原项目：

- Copyright (c) darksidewalker and contributors
- Source: https://github.com/darksidewalker/ComfyUI-DaSiWa-Nodes
- License: GNU General Public License version 3

本插件同样以 GPL-3.0 授权。具体许可文本见 `LICENSE`。

人脸跟踪、裁剪、合成与逐帧重绘后端随本插件分发：

- Source: https://github.com/Carasibana/ComfyUI-H3-FaceRefine
- Revision: d8521d14fe0d721d80cd9417fff5a559cbc21aba
- Copyright (c) 2026 Carasibana
- License: MIT，全文见 `nodes/_vendor/FaceRefine-LICENSE.txt`
- `nodes/_vendor/h3_face_refine.py` 保留上游实现，项目适配位于 `nodes/face_refine.py`。
- 只注册本项目 `MiniMaxH3Face*` 节点，不注册上游视频选择接口或前端脚本。
