# ComfyUI MiniMax H3 Director Plus

独立的中文优先 MiniMax H3 导演台插件。它保留 U10 的深色分区风格和稳定布局，集中提供 4-15 秒时长、横竖分辨率、四种种子模式、最多 9 张 REF2VA 图片、3 路编号音色、I2VA/FL2VA 的 H3 原生 reference 路由、按 FL/Reference 隔离的训练型 3D latent 二采、显存安全预算、本机能力状态和简化 HTTP API。

2026-09-06：默认预设更名为“智能画质（自动适配）”，保留 1080p / 2K / 4K 三档。标准 2K 在满足显存和依赖预算时支持 4–15 秒，自动选择分块二采直出或二采后 SeedVR2；加入二采 OOM 有限重试与等比输出修正。显卡实测边界、升级方法见 [本次交付说明](docs/U11清晰度增强版交付说明.md)。

安装后导入：

- 本项目 `examples/U11-MiniMaxH3-导演台Plus-中文增强版-AutoDL混合底模版-清晰度增强版.json`。

2026-09-14：内置 FaceRefine 人脸跟踪、裁剪重绘、原音轨锁定与合成节点。
上方示例已接入右侧人物修复分支，原版输出自动作为修复输入；另附左右人物和仅跟踪示例。
首次使用需安装可选依赖与模型，见 [人脸修复说明](examples/face_refine/README.md)。
当前通过代码和静态接线检查，画质提升与耗时仍待 GPU 实测。

只需导入这一个中文增强版工作流。API 模板由插件内部的 `templates/u11_api.json` 管理，不需要另行导入。完整用法见 `docs/使用说明.md`，接口字段见 `docs/API说明.md`，异常处理见 `docs/故障排查.md`。原 U10 与 `ComfyUI-DaSiWa-Nodes` 不会被修改。
