# ComfyUI MiniMax H3 Director Plus

独立的中文优先 MiniMax H3 导演台插件。它保留 U10 的深色分区风格和稳定布局，集中提供 4-15 秒时长、横竖分辨率、四种种子模式、最多 9 张 REF2VA 图片、3 路编号音色、I2VA/FL2VA 的 H3 原生 reference 路由、按 FL/Reference 隔离的训练型 3D latent 二采、显存安全预算、本机能力状态和简化 HTTP API。

安装后导入：

- `ComfyUI\user\default\workflows\minimaxH3\U11-MiniMaxH3-导演台Plus-中文增强版.json`

只需导入这一个中文增强版工作流。API 模板由插件内部的 `templates/u11_api.json` 管理，不需要另行导入。完整用法见 `docs/使用说明.md`，接口字段见 `docs/API说明.md`，异常处理见 `docs/故障排查.md`。原 U10 与 `ComfyUI-DaSiWa-Nodes` 不会被修改。
