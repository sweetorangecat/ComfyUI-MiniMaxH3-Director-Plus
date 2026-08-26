# 1080p 平衡二采实施计划

## 实施顺序

1. 在 `tests/test_vram_budget.py` 添加横竖 FHD 超采样尺寸测试，并确认旧实现失败。
2. 在 `tests/test_director.py` 添加专用输出路由、中文提示、跳过 RTX VSR 探测和 2K 回归测试。
3. 在 `tests/test_stream_output.py` 添加中心等比裁切 Lanczos、帧数与音频合成契约测试。
4. 在 `nodes/vram_budget.py` 为标准质量的精确 FHD 增加官方 768 短边首采规划。
5. 在 `nodes/director.py` 仅为该规划解析内部 `balanced_fhd_downscale` 路由。
6. 在 `nodes/stream_output.py` 以最多四帧的小批次执行中心裁切与 Lanczos 缩小，并覆盖首尾帧导出。
7. 更新 `docs/使用说明.md` 与 `docs/API说明.md`，说明无新增入参及路由元数据。
8. 运行定向测试、全量 pytest、compileall 和 `git diff --check`，随后提交并推送远程 `main`。

## 非目标

- 不修改 2K/4K、低显存二采、RIFE、AI 超分和普通 RTX VSR。
- 不增加采样步数、模型依赖或导演台控件。
- 不承诺把模型未生成的纹理恢复为原生摄影细节。
