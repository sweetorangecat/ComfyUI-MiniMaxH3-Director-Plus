# VOSR2 配置与短片对比

默认使用 `legacy`，另两档需要手动选择；系统不会根据卡名或一次计时自动切换配置。原生 TE 节点保留原调用参数，已验证的 Linux 兼容层可按下面的能力声明和实测显存条件批量处理独立分块。

Linux兼容层的定点维护例外：`tools/patch_linux_vosr2.py` 可对已经检查过的 `TE-Speed-VOSR2-linux-compat` 源码增加 DiT 分块批量处理，默认先检查、显式 `--apply` 才写入并保留 `.bak`。它保持512分块、重叠、种子、颜色对齐和逐帧重建算法，视频过程复用CUDA分配缓存。遇到不匹配的源码结构会拒绝修改。只在该后端明确声明支持、释放模型后实测总显存≥30GiB且空闲≥24GiB时，`legacy` 将 `image_batch` 设为4；这里表示四个独立DiT小块，不是四帧跳算或时序缓存。Windows原生TE节点、低显存和手动比较档继续原值。该机制不新增用户选项。

这个兼容层的模型缓存不受 ComfyUI 自带的模型管理器控制。维护补丁同时声明专用释放函数，导演台在整段超分结束或报错后调用，避免模型一直占用显存并影响下一条 H3 生成；不会在每个帧批次之间反复卸载。`backend_released` 记录释放结果，收尾异常保留原始生成错误。未声明此能力的后端不受影响。

| 配置键 | 上游质量 / tile 策略 | tile / VAE tile | frame / DINO batch | 显存策略 | 用途 |
|---|---|---|---|---|---|
| `legacy` | `speed` / `auto` | 512 / 1024 | 2 / 2 | `auto` | 原有行为；完全空闲的 30GB+ 显存自动把独立空间 tile batch 提到 8 |
| `manual_baseline` | `manual` / `tiled` | 512 / 1024 | 2 / 2 | `staged` | 固定 tile 的保守对比基线 |
| `experimental_32gb` | `manual` / `tiled` | 512 / 1024 | 4 / 4 | `staged` | 明确实验档，只测试更大的帧与 DINO 批量 |

除上述兼容层批量处理外，普通显存档的 `image_batch` 保持1；已验证的 Linux 兼容层在实测总显存至少 30 GiB、空闲至少 29 GiB 时，`legacy` 自动使用 8 个独立空间 tile 的批量调用，空闲不足回到 4 或 1。全部配置保持 tile overlap=32、VAE overlap=32、`temporal_cache=False`、`torch_compile=False`、`vae_encode_amp=False`、wavelet 颜色对齐。frame batch 同时传给 Settings 与 Video，避免 Video 覆盖所选配置。`legacy` 的 `speed/auto` 仍允许上游自行调整有效设置；日志中的 `requested_config` 是实际传入参数，`upstream_settings` 只记录上游返回且可读取的标量字段，不冒充内核内的全部最终设置。

上游 TE-Speed-VOSR2 1.0 README 提醒：`speed` 更大的 tile batch 或 VAE tile 可能提高显存峰值并触发换载，因此可能比 `manual` 慢。本次保持 tile 尺寸，仅对已验证的 Linux 兼容层合并独立分块调用；实测范围见下文。

`experimental_32gb` 在释放 ComfyUI 已缓存的 H3 模型之后，读取实际运行 CUDA 设备的总量和空闲显存。只有总量至少 **30 GiB** 且空闲至少 **24 GiB**，两者均为有效测量时才允许运行。门槛只表示允许实验，不保证任意分辨率/片长都不会 OOM。测量不可用、非有限值或不足会明确报错，不会偷偷回落或改选其他档。

## 在导演台或独立节点选择

选择 VOSR2 后，设置 `vosr2_profile`。旧工作流没有此字段时使用 `legacy`。独立 `MiniMaxH3VideoFinish` 在旧四个必填输入后追加可选配置，原有 widget 顺序保持不变。高级 `benchmark_run` 日常留空；对比工具用它写入唯一标识，让每次超分实际重新执行，并通过 ComfyUI history 返回诊断数据。

VOSR2 按实际源帧计算所需倍率，目标分辨率是下限。导演台 1344×768 到至少 1920×1080，按源比例和偶数边界输出 1920×1098；独立节点按短边 1080 输出 1890×1080。原始输入不会先缩小，已经超过目标的源帧保留尺寸。整数倍率直接调用后端；小数倍率用 float32 准备目标画布后执行后端 1 倍重建，模型实际处理必要的尺寸。不能将后端 scale=1 误读为没有执行 AI 超分：记录中的 `scale` 是相对源图的有效倍率，`backend_scale` 是后端整数参数，`processing_input_size` 是后端实际输入。

普通页面自动处理倍率；配置比较入口位于默认收起的「超分高级设置」，保存的旧配置继续生效。后端返回丢帧、非有限像素、错误 IMAGE 格式或低于目标的尺寸时会停止，不会用插值掩盖不足。音频不进入超分；对比工作流使用同一份输入音频和固定帧率重新封装。减少处理像素不等于已测得同比例提速；相同源片的耗时和细节/闪烁对照仍需单独记录。

## 运行可重复对比

在实际测试机器启动更新后的 ComfyUI，确保 `TESpeedVOSR2Loader`、`TESpeedVOSR2Settings`、`TESpeedVOSR2Video` 与 VideoHelperSuite 可用。工具只通过标准 `/prompt`、`/history`、`/object_info`、`/system_stats` API 提交和读取，不会在独立 Python 进程导入 ComfyUI 或上游自定义节点。

先准备一个代表性的源视频。`--video` 必须是 **ComfyUI 服务器上的路径**；本地工具不会上传它。每次对比加载同一文件开头，固定 seed、帧率、分辨率和放大规则；最多 5 秒。`--fps` 应填写准备比较的帧率，默认 24；VHS 会按此值读取/重采样，不能把它理解为自动保留未知的源帧率。

```bash
python tools/benchmark_vosr2.py \
  --url http://127.0.0.1:8188 \
  --video /root/autodl-tmp/ComfyUI/input/vosr-source.mp4 \
  --profiles legacy manual_baseline \
  --duration 5 --fps 24 --seed 42 --resolution 1080p \
  --repeat 3 --output-dir benchmark-results
```

实验档必须显式加入：

```bash
python tools/benchmark_vosr2.py --video /root/autodl-tmp/ComfyUI/input/vosr-source.mp4 --profiles legacy manual_baseline experimental_32gb --duration 5 --fps 24 --seed 42 --repeat 3
```

Windows PowerShell 可写成一行，把 `python` 换成 ComfyUI 的嵌入式 Python 路径，把输入路径换为服务器本机路径。例如：

```powershell
& 'D:\ComfyUI_windows_portable-G313\python_embeded\python.exe' tools/benchmark_vosr2.py --video 'D:\clips\vosr-source.mp4' --profiles legacy manual_baseline --duration 5 --fps 24 --seed 42 --repeat 3
```

`--dry-run` 只生成可审阅的 API 工作流，不连接服务器、不计时。`--repeat 3` 会轮换各档执行顺序，降低首档冷加载对比较的偏差；每次都保留记录，不剔除慢结果。请让测试期间的 GPU 负载和其他队列任务保持稳定。工具不会清空他人的队列或自动重启服务。超时会停止继续提交，保留 prompt ID；服务器原任务可能仍在执行，需要先查看队列。

每个本地结果目录包含：

- `workflow.json`：实际提交的 API 工作流，包含同一输入、seed、帧率与唯一运行标识。
- `submission.json`：ComfyUI 返回的 prompt ID。
- `history.json`：完整服务器执行记录，含失败信息。
- `summary.json`：配置、计时、内存测量、输入/输出尺寸、帧数、服务器输出文件名/子目录及 `/view` 链接。

顶层 `summary.json` 汇总所有运行和服务器 `/system_stats`。生成的视频保存在 **服务器** 的 `ComfyUI/output/vosr2-comparison/` 下，默认 H.264 CRF 18；JSON 保存在运行工具的机器上。路径由服务端返回，不假装是客户端已下载的视频。

## 如何读测量

`vosr2.elapsed_seconds` 是释放模型、加载、Settings、VOSR2 重建及后端收尾释放的合并墙钟时间，CUDA 在边界同步；不包括最后 CPU resize 和视频编码。`finish_elapsed_seconds` 再包含独立节点的最终尺寸处理。`api_wall_seconds` 还包含提交、队列等待、解码、编码和 history 轮询，不能当作纯超分内核时间。

`memory_before/after` 的 `free_vram_gb`、`total_vram_gb` 是 GPU 全设备视角；`allocated_gb`、`reserved_gb` 是当前 Python 进程的 PyTorch allocator 视角。`process_peak_*` 是进程存活期/上次外部重置后的峰值，本工具不重置共享计数器，**不能用它直接宣称某档的独立峰值**。没有测到的内存会标明不可用，不写成零显存占用。失败报告使用 `failed`；API 成功却没有本次超分证据时标为 `incomplete`，不会当成一次成功测速。

先比较同一时刻的人脸、眼睛、口型、头发、文字、细线、运动边缘、首尾帧与相邻帧闪烁，再结合重复运行计时和换载情况选择。保持放大后尺寸、帧数与音频时长一致。配置测试只能证明参数和保护逻辑；下面的短样本测量不能代替完整长片的质量验收。工具不输出“最快”或自动保存胜出配置。

## 2026-10-20 AutoDL RTX 5090 实测

在队列空闲时，用独立进程比较兼容层原实现与补丁。输入来自同一段 H3 原生输出的第12、48、84帧，准备相同1920×1098画布，固定 seed=42、模型、512分块和重叠。排除预热，每个变体跑两次，顺序正反各一次。

| 变体 | 两次三帧耗时 | 平均耗时 |
|---|---|---:|
| 原实现 | 4.5863 / 4.5771 s | 4.5817 s |
| 仅保留帧间 CUDA 分配缓存 | 4.5471 / 4.5594 s | 4.5533 s |
| 缓存复用 + 四分块批量 | 3.8242 / 3.7673 s | 3.7958 s |

四分块方案在此样本少用17.2%的重建时间（约1.21倍），仅去掉逐帧清缓存收益约0.6%。独立进程峰值 allocated 均为5.704GiB，reserved 为7.709–7.734GiB；这不代表完整 ComfyUI 工作流的显存峰值或15秒视频端到端耗时。

仅缓存复用与基线逐像素一致；四分块因批量浮点计算有差异，0–1像素平均绝对误差0.00159、最大局部误差0.25781、PSNR49.67dB。三组对应画面已目视检查，未见明显新增模糊或伪影，但不宣称逐像素一致、清晰度提升或已完成长片时序质量验收。原引擎 SHA256 为 `58f94395258d53fd403d55d215cd4074e6967d438300ce8244161f2767a1cdb0`。
