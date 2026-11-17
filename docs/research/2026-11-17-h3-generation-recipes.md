# H3 生成端候选：执行配方复核

后续实际推理、画面验收和最终采用结论见[生成与交付实测](2026-11-17-h3-generation-trials.md)。本文保留作者配方与权重核查结果。

日期：2026-11-17。只研究一手配置和权重元数据；未用SSH、未运行GPU、未改生产代码或默认。承接恢复端实测未解决肤质的问题。本文件给主执行者提供可复现输入，不代表任何候选已在本机画质胜出。

## 执行顺序

先测 **SMD REF2VA专训8步完整compact配方**，再测 **Mystic V5 REF作者单采12步配方**。LightX REF8 v1.1保留，但其当前公开官方示例没有给出完整对应的REF8图，暂不填猜测调度。每条先检查无超分的原生768p肤质、人物/道具/动作，再测原生参考音频；超分不能遮掩生成端失败。

这个排序依据可复现程度和可控成本，不是认为SMD已经最真实。SMD有同作者的原生图、精确模型清单、单独训练的8步权重；Mystic贴近质感目标，但叠旧FL蒸馏器且实际12步，干扰项更多。

## SMD REF2VA 8步

官方转换仓库revision：`9b119223724535142a758eaed65d9e0fc50b37ee`。

| 项目 | 精确值 |
| --- | --- |
| LoRA文件 | `SMD-MiniMax-H3-Ref2VA-8step.safetensors` |
| 大小 | 1,956,183,672 bytes |
| SHA256 | `89cc678b60b2e85f10cdf69739900d948475fa52512b165b635b8b65aa364a8b` |
| 模型顺序 | 纯REF2VA底模 → 此LoRA强度1 → `MiniMaxH3SigmaShift` 12/3 |
| 引导/采样 | `BasicGuider`，无CFG；`euler`；一次完整8步，不接SelfLift |
| sigmas | 作者`ManualSigmas`，见下方；不要再次shift这串视频sigma |
| 原作者图参考预处理 | 每图`ResizeImageMaskNode`: shorter dimension512、bicubic；随后原生REF节点`ref_image_size=match` |
| 原作者样片 | 1344×768，124帧，24fps；seed42 |

```text
1.0000000000, 0.9873417722, 0.9705882353,
0.9473684211, 0.9130434783, 0.8571428571,
0.7500000000, 0.4615384615, 0.0000000000
```

最后原始时间区间按0.5缩放，再分别做视频/音频时移。应直接复用作者原生图的sigma和AV打包方式，不能用普通8步simple代替。不要叠H3 Lightning、LightX Turbo、DMAD、Mystic、MotionRepair或现SelfLift二采；那会变成另一条未经训练配方验证的路线。[原生API图](https://huggingface.co/zjwfufu/SMD-MiniMax-H3-Ref2VA-8step-ComfyUI/blob/9b119223724535142a758eaed65d9e0fc50b37ee/workflow_api.json)、[sampling.json](https://huggingface.co/zjwfufu/SMD-MiniMax-H3-Ref2VA-8step-ComfyUI/blob/9b119223724535142a758eaed65d9e0fc50b37ee/sampling.json)。

本次有界HTTP Range实际读到8步转换权重的64,304字节safetensors header：624项为416个BF16 A/B矩阵和208个F32 alpha，没有AdaLN或time_embed分支；metadata记录原始源文件SHA256 `f8aabc9258d443a5c04833a637246ecb3551cb53847e5aba1ef4a00b444e462c`以及QKV/SwiGLU转换。它不涉及IC测试曾遇到的8维曲线分支，但仍须在实际Comfy loader中确认208目标均匹配，不能以文件头检查代替推理验收。

### Compact精确组件

以下模型均来自`Comfy-Org/MiniMax-H3` revision `e5eb578a89295337b8ff433a035929ce0279e0b6`。

| 组件 | bytes | SHA256 |
| --- | ---: | --- |
| `diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors` | 20,970,379,616 | `9255f52b6677845ad238f20dfaafa94727053694127ab7f255c048f0f9365779` |
| `text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | 15,687,142,551 | `35a88d51044231fe332301d7a62aa81e3f2cba62febeb446e2c1e3e0ef76f2c6` |
| `vae/minimax_h3_video_vae_int8_convrot.safetensors` | 2,811,065,184 | `52a2c8c73583c86e4f41cdcce3a6ad0ea562987bc0bf3d60a0cef5f5c8e60c0e` |
| `vae/minimax_h3_audio_vae_fp32.safetensors` | 605,254,808 | `8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48` |

这组量化精度是作者已测的完整compact profile，**并非SMD LoRA数学上必须使用的精度**。作者还提供同一LoRA的正式BF16 profile：完整REF BF16底模、INT8文本编码器、FP16视频VAE和同一FP32音频VAE。故可以设计FP16 VAE等适配变体，但需如实标记，不能称原compact完全复现。32GB首测优先完整compact，其他组合另做单变量比较。[compact模型清单](https://huggingface.co/zjwfufu/SMD-MiniMax-H3-Ref2VA-8step-ComfyUI/blob/9b119223724535142a758eaed65d9e0fc50b37ee/models.json)、[BF16清单](https://huggingface.co/zjwfufu/SMD-MiniMax-H3-Ref2VA-8step-ComfyUI/blob/9b119223724535142a758eaed65d9e0fc50b37ee/models_bf16.json)。

不要只按文件名认模型。主执行者发现远端同名INT8视频VAE为3,171,670,912 bytes，与此清单不同；必须匹配hash或明确为不同版本后再比较。

### 音色与性能边界

原生8步作者图未挂参考音频。其standalone runner确有image/video/audio参考接口，支持最多3条音频，但这不证明短参考克隆新对白的效果已通过。原生图接入本项目的参考音频属于额外实测；必须把实际音频条件挂入H3，不能最终换音轨冒充克隆。无需增加用户上传数量。

作者compact测试为A100 80GB限制显存预算、124帧，音频只做解码/电平检查，未试听。不能将其167.447秒或约24.3GiB采样峰值标为物理32GB、15秒或音色通过。[作者验证记录](https://huggingface.co/zjwfufu/SMD-MiniMax-H3-Ref2VA-8step-ComfyUI/blob/9b119223724535142a758eaed65d9e0fc50b37ee/validation.json)、[原始参考输入代码](https://huggingface.co/zjwfufu/SMD-MiniMax-H3-Ref2VA-4step/blob/79cee597c038a3d2f19b5cfe8667a068cca82b25/tools/infer_minimax_h3.py)。

## Mystic V5 REF：必须纠正为12步

作者版本ID`3416478`，REF权重fileID`3311427`，文件`MysticXXX_MMH3-V5-ref2va.safetensors`，138,339,432 bytes（API sizeKB×1024），SHA256：`F57C76E761F7A5113E541AD58E84C648A5FE4D3CFBC277D0C44AE9763DA29AFE`。作者附图fileID`3311449`，原始JSON SHA256：`77B4B136C065DDFE49EA5A3E7F83687D66816FBBD80C1B425121E2A00C373783`。[版本API](https://civitai.com/api/v1/model-versions/3416478)、[作者图下载](https://civitai.com/api/download/models/3416478?fileId=3311449)。

本次追踪实际连线，发现此前研究只看`BasicScheduler.widgets_values=[simple,8,1]`得出的“作者8步”有误：外层`PrimitiveInt#86`实际值为**12**，经link212接入subgraph#94的steps；内部link167再接到`BasicScheduler#48.steps`，覆盖内部未连接时显示的8。因此作者实际路线为：

```text
纯REF pruned INT8 ConvRot
→ ModelAttentionBackend(comfy kitchen attention)
→ 旧FL4 resized rank21 LoRA，0.75
→ Mystic V5 REF，1.0
→ BasicGuider + res_multistep + BasicScheduler(simple, 12, denoise1)
→ 单次采样、视频/音频VAE解码
```

作者图没有显式SigmaShift节点；图的Comfy属性版本为0.33/0.34，Comfy v0.34.0的MiniMaxH3默认sampling settings为视频12、音频3。复现时记录实际底模默认值，不能误写成作者图里有显式12/3节点。[Comfy v0.34.0默认值](https://github.com/Comfy-Org/ComfyUI/blob/v0.34.0/comfy/supported_models.py)。

参考图先`ImageScaleToMaxDimension(area,512)`，这里是**最大边512**，与SMD的**短边512/bicubic**不同；再用`ref_image_size=match`。作者图用NVFP4-AWQ TE、INT8视频VAE、FP32音频VAE；没有接参考音频。V5 REF权重和V5 FL权重不可互换，当前SelfLift策略会回到旧U22质感链，所以必须绕开该自动替换来测试作者整套单采。

旧FL4文件精确名为`minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy_resized_avg_rank_21_bf16.safetensors`，314,878,200 bytes，SHA256 `3a069f26fbc33f377a60dc72dd9e15f2aa42aa1d1b44915fded835716672dd36`。这是作者实际组合的旧FL适配器，不能擅自换成新REF4或新REF8后仍称作者配方。[Kijai原转换文件](https://huggingface.co/Kijai/MiniMax-H3_comfy/blob/37ae5cbe1d6f2243484812fc511f9fa427b12a30/loras/minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy_resized_avg_rank_21_bf16.safetensors)。

作者宣称肤质/光照/运动改善，不是本项目真人中景、手和道具、参考音色均已通过的证据。作者图内的示例文本只作数据解析，不沿用到本项目；测试继续使用用户已确认的原资产和正常对白场景。

## LightX2V REF8 v1.1：权重明确，完整对应图仍有缺口

HF revision `c2bf13b837e33bd851b370285fc7db99482d2ab2`：

| 格式 | 文件 | bytes | SHA256 |
| --- | --- | ---: | --- |
| ComfyUI | `minimax_h3_ref2v_turbo_8step_v1.1_768p_comfyui_bf16.safetensors` | 1,956,193,000 | `e56863a7524e0300729cbb7b64cc8b548d90698c1abbf7d83898b0122fbe1e81` |
| 原生 | `minimax_h3_ref2v_turbo_8step_v1.1_768p_bf16.safetensors` | 1,383,677,808 | `46c64fc94509489f18dd1f545172e30c8df7d4cc50a30e96744409b4288c937d` |

作者发布帖明确新的FL版shift6建议不适用于REF；帖尾REF8是否12/3的提问没有作者答复。当前ModelTC/Minimax-H3-Turbo示例REF图仍默认旧REF4 v0.1；LightX2V目录的8步配置三份均为FL2V v1.2，不是REF8 v1.1配置。不能用FL新版本配置或旧REF4 JSON填补空缺后声称完整复现。需要从对应训练/权重元数据或作者明确配置进一步锁定才执行；现在先测配方完整的SMD。

实际有界Range读取73,632字节header后，转换metadata没有shift字段，而且`base_model`写的是`Comfy-Org/MiniMax-H3 minimax_h3_fl2va_bf16.safetensors`。这可能是转换器复用的模板描述，**不能据此直接宣布REF权重错误**；也不能拿这段metadata补出REF8推理参数。训练rank128、alpha8、scale0.0625有记录，保持作者转换的逐层alpha，不要自行全局改成rank128的alpha或把strength2当稳定修复。

[作者发布帖](https://huggingface.co/lightx2v/Minimax-h3-Turbo/discussions/57)、[官方Comfy说明](https://github.com/ModelTC/Minimax-H3-Turbo/blob/main/COMFYUI_SETUP_AND_INFERENCE.md)、[LightX2V官方推理说明](https://github.com/ModelTC/LightX2V/blob/main/scripts/minimax_h3/README.md)、[精确REF8文件](https://huggingface.co/lightx2v/Minimax-h3-Turbo/blob/c2bf13b837e33bd851b370285fc7db99482d2ab2/minimax_h3_ref2v_turbo_8step_v1.1_768p_comfyui_bf16.safetensors)。发布帖的新版REF8语音不连贯反馈是用户自报，不能当所有配置必失败的结论。

## 原始证据与验收口径

本次新取的官方小文件、API（含LFS hash/size）位于`outputs/diagnostics/quality_research_20261117/generation_recipe_check/`；Mystic原始图保留于上一级。没有下载权重主体。最终应保留同素材、同seed/提示词、实际条件图/音频尺寸、采样sigma、已加载LoRA/精度/注意力、原生无损帧、原音轨与分项耗时。新LoRA输出更锐或更光滑都不能独自代表写实质量胜出；本轮目标仍是皮肤发丝更自然且人物、动作、颜色、语音不退化。
