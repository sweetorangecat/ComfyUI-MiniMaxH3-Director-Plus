# 短视频渐进采样实验设计

用户已授权由实现者决定采用方案。选择研究报告的方案B，限定短视频；本次不实现长视频接续、动作迁移或48FPS。

## 行为

导演台新增高级配置 `sampling_method=standard|selflift`（默认standard）和 `community_lora_profile=off|continuity|combat|legacy`（默认off）。仅质量优先二采样使用这些配置；SelfLift首先限REF2VA、无音色条件、4–5秒、最多1080p、24FPS。低显存菜单及现有计算路线保留。旧导演指南缺少新字段时保留旧LoRA开关语义。

SelfLift目标H3网格按原生画布限制在1344×768面积以内，最终输出由所选超分完成；不可把首采的小网格再次当作SelfLift目标。前4步在宽高约0.65倍网格，后4步在目标网格，simple8/Euler、rho=0、外部learned upscaler、用完卸载。初始实验不使用空间分块、TST或额外refiner。显存前置要求总量至少20GB且空闲至少18GB，仅为保守启动门槛，不承诺无OOM。依赖、VAE、Euler或AV契约不满足时终止，不旁路成另一采样方法。

SelfLift通过独立适配模块调用已安装节点，不复制上游算法。正常二采模型路由可复用匹配的v4 LoRA，SelfLift两阶段使用同一个模型；保留现有精确注意力补丁，实验关闭Sage。VAE作为采样节点新增可选输入，官方示例两条采样分支均连接；旧图使用standard无需补线。

动作连续性使用Motion_Repair 0.25；武戏使用Motion_Repair 0.25+H3_Combat_V2 0.5。不自动添加GunFu/fafafast或电影风格。legacy才复用旧社区链与环境开关。记录实际加载项目，并明确配置只作用于二采。

最终超分显示VOSR2、SeedVR2、通用AI三个选项，二采也允许选择。显式选择时必须执行该引擎；SeedVR2缺失时报错，不改为AI。二采的过采样/下缩路线不得覆盖显式超分；源尺寸等于或大于目标时也先执行所选SR再适配尺寸。既有native/Lanczos/RTX API兼容保留。

## 验证

纯Python测试覆盖模式与参数拒绝、横竖网格、真实8步合同、依赖缺失、外部节点参数与结果验证、LoRA选择和路由。CPU集成测试覆盖导演指南、调度器、采样分发；JS测试覆盖可见控件、切换与保存恢复；示例JSON检查VAE连线。正式部署前仍需AutoDL同模型同种子4–5秒A/B，以及实际8GB低显存对照；本地CPU测试不证明画质或峰值显存。
