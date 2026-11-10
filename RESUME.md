# 后续恢复入口

1. 从本分支 README 与 inventory 确认方法版本、约束、真实完成/失败/取消状态。归档不会自动启动任何任务。
2. 校验 `SHA256SUMS.txt`。按 `SOURCE_FILE_MAP.json` 查找原路径；不能将不同父模型、Mask、补偿器或论文版本互换。
3. 从本地保存位置取回所需 checkpoint/Mask/Geometry，对照 `EXTERNAL_ASSETS_SHA256.json` 和原 run manifest 的父子哈希；数据集按原官方 CIFAR-100 审计准备。没有文件时先标记缺失，不重训替代。
4. 源码保留原包布局。实验工程在 `project/vpr_sam_research/experiments/post_training_nm_benchmark`，依赖的 VPR 包在 `project/vpr_sam_research/src`；HQ 独立工程在 `project/HQ-APBR-V1`。以本分支实际存在目录为准。
5. 从实际运行的 freeze/config/environment/source identity 文件复原环境与命令。顶层计划配置不一定是最终运行配置；不要直接执行历史队列启动器。历史绝对路径需要在新副本中显式映射，原配置保持不变。
6. 先确认资源与原门禁、再做所需针对性验证。已完成的 Test 不重复执行；失败不自动重试；取消的队列不恢复。本次用户仅授权存档，新的训练或评测需新的明确指令。

此归档进行了文件复制/哈希/身份检查，没有重新运行科研测试。历史测试日志仅代表其记录的代码与环境。
