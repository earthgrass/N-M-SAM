# APBR near-optimal conditional OBS: Seed 0 train-only pilot

本次是 APBR-VPR Seed 0 同父模型的真实 CIFAR-100 单层 CPU 诊断；仅使用官方 Train，未读取 Test。
层：`layer1.0.conv1`，K=576；冻结的近优窗口 τ=1.00%。
四臂共用物理 Hessian 和联合 OBS 补偿，只改变近优窗口内的选择。
`physical` 直接取最低物理 OBS 分数；`unit` 用 |w|² 裁决；`true` 用 |w exp(-s)|²；`shuffled` 在每个四元组内固定打乱 s。
第二次选择使用虚拟删除后的权重与条件逆曲率。各步近优界限不保证整对总代价或最终准确率的相同界限。

| 选择 | 训练侧诊断准确率 | CE | 物理二次误差 | 相对 physical Mask 翻转位置 | 两步近优候选>1 |
|---|---:|---:|---:|---:|---:|
| physical | 100.00% | 0.0024 | 0.00200778 | 0 | 4,318/3,703 |
| unit | 100.00% | 0.0024 | 0 | 0 | 9,216/3,648 |
| true | 100.00% | 0.0024 | 0 | 0 | 9,216/3,648 |
| shuffled | 100.00% | 0.0024 | 0 | 0 | 9,216/3,648 |

稠密模型在同一训练诊断子集：100.00%，CE=0.0024。
本机无 CUDA；没有全模型准确率或 GPU 资源结论。APBR-VPR 的结果也不能代替 APBR-L2 结论。
真实几何优于打乱或单位几何的判断须同时看训练侧分类指标、物理重建误差和多层/多 Seed 稳定性。
附加一条输出行的锁定官方 TrueOBS 参照；它跨组全行选择，和本 Pilot 的固定四元组处理顺序不同，不能直接归因方法优劣。
证据：`C:\Users\earthgrass73\Desktop\NM稀疏\APBR-post-training-benchmark\vpr_sam_research\experiments\post_training_nm_benchmark\artifacts\apbr_nearoptimal_conditional_seed0_train_pilot_20261103_v1`。
