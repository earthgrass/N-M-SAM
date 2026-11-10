# APBR-L2 Seed 3：条件贪心 OBS 独立结果

使用官方 CIFAR-100、APBR-L2 Seed 3 Epoch-200 Dense Final、固定的 1,024 张 Train 校准样本、19 层 2:4 以及冻结的 1% 近优窗口。四路均从同一个稠密权重出发，使用相同的物理联合 OBS 补偿；没有 Recovery 或 BN-only。全部补偿 checkpoint 和 Mask 先冻结 SHA256，再各做一次 10,000 张 Test。

| Seed 3 路径 | Test Accuracy | Test CE | 相对 Dense |
|---|---:|---:|---:|
| APBR-L2 Dense Final | 78.64% | 0.809448 | — |
| 纯物理条件贪心 OBS | **77.47%** | 0.832367 | −1.17 pp |
| 单位几何近优裁决 + OBS | 77.39% | 0.834014 | −1.25 pp |
| 真实 APBR-L2 几何近优裁决 + OBS | 77.38% | 0.834096 | −1.26 pp |
| 组内打乱几何近优裁决 + OBS | 77.36% | 0.834124 | −1.28 pp |

真实几何比纯物理低 0.09 pp，比打乱几何高 0.02 pp。它相对单位几何翻转 13,894 个权重位置，涉及 6,655 个四元组。此 Seed 没有显示几何裁决改善物理条件贪心。

父 checkpoint SHA256：`922a5adf12aba494086cff3f0b00d3cfed6d364e5f33ecea667bf7f6fe3284ad`。本地回传哈希与远端队列一致；Epoch 200、78,200 steps、训练阶段 `test_accessed=false`。独立产物审计通过：每路 19 层 bool keep-mask、11,157,504 eligible 坐标、5,578,752 删除位置、严格 2:4、删除权重精确为零、数值有限；BN 和排除层与父模型逐项相同。

证据：[结果 JSON](../../../artifacts/apbr_l2_nearoptimal_conditional_seed3_full_cpu_20261104_v1/results.json)、[Test 前冻结清单](../../../artifacts/apbr_l2_nearoptimal_conditional_seed3_full_cpu_20261104_v1/pre_test_freeze.json)、[独立审计](../../../artifacts/apbr_l2_nearoptimal_conditional_seed3_full_cpu_20261104_v1/independent_audit.json)。对应 SAM Seed 3 Dense Final 尚未完成，因此此报告没有填入跨父对照。
