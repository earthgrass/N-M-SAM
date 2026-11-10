# APBR-L2 Seed 2：条件贪心 OBS 独立结果

使用官方 CIFAR-100、APBR-L2 Seed 2 Epoch-200 Dense Final、固定的 1,024 张 Train 校准样本、19 层 2:4 以及冻结的 1% 近优窗口。四路均从同一个稠密权重出发，使用相同的物理联合 OBS 补偿；没有 Recovery 或 BN-only。全部补偿 checkpoint 和 Mask 先冻结 SHA256，再各做一次 10,000 张 Test。

| Seed 2 路径 | Test Accuracy | Test CE | 相对 Dense |
|---|---:|---:|---:|
| APBR-L2 Dense Final | 78.80% | 0.800291 | — |
| 纯物理条件贪心 OBS | 77.28% | 0.838268 | −1.52 pp |
| 单位几何近优裁决 + OBS | 77.32% | 0.838872 | −1.48 pp |
| 真实 APBR-L2 几何近优裁决 + OBS | **77.36%** | 0.838709 | −1.44 pp |
| 组内打乱几何近优裁决 + OBS | 77.34% | 0.838883 | −1.46 pp |

真实几何比纯物理高 0.08 pp，比打乱几何高 0.02 pp。它相对单位几何翻转 15,018 个权重位置，涉及 7,213 个四元组。这是单 Seed 探索结果，不能证明学习到的几何对应关系稳定有效。

父 checkpoint SHA256：`755af8c38f89babd83b967e757198383f69c9c756aca3a5c2da0ed9c6226b44e`。本地回传哈希与远端队列一致；Epoch 200、78,200 steps、训练阶段 `test_accessed=false`。独立产物审计通过：每路 19 层 bool keep-mask、11,157,504 eligible 坐标、5,578,752 删除位置、严格 2:4、删除权重精确为零、数值有限；BN 和排除层与父模型逐项相同。

证据：[结果 JSON](../../../artifacts/apbr_l2_nearoptimal_conditional_seed2_full_cpu_20261104_v1/results.json)、[Test 前冻结清单](../../../artifacts/apbr_l2_nearoptimal_conditional_seed2_full_cpu_20261104_v1/pre_test_freeze.json)、[独立审计](../../../artifacts/apbr_l2_nearoptimal_conditional_seed2_full_cpu_20261104_v1/independent_audit.json)。随后找回并验证了 SAM Seed 2 的完整历史 Final，跨父对照见[独立比较](../sam_apbr_l2_conditional_seed2_20261104/COMPARISON.md)。
