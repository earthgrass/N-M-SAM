# APBR-L2 Seed 0：条件贪心 OBS 同父对照

## 结果

这是独立的 CPU 探索实验。APBR-L2 Seed 0 Epoch-200 Dense Final 为唯一新实验父模型；四路均使用同一 1,024 张 CIFAR-100 Train 校准图片、19 个 eligible Conv2d、相同阻尼和物理联合 OBS 补偿。近优几何裁决的窗口固定为 1%。没有重训、Recovery 或 BN-only。四路 checkpoint 和 Mask 在打开 Test 之前冻结并计算 SHA256，随后每路仅评测一次固定的 10,000 张 Test。

| Seed 0 路径 | Dense Final | 剪枝补偿后 | 相对 Dense | Test CE |
|---|---:|---:|---:|---:|
| SAM + 条件贪心 OBS（历史对照） | 78.84% | 77.84% | −1.00 pp | 0.828370 |
| APBR-VPR + 同一条件贪心 OBS（历史对照） | 78.68% | 77.72% | −0.96 pp | 0.829459 |
| APBR-VPR + 几何近优条件贪心 OBS（历史对照） | 78.68% | 77.83% | −0.85 pp | 0.829065 |
| **APBR-L2 + 同一条件贪心 OBS（本次）** | **78.91%** | **77.75%** | **−1.16 pp** | **0.835942** |
| **APBR-L2 + 几何近优条件贪心 OBS（本次）** | **78.91%** | **77.60%** | **−1.31 pp** | **0.835403** |

同父对照中，APBR-L2 的几何近优裁决比纯物理条件贪心低 **0.15 pp**，尽管 CE 低 0.000539。相对 SAM 历史条件贪心，APBR-L2 的纯物理和真实几何路径分别低 0.09 和 0.24 pp。APBR-L2 的 Dense Final 比 SAM 高 0.07 pp；这个优势没有延续到上述剪枝补偿终点。这是一个 Seed 的探索性比较，不能归因于训练算法或几何状态的稳定因果效应。

| APBR-L2 同父选择消融 | Test Accuracy | Test CE | 相对纯物理的 Mask 翻转位置 |
|---|---:|---:|---:|
| 纯物理条件贪心 | 77.75% | 0.835942 | 0 |
| 单位几何近优裁决 | 77.59% | 0.835809 | 867,822 |
| 真实 APBR-L2 几何近优裁决 | 77.60% | 0.835403 | 867,630 |
| 组内打乱 APBR-L2 几何 | 77.65% | 0.835870 | 867,462 |

真实几何只比单位几何高 0.01 pp，却比打乱几何低 0.05 pp；这批数据没有显示学到的坐标对应关系带来可靠优势。真实几何与单位几何之间翻转 18,462 个位置，涉及 8,784 个四元组。

## 身份与数值审计

- APBR-L2 Dense 父 checkpoint SHA256：`d68fbffc5f862be07656c5b45b8dcde2a5411865c0e5e03f2f45c2f97926b742`；Epoch 200、78,200 steps、`test_accessed=false`；匹配的 FP64 `fixed_l2` 几何状态含 62 项，参数形状映射逐项一致。
- SAM 和 APBR-L2 Seed 0 的初始模型 SHA256 相同：`87aa9dd7492a89adf9e3ab78ca4a3227f30bc8de53b18a3bd73cd61ab0bd759d`。四路使用同一 APBR-L2 Dense 父模型；SAM/VPR 行仅作历史跨父对照。
- 每路 19 层、11,157,504 个 eligible 权重、5,578,752 个删除位；bool keep-mask、严格 2:4、删除位精确为零、全部权重有限；BN 和排除层与父模型逐项相同。独立审计状态为 `INDEPENDENT_ARTIFACT_AUDIT_PASS`。
- CPU 数值回归：`19 passed`。本次执行使用 CPU PyTorch；没有占用或改动远端正在运行的 APBR-L2/SAM 训练队列。没有根据 Test 修改窗口、算法或补偿设置。

证据：[结果 JSON](../../../artifacts/apbr_l2_nearoptimal_conditional_seed0_full_cpu_20261103_v1/results.json)、[Test 前冻结清单](../../../artifacts/apbr_l2_nearoptimal_conditional_seed0_full_cpu_20261103_v1/pre_test_freeze.json)、[独立审计](../../../artifacts/apbr_l2_nearoptimal_conditional_seed0_full_cpu_20261103_v1/independent_audit.json)、四份独立补偿 checkpoint 及逐层日志位于同一产物目录。历史 SAM/VPR 比较见 `docs/research/apbr_nearoptimal_conditional_seed0_20261103_v2/FULL_MODEL_REPORT.md`。
