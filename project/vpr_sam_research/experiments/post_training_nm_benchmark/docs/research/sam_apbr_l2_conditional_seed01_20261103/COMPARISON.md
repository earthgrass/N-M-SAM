# SAM / APBR / APBR-L2：条件贪心 OBS Seed 0–1

同一 CIFAR-100 训练校准 ID、19 层严格 2:4、相同物理联合 OBS；均无 Recovery。以下是两个 Seed 的探索性结果，不是正式多 Seed 结论。

| 路径 | Seed 0 | Seed 1 | 均值 ± 样本标准差 |
|---|---:|---:|---:|
| SAM + 条件贪心 OBS | 77.84% | 77.77% | 77.81 ± 0.05% |
| 原版 APBR + 几何近优 OBS | 77.83% | 77.43% | 77.63 ± 0.28% |
| APBR-L2 + 条件贪心 OBS | 77.75% | 77.55% | 77.65 ± 0.14% |
| APBR-L2 + 几何近优 OBS | 77.60% | 77.44% | 77.52 ± 0.11% |

| 同 Seed 配对差值 | Seed 0 | Seed 1 | 均值 |
|---|---:|---:|---:|
| vpr_geometric_minus_sam | -0.01 pp | -0.34 pp | -0.18 pp |
| l2_geometric_minus_sam | -0.24 pp | -0.33 pp | -0.28 pp |
| l2_physical_minus_sam | -0.09 pp | -0.22 pp | -0.16 pp |
| l2_geometric_minus_l2_physical | -0.15 pp | -0.11 pp | -0.13 pp |
| l2_geometric_minus_vpr_geometric | -0.23 pp | +0.01 pp | -0.11 pp |

两 Seed 样本很小；跨 SAM/VPR/L2 比较同时改变了稠密训练和部分 Mask 选择，不能把配对差值归因于单一机制。L2 的物理与几何路径为同父对照。Test 结果仅从既有一次性评估记录读取，未重新评估或选参。

机器可读结果及各检查点 SHA256 见 `comparison.json`。
