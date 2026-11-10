# Seed 2：SAM 与 APBR-L2 条件贪心 OBS

两路均使用 CIFAR-100、200 Epoch Dense Final、固定的 1,024 张训练图片作校准、19 层严格 2:4，以及相同的物理条件贪心与联合 OBS。SAM Seed 2 使用已验证的历史 `last.pt`：其 SHA256 与当时发布记录中的 Final SHA256 完全相同；另一个本地 `final_epoch_200.pt` 副本不完整，未用于实验。两路初始模型哈希一致。剪枝期间未访问 Test；checkpoint 和 Mask 冻结后各作一次固定 Test。无 Recovery。

| 路径 | Dense Final | 条件贪心 OBS | 相对 Dense | Test CE（剪枝后） |
|---|---:|---:|---:|---:|
| SAM | 78.80% | **77.43%** | −1.37 pp | 0.839221 |
| APBR-L2，纯物理选择 | 78.80% | 77.28% | −1.52 pp | 0.838268 |
| APBR-L2，真实几何近优裁决 | 78.80% | 77.36% | −1.44 pp | 0.838709 |
| APBR-L2，组内打乱几何 | 78.80% | 77.34% | −1.46 pp | 0.838883 |

在这个 Seed，APBR-L2 真实几何比其纯物理路径高 0.08 pp、比打乱几何高 0.02 pp；但比 SAM 的相同物理选择低 0.07 pp。单个 Seed 的 0.02 pp 差距不足以证明几何坐标对应关系有稳定收益。

SAM 父 checkpoint SHA256：`558ffa8da0a886a1291746fb6b5d979a2e83938b1150136568edea5e116fb977`。APBR-L2 父 checkpoint SHA256：`755af8c38f89babd83b967e757198383f69c9c756aca3a5c2da0ed9c6226b44e`。独立审计均通过：19 层、11,157,504 个 eligible 坐标、5,578,752 个删除坐标、严格 2:4、删除权重为零、数值有限；BN 与排除层保持父模型状态。

证据：[SAM 结果](../../../artifacts/sam_conditional_seed2_cpu_20261104_v1/results.json)、[SAM 独立审计](../../../artifacts/sam_conditional_seed2_cpu_20261104_v1/independent_audit.json)、[APBR-L2 结果](../../../artifacts/apbr_l2_nearoptimal_conditional_seed2_full_cpu_20261104_v1/results.json)、[APBR-L2 独立审计](../../../artifacts/apbr_l2_nearoptimal_conditional_seed2_full_cpu_20261104_v1/independent_audit.json)。
