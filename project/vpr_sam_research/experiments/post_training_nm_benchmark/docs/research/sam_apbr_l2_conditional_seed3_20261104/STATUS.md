# Seed 3：独立结果与待完成的 SAM 对照

三路使用相同初始化哈希。原版 APBR 与 APBR-L2 的 200 Epoch Dense Final、条件贪心 OBS、19 层 2:4 审计和各一次固定 Test 已完成。SAM Seed 3 的可用完整 Final checkpoint 尚未生成；此前本地旧副本不完整，不能用于评测。当前远端队列会在 APBR-L2 Seed 4、SAM Seed 2 之后训练 SAM Seed 3。本文件保持未完成状态，不能用其他 Seed 填补 SAM 数据。

| 路径 | Dense Final | 纯物理条件贪心 OBS | 真实几何近优裁决 + OBS | 打乱几何 + OBS |
|---|---:|---:|---:|---:|
| 原版 APBR | 78.61% | 77.31% | 77.21% | 77.16% |
| APBR-L2 | 78.64% | **77.47%** | 77.38% | 77.36% |
| SAM | 待完整 Final | 待运行 | 不适用 | 不适用 |

在 Seed 3，两种 APBR 的真实几何裁决均低于各自纯物理条件贪心，分别为 −0.10 和 −0.09 pp。不同 Dense 父模型的直接差值不能单独归因于几何选择。所有已完成的剪枝均从对应的原始 Dense 权重出发，无 Recovery、无 BN-only；Test 在 Mask 与 checkpoint 冻结后执行。

原版 APBR 父 checkpoint SHA256：`b5289b0bb6be67cef28d675c4d1dc6a4703ab615c60d707296f12c6bbd00b5e3`。APBR-L2 父 checkpoint SHA256：`922a5adf12aba494086cff3f0b00d3cfed6d364e5f33ecea667bf7f6fe3284ad`。独立审计均通过：19 层、11,157,504 个 eligible 坐标、5,578,752 个删除坐标、严格 2:4、删除权重为零、数值有限；BN 与排除层保持父模型状态。

证据：[原版 APBR 结果](../../../artifacts/apbr_vpr_nearoptimal_conditional_seed3_full_cpu_20261104_v1/results.json)、[原版 APBR 独立审计](../../../artifacts/apbr_vpr_nearoptimal_conditional_seed3_full_cpu_20261104_v1/independent_audit.json)、[APBR-L2 结果](../../../artifacts/apbr_l2_nearoptimal_conditional_seed3_full_cpu_20261104_v1/results.json)、[APBR-L2 独立审计](../../../artifacts/apbr_l2_nearoptimal_conditional_seed3_full_cpu_20261104_v1/independent_audit.json)。
