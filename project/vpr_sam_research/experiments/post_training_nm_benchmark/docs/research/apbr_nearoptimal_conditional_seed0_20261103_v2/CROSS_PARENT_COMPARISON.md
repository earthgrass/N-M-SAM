# SAM 与 APBR：Seed 0 条件贪心 OBS 对照

同一初始权重、100 类 ResNet18、200 epochs、官方 CIFAR-100、同一 1024 张 Train 校准样本、19 层严格 2:4、相同物理联合 OBS 补偿。两条路径的 Dense 训练和 Mask 选择不同：SAM 使用纯物理条件贪心；APBR 使用已冻结的 1% 近优窗口和真实 APBR 几何裁决。均无 Recovery/BN-only。

| 完整路径 | Dense Final | 剪枝及补偿后 Final Test | 相对本身 Dense 的变化 | CE |
|---|---:|---:|---:|---:|
| SAM + 条件贪心 OBS | 78.84% | 77.84% | -1.00 pp | 0.828370 |
| APBR + 同一条件贪心 OBS | 78.68% | 77.72% | -0.96 pp | 0.829459 |
| APBR + 几何近优条件贪心 OBS | 78.68% | 77.83% | -0.85 pp | 0.829065 |

最终准确率 APBR−SAM = -0.01 pp，即 10,000 张 Test 中 APBR 少正确 1 张。
APBR 同父消融：纯物理条件贪心 77.72%，单位几何 77.76%，真实几何 77.83%，打乱几何 77.82%。真实几何只比打乱高 0.01 pp；一个 Seed 无法证明学习到的几何稳定有效。
这不是单独归因 APBR 几何的严格因果对照，因为训练路径与 Mask 选择都变了。同父 unit/true/shuffled 消融才直接检验几何坐标对应关系；目前该证据很弱。

SAM 历史条件贪心 checkpoint、APBR 四组冻结 checkpoint、父模型、校准 ID、Test pickle、架构及正式配置哈希均核对。APBR 纯物理路径与历史条件贪心产物在 19 层逐权重 Bitwise Equal。此次比较脚本没有重新评估 Test。
SAM 原始结果：`C:\Users\earthgrass73\Desktop\NM稀疏\APBR-post-training-benchmark\vpr_sam_research\experiments\post_training_nm_benchmark\artifacts\sam_singleton_conditional_seed0_cpu_20261103\results.json`。APBR 新结果：`C:\Users\earthgrass73\Desktop\NM稀疏\APBR-post-training-benchmark\vpr_sam_research\experiments\post_training_nm_benchmark\artifacts\apbr_nearoptimal_conditional_seed0_full_cpu_20261103_v1\results.json`。机器可读比较：`C:\Users\earthgrass73\Desktop\NM稀疏\APBR-post-training-benchmark\vpr_sam_research\experiments\post_training_nm_benchmark\docs\research\apbr_nearoptimal_conditional_seed0_20261103_v2\CROSS_PARENT_COMPARISON.json`。
