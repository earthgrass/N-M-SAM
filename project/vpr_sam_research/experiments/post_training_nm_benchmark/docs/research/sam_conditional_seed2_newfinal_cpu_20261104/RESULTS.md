# 新回传 SAM Seed 2：本地条件贪心 OBS

使用本次新服务器回传并验证的 SAM Seed 2 Epoch-200 Dense Final，独立运行固定 1,024 张 CIFAR-100 训练图片的校准、19 层 2:4 条件贪心选择和联合 OBS 补偿。无 Recovery、无 BN-only。Mask 与补偿 checkpoint 的哈希在 Test 前冻结，然后各对 Dense 和剪枝后模型执行一次固定 10,000 张 Test。

| 终点 | Test Accuracy | Test CE | 相对 Dense |
|---|---:|---:|---:|
| 新回传 SAM Dense Final | 78.80% | 0.804762 | — |
| 新运行条件贪心 OBS | **77.43%** | **0.839221** | −1.37 pp |

新父 checkpoint SHA256：`070c8d04af80a0bf011647aa8bad65e65e3fd64bedde64780adc029b40b07ec4`。旧父 checkpoint SHA256：`558ffa8da0a886a1291746fb6b5d979a2e83938b1150136568edea5e116fb977`。虽然 checkpoint 元数据及配置哈希不同，父模型全部 122 个状态张量逐项 Bitwise Equal。本次新剪枝与旧运行的 19 层 Mask、全部补偿后模型状态张量、Accuracy 和 CE 也逐项相同。该一致性来自两次实际独立运行的结果核对。

独立产物审计通过：11,157,504 个 eligible 坐标、5,578,752 个删除位置、严格 2:4、bool Mask、删除权重精确为零、数值有限，BN 与排除层保持父模型状态。训练、校准和剪枝阶段均未访问 Test。

证据：[新运行结果](../../../artifacts/sam_conditional_seed2_newfinal_cpu_20261104_v1/results.json)、[Test 前冻结](../../../artifacts/sam_conditional_seed2_newfinal_cpu_20261104_v1/pre_test_freeze.json)、[独立审计](../../../artifacts/sam_conditional_seed2_newfinal_cpu_20261104_v1/independent_audit.json)、[新旧逐张量复核](../../../artifacts/sam_conditional_seed2_newfinal_cpu_20261104_v1/replay_comparison.json)、[父 checkpoint 回传证明](../../../artifacts/dense_sam_apbr_l2_seed1to4_20261103_local_copy/seed2/dense_sam/LOCAL_SHA256.json)。
