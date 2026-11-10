# HQ-APBR V1 阶段状态

核查日期：2026-11-05。此仓库是 `NM稀疏/HQ-APBR-V1` 下的独立研究实现；现有 APBR-SAM、Dense-SAM、SparseGPT、OBS 代码和正式产物只作只读参照。

| 阶段 | 当前状态 | 已有证据与限制 |
|---|---|---|
| Phase 0：来源仓库与协议核查 | 已完成 | [Repository Audit](hq_apbr_repository_audit.md) 核对了 CIFAR-100 正式配置、100 类模型、19 层剪枝范围、原始 APBR 两步优化、SparseGPT 校准和来源状态。原项目工作区本来就是 dirty；未清理或覆盖。 |
| Phase 1：数学定义 | 已完成定义，待实验检验 | [数学文档](../docs/hq_apbr_v1_math.md) 区分带阻尼局部重建目标、历史几何 EMA、固定全局 L2 扰动和静态 Final-Q Mask；明确与原 APBR 能量反馈及原生 SparseGPT 动态 Mask 的差异。 |
| Phase 2：CPU 数值门禁 | **CPU_NUMERICAL_GATE_PASS（参考实现）** | [数值报告](hq_apbr_cpu_numerical_audit.md)、[JUnit](hq_apbr_cpu_pytest.xml) 和 [误差 JSON](hq_apbr_v1_cpu_numerical_metrics_v2.json) 记录修订后 **12 passed**、具体误差和源码哈希。H 是确定性合成短训练的恢复检查，不能替代真实 HQ 训练恢复门禁。 |
| Phase 3：真实 GPU 性能门禁 | **BLOCKED_FOR_CURRENT_GPU_MEASUREMENT** | [Compute Audit](hq_apbr_compute_audit.md) 已计算真实 19 层 Gram 尺寸及静态内存；本机 PyTorch 是 CPU build，CUDA 不可用。历史 GPU SparseGPT/Joint smoke 不是 HQ 每 Epoch 精确逆对角的耗时或峰值显存。 |
| Phase 4–5：训练适配与 Pilot | 未启动 | 当前代码仅是独立数学/CPU 参考；没有完整的 HQ 训练循环、真实校准状态提交、GPU Pilot 或正式 checkpoint。 |
| Phase 6–7：对照和完整训练 | 未启动 | 未访问 Test，未训练 200 Epoch，未评测 HQ 稀疏准确率；现阶段不具备启动完整正式实验的证据。 |

## 已建立、已验证与未知

- **数学已建立：** 在指定正定、带阻尼的局部二次目标下，单坐标 OBS 代价为 `w_i²/(2Q_ii)`；按 eligible 权重数全局中心化的目标满足零和；EMA 给出历史加权尺度；非零方向的全参数扰动 L2 范数为 `rho`。这些结论并不推出分类性能优势。
- **实现验证范围：** 已通过合成矩阵、合成恢复训练和本机 CPU 的映射/边界测试。历史模型的结构和数据身份审计不能替代真实 HQ 前向、反向、BN 隔离与 GPU 状态恢复测试。
- **经验上未验证：** HQ 是否改善 Dense、Mask-only 或固定 Mask OBS 后的准确率；是否优于 SAM/APBR、计算开销是否合理、跨 Seed 是否稳定。上述项目均不能从公式或历史方法结果推定。

## 下一门禁

1. 冻结训练侧校准 ID/patch 上限、刷新频率、`eta`、Hessian 常数口径及死特征规则、阻尼、精度与数值容差；保留配置和源码哈希。
2. 在真实 GPU 上逐层测量 19 层的激活采集、Gram、Cholesky、精确 inverse-diagonal、几何状态更新和同步 CUDA 时间及峰值显存；量化对每 Epoch 训练的增量成本。若不足以支撑每轮刷新，只提出新候选协议，不改动 V1 并继续宣称同一方法。
3. 只有 CPU 与 GPU 门禁、真实 HQ 小规模 Pilot、BN/SGD/恢复状态检查及超参数冻结均完成后，才评估是否具备完整 200 Epoch 的启动条件。本仓库不会自动启动该训练。

目前没有 HQ checkpoint、Mask、Final Test 或结果表；无需将历史实验数据当作 HQ 的表现。
