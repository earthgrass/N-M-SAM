# APBR-SAM Seed 0 + Fixed-Mask OBS 实验报告

## 结论

本次独立探索实验工程验收通过。Seed 0 APBR-SAM Epoch 200 稠密 checkpoint 生成的 19 层 APBR 2:4 Mask 与历史 Raw Hard Mask 逐元素一致；在 Mask 固定不变的前提下，OBS 补偿真实修改了幸存权重，并将 CIFAR-100 Test 准确率从 59.22% 提升到 **76.37%**，提升 **17.15 个百分点**。Test CE 从 1.64040 降到 **0.90398**。

该结果不替代预注册 APBR-SAM Recovery-20 终点，也没有加入三 Seed 正式主表。它表明 APBR 几何 Mask 可以从二阶误差补偿中获得显著收益，值得继续研究 Geometry-Aware Compensation。

## 固定实验身份

- 数据：官方 CIFAR-100，50,000 Train + 10,000 Test。
- 模型：`ssam2022_cifar_resnet18_c100_v1`。
- Seed：0。
- 稠密父 checkpoint SHA256：`4332079becdb58fb50f0e68b8bd0dcc251db0184a6050facb524817520a51d82`。
- 历史 APBR Raw Hard SHA256：`74ea702356b6cb3f10ad34985ca8de6757ccb80702cced3debe4ec5936b9b9b3`。
- APBR Mask SHA256：`79d9c410c67e931101cf46112e732c68133a9935e4f99a41503a0de8ee7b40f4`。
- 新 compensated checkpoint SHA256：`4907a607f7a45ff3a9cc9a5c0bf66b2cfdc26260e2f019b3967a7095e22ae86f`。
- GPU：NVIDIA RTX PRO 6000 Blackwell Server Edition，UUID `GPU-65be6845-be69-8ba1-72f8-e34688ce53c5`。
- PyTorch：2.8.0+cu128；驱动 580.82.09。

## 结果

| 方法 | 机制 | Test Accuracy | Test CE Loss |
|---|---|---:|---:|
| APBR-SAM Raw Hard | Geometry Mask-only | 59.22% | 1.64040 |
| **APBR + Fixed-Mask OBS** | Geometry Mask + 固定支持二阶补偿 | **76.37%** | **0.90398** |
| APBR-SAM Recovery-20 Final | Geometry Mask + 20 epochs 固定 Mask 微调 | 78.72% | 0.82378 |
| SparseGPT Native | Dynamic OBS Mask + 二阶补偿 | 77.47% | 0.85026 |
| ROSE Native | 重排序 + 二阶补偿 | 78.03% | 0.82311 |

相对于 APBR Raw Hard，Fixed-Mask OBS 提升 17.15 pp；距离 APBR Recovery-20 仍有 2.35 pp，距离 SparseGPT Native 1.10 pp，距离 ROSE Native 1.66 pp。SparseGPT/ROSE 使用 Dense-SAM 父模型，而本实验使用 APBR-SAM 父模型，因此跨父模型数字只能比较完整流程表现，不能作为 Mask 质量的严格因果结论。

## 执行边界

- 校准仅访问固定的 CIFAR-100 Train 样本；补偿阶段 `test_accessed=false`。
- Test 只在 compensated checkpoint 固定后评估一次，没有选模、调参或自动重试。
- 没有梯度训练，没有 Recovery epoch，没有更新 APBR 几何。
- 首次启动因预登记 GPU UUID 多写一位而在 GPU 身份前检阶段失败；没有构造 Hessian、没有访问 Test。修正为从 `nvidia-smi` 实测的完整 UUID 后执行唯一一次数值实验，失败证据保留。

## 科研判断

Fixed-Mask OBS 对 APBR Mask 明显有效：它恢复了 Raw Hard 与 Recovery-20 间约 89.7% 的准确率差距（17.15 / 19.50 pp），且全部 19 层的局部二次重构误差均低于对应 Mask-only 误差。它尚未达到 APBR Recovery-20，但只需秒级补偿而非 20 epochs 训练。下一步若继续，应把“几何 Mask 与 OBS 消元顺序/分块的耦合”作为独立研究问题，而不是用 Test 反复搜索阻尼或顺序。

完整机器可读证据见 `experiment_evidence.json` 及本地 artifact 目录中的 `compensation_evidence.json`、`evaluation.json`、checkpoint 和 SHA256 清单。
