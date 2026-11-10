# Wanda vs APBR-SAM Seed 0 只读审计

## 核心结论

- `VERIFIED`：当前 Wanda-CNN adapter 保留官方核心量 `|W_ij| sqrt(scaler_row_j)`、逐 output-row 连续 2:4 组内删除、Mask-only 写回和幸存权重不变。Conv2d 的 `unfold` 顺序与 OIHW 展平一致。
- `VERIFIED`：Seed 0 Wanda 的 19 层 bool keep-mask 全部满足每个完整四元组保留 2 个、dense tail、删除位为零、幸存权重与 Dense 父 checkpoint 逐元素完全相等。
- `SUPPORTED_HYPOTHESIS`：APBR Raw accuracy 比 Wanda 高 0.68 pp；相对各自父模型的下降少 0.84 pp。这支持“Seed 0 上 APBR 几何与更小的即时准确率损伤相关”，不证明机制因果或跨 seed 优势。
- `UNVERIFIED`：BN-only 没有 Test 评估；Recovery 过程没有逐 epoch Test。因此不能把 Raw 差异归因于 BN，也不能观察测试准确率的反超时刻。
- `VERIFIED`：Wanda Recovery-20 确实从 Wanda Raw checkpoint 开始，APBR 同理；父 SHA、Mask SHA、20 epochs/7,820 steps 均匹配。

## 运行与协议状态

本报告只读取本地 Seed 0 完整镜像。审计启动时远端控制器仍在运行 Seed 1 Dense-SAM，未中断、未重启、未修改运行代码或配置；Seed 2 尚未启动。

## 官方 Wanda 与当前 adapter

官方锁定 commit [`8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6`](https://github.com/locuslab/wanda/tree/8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6) 的 [`prune_wanda`](https://github.com/locuslab/wanda/blob/8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6/lib/prune.py) 与 [`WrappedGPT`](https://github.com/locuslab/wanda/blob/8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6/lib/layerwrapper.py) 以输入列平方范数累计 `scaler_row`，评分为 `abs(W)*sqrt(scaler_row)`，N:M 在每个 output row 的连续列内删除低分项，不更新幸存权重，也没有默认 OBS 补偿。官方视觉 [`prune_utils.py`](https://github.com/locuslab/wanda/blob/8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6/image_classifiers/prune_utils.py) 只查找 `nn.Linear`，不是 ResNet18 Conv2d N:M 实现。

当前调用链是 `run_cifar100_50k_full_queue.py` → `run_cifar100_50k_postprocess.py::prune` → `WandaAdapter.add_batch/prune`。1024 个固定 Train ID、无增强、单次前向；`model.eval()` 避免 BN buffer 更新。每层只有一个 pre-hook，循环结束立即移除。FP64 累积会造成与官方 FP32 buffer 的微小数值差异；此前单层对照最大 score 差约 `1.08e-7`、Mask 完全一致。Seed 0 并未保存运行时 score/激活统计，因此这些张量不能从 checkpoint 事后复核。

## 19 层全模型检查

| Layer | Weight shape | K | Calibration images | Score stats | Mask SHA256 | Sparsity | Survivors equal parent | Audit |
|---|---:|---:|---:|---|---|---:|---|---|
| layer1.0.conv1 | [64, 64, 3, 3] | 576 | 1024 | NOT SAVED | `243149cff84c423e…` | 50.00% | True | PASS |
| layer1.0.conv2 | [64, 64, 3, 3] | 576 | 1024 | NOT SAVED | `330b2586e8a6ec12…` | 50.00% | True | PASS |
| layer1.1.conv1 | [64, 64, 3, 3] | 576 | 1024 | NOT SAVED | `55f3f246934f2742…` | 50.00% | True | PASS |
| layer1.1.conv2 | [64, 64, 3, 3] | 576 | 1024 | NOT SAVED | `e700af11ff2d93fc…` | 50.00% | True | PASS |
| layer2.0.conv1 | [128, 64, 3, 3] | 576 | 1024 | NOT SAVED | `b86c263120496f75…` | 50.00% | True | PASS |
| layer2.0.conv2 | [128, 128, 3, 3] | 1152 | 1024 | NOT SAVED | `ecd687b4a98f71fd…` | 50.00% | True | PASS |
| layer2.0.shortcut.0 | [128, 64, 1, 1] | 64 | 1024 | NOT SAVED | `39d63bc95d29bf49…` | 50.00% | True | PASS |
| layer2.1.conv1 | [128, 128, 3, 3] | 1152 | 1024 | NOT SAVED | `6fc12c8a291b864a…` | 50.00% | True | PASS |
| layer2.1.conv2 | [128, 128, 3, 3] | 1152 | 1024 | NOT SAVED | `c2e8b67f4854eaf1…` | 50.00% | True | PASS |
| layer3.0.conv1 | [256, 128, 3, 3] | 1152 | 1024 | NOT SAVED | `f4717a441ad62724…` | 50.00% | True | PASS |
| layer3.0.conv2 | [256, 256, 3, 3] | 2304 | 1024 | NOT SAVED | `2103e8037bb603d2…` | 50.00% | True | PASS |
| layer3.0.shortcut.0 | [256, 128, 1, 1] | 128 | 1024 | NOT SAVED | `9b6a5ea83e3de1e9…` | 50.00% | True | PASS |
| layer3.1.conv1 | [256, 256, 3, 3] | 2304 | 1024 | NOT SAVED | `f72bcc50e02088f9…` | 50.00% | True | PASS |
| layer3.1.conv2 | [256, 256, 3, 3] | 2304 | 1024 | NOT SAVED | `73a2694e8268efe7…` | 50.00% | True | PASS |
| layer4.0.conv1 | [512, 256, 3, 3] | 2304 | 1024 | NOT SAVED | `b790ef072aa7a6c2…` | 50.00% | True | PASS |
| layer4.0.conv2 | [512, 512, 3, 3] | 4608 | 1024 | NOT SAVED | `cb4173280b195eb8…` | 50.00% | True | PASS |
| layer4.0.shortcut.0 | [512, 256, 1, 1] | 256 | 1024 | NOT SAVED | `7211fee0997d70b9…` | 50.00% | True | PASS |
| layer4.1.conv1 | [512, 512, 3, 3] | 4608 | 1024 | NOT SAVED | `992a8a3e5532c743…` | 50.00% | True | PASS |
| layer4.1.conv2 | [512, 512, 3, 3] | 4608 | 1024 | NOT SAVED | `108bbc074e098475…` | 50.00% | True | PASS |

总体：11,157,504 个 eligible 权重，删除 5,578,752，实际稀疏率 50.00%。父模型每层绝对值 top-10% 权重中，Wanda 删除 12.783%。运行时 score min/max/mean 和 finite 状态均标为 `NOT_VERIFIABLE_FROM_EXISTING_ARTIFACTS`，没有重新校准或重新剪枝来替代历史证据。

## Raw Hard：准确率与交叉熵

| 方法 | Parent Acc | Raw Acc | ΔAcc | Parent Loss | Raw Loss | ΔLoss |
|---|---:|---:|---:|---:|---:|---:|
| Wanda | 78.84% | 58.54% | -20.30 pp | 0.799886 | 1.631037 | +0.831150 |
| APBR-SAM | 78.68% | 59.22% | -19.46 pp | 0.803110 | 1.640396 | +0.837286 |

APBR 的 Raw accuracy 高 0.68 pp，但其 Raw cross-entropy 反而比 Wanda 高 +0.009359。因此“accuracy 更高”不能外推为“交叉熵损伤更小”。APBR 与 Wanda 使用不同父模型；Mask 坐标相同率 51.16%、keep-set Jaccard 0.3437 仅作描述，不是同父 Mask 质量比较。

## BN-only 诊断

源码对 `copy.deepcopy(model)` 重置 BN running stats，只让 BatchNorm 进入 train；前后参数 SHA 必须不变。checkpoint 使用 4096 个固定 Train ID、无增强，并记录 Raw parent SHA。该语义通过源码 guard 与 checkpoint 结构核验。但 12 个 Final Test target 不包含 BN-only，因此 `Δ_BN` 不可观测，BN 失配只能作为待检验解释。

## Recovery-20

| Epoch | Wanda train loss | Wanda train acc | APBR train loss | APBR train acc |
|---:|---:|---:|---:|---:|
| 1 | 0.022986 | 99.962% | 0.022817 | 99.956% |
| 5 | 0.013057 | 99.968% | 0.013156 | 99.948% |
| 10 | 0.011281 | 99.956% | 0.011248 | 99.960% |
| 15 | 0.010633 | 99.974% | 0.010665 | 99.970% |
| 20 | 0.010478 | 99.974% | 0.010654 | 99.974% |

两者 optimizer、LR schedule、weight decay、momentum、增强、train order hash、Mask 固定投影与 20-epoch budget 一致。Wanda Final 为 78.93%，APBR Final 为 78.72%，Wanda 高 0.21 pp。训练日志能显示训练损失轨迹，但没有中间 Test 指标，不能声称测试性能在某个 epoch 发生反超，也不能用 Test 选择 epoch。

## 结论边界

Seed 0 支持优先假设，但效应只有 0.68 pp 且父模型不同。必须等待 Seeds 1、2，报告三 seed mean ± sample SD 与同 seed 配对差，才能判断方向是否稳定。现有证据没有发现 Wanda/APBR 执行路径、父 checkpoint 或 Mask 固定方面的实现错误。
