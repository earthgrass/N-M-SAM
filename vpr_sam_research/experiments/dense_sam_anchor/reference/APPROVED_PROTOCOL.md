# Dense-SAM 强基线冻结与执行规范

**实验名称：** Dense-SAM-SSAM-Config-FSAM-BN  
**版本：** v1.0  
**日期：** 2026-10-24  
**状态：** 已批准执行配置；先完成最小验收，再执行正式 200-epoch 训练。  
**首轮种子：** seed=0

---

## 1. 实验目标与边界

本实验的唯一目标是建立一个强、稳定、可复现的 CIFAR-10 / ResNet-18 Dense-SAM 基线，为后续 VPR + Guide 提供统一的稠密模型起点。

主要参考：

1. **SSAM，NeurIPS 2022**：Dense-SAM 的超参数、网络结构、数据增强与 96.83% 的论文参考值。
2. **F-SAM，CVPR 2024**：第二次 SAM 前向传播的 BatchNorm running statistics 冻结策略。
3. 原始 SAM：标准两次前向传播、两次反向传播和 SGD 更新机制。

本实验不是 SSAM 的逐项原样复现。我们使用独立验证集进行 checkpoint 选择，采用 F-SAM 风格的 BN 处理，并沿用项目现有的 45,000 / 5,000 数据划分。

**96.83% 是文献参考值，不是本次训练必须达到的硬性门槛。** 不得通过测试集选模、测试集调参或事后挑选训练配置来追求该数字。

## 2. 最终冻结的训练配置

| 配置项 | 冻结值 |
|---|---|
| Dataset | CIFAR-10 |
| Train / Validation / Test | 45,000 / 5,000 / 10,000 |
| Model | CIFAR-ResNet-18 |
| Initialization | 从头训练，随机初始化 |
| Epochs | 200 |
| Batch size | 128 |
| Optimizer | SGD |
| Initial learning rate | 0.05 |
| Momentum | 0.9 |
| Weight decay | 0.001 |
| Nesterov | False |
| Scheduler | Cosine Annealing |
| Scheduler interval | 每个 epoch 更新一次 |
| `eta_min` | 0.0 |
| SAM radius | `rho = 0.1` |
| Adaptive SAM | False |
| Loss | CrossEntropyLoss |
| Label smoothing | 0 |
| AMP / FP16 / BF16 | 禁用 |
| TF32 | 禁用 |
| `torch.compile` | 禁用 |
| Training batch tail | 保留最后一个不完整 batch |
| DataLoader workers | 2 |
| Pin memory | True |
| Seed | 0 |
| Early stopping | 禁用 |
| Checkpoint selection | Best Validation Top-1 Accuracy |

对于 45,000 张训练图片，保留尾 batch 时，每个 epoch 有 352 个 optimizer steps，总计 70,400 steps。

### 2.1 网络结构

采用 SSAM 官方实现中的 CIFAR-ResNet-18 结构，不使用未经核对的 ImageNet 版 torchvision ResNet-18。

必须核对以下结构特征：

- 第一层卷积为 `3×3`，stride=1，padding=1。
- 使用 BasicBlock，四个 stage 的 block 数量为 `[2, 2, 2, 2]`。
- 各 stage 的通道数为 64、128、256、512。
- stage 下采样步长为 1、2、2、2。
- 使用 CIFAR 输入对应的全局平均池化和 10 类分类头。
- 不加载预训练权重。

Codex 必须检查当前项目的模型定义，并将实际结构、参数名称、参数形状和源码 SHA256 写入 manifest。若现有模型与上述结构不一致，应在独立实验模块中实现正确结构，不得静默修改历史 VPR 代码。

### 2.2 数据预处理

采用 SSAM 官方代码的 CIFAR-10 归一化参数：

```yaml
normalize:
  mean: [0.4913725490196078, 0.4823529411764706, 0.4466666666666667]
  std:  [0.2470588235294118, 0.2435294117647059, 0.2615686274509804]
```

训练集增强顺序：

1. `RandomCrop(32, padding=4)`
2. `RandomHorizontalFlip()`
3. `ToTensor()`
4. `Normalize(mean, std)`
5. `Cutout(size=16, p=0.5)`

Cutout 在归一化后执行，与 SSAM 官方实现保持一致。

验证集和测试集仅使用：

1. `ToTensor()`
2. `Normalize(mean, std)`

不使用随机裁剪、翻转、Cutout、MixUp、AutoAugment、EMA 或其他未预先规定的增强。

### 2.3 数据划分

- 从 CIFAR-10 官方 50,000 张训练图片中划分 45,000 张训练图片和 5,000 张验证图片。
- 优先复用项目已经冻结的 seed=0 划分及 sample IDs。
- 如果不存在可复用的划分文件，使用固定种子的分层划分，并保存完整 sample IDs、类别分布和 SHA256。
- 验证集不参与梯度更新。
- 官方 10,000 张测试图片不参与训练、验证、checkpoint 选择或超参数调整。

本实验与 SSAM 原论文使用全部 50,000 张训练图片的协议存在数据量差异。必须如实记录，不能将本实验标为 SSAM 的严格复现。

## 3. BatchNorm：唯一执行规则

**每个训练 batch 仅允许第一次前向传播更新 BN running buffers。**

SAM 仍然需要两次前向传播和两次反向传播。

### 第一次前向传播

- `model.train()`。
- BN 使用当前 batch statistics。
- 正常更新 `running_mean`、`running_var` 和 `num_batches_tracked`。
- 计算原始参数位置的 loss 和梯度 \(g_1\)。

### 第二次前向传播

- 保持模型训练模式，不能对整个模型调用 `model.eval()`。
- BN 仍使用当前 batch statistics 进行前向计算。
- 冻结 BN running buffers，使第二次前向不会再次更新 running mean、running variance 或计数器。
- 计算扰动参数位置的 loss 和梯度 \(g_2\)。

建议实现方式：在第二次前向之前，备份所有 BatchNorm 模块的 `running_mean`、`running_var`、`num_batches_tracked`；完成第二次前向和反向传播后恢复这些 buffers。也可采用 F-SAM 的 `enable_running_stats` / `disable_running_stats` 思路，但必须通过测试确认三个 buffers 均符合上述规则。

不得通过将 BN 切换到 eval 模式来冻结统计量，因为这会改变第二次前向传播使用的统计量。

### 必须通过的 BN 测试

使用一个包含 BatchNorm 的小模型和固定输入，验证：

1. 第一次前向后 running mean / variance 按预期发生变化。
2. 第二次前向后所有 BN running buffers 与第二次前向之前完全一致。
3. 第二次前向仍能正常反向传播并产生有限梯度。
4. 完整 SAM step 后，参数和 BN buffers 均无 NaN / Inf。
5. 验证阶段使用 `model.eval()`，且不改变 BN buffers。

该策略是 F-SAM 风格的受控 BN 处理，与 SSAM 官方代码的双次更新行为不同，必须在实验报告中披露。

## 4. SAM optimizer 实现要求

使用标准 SAM + SGD，不使用 FriendlySAM、ASAM 或其他变体。

对于原始参数 \(\theta\)，第一次梯度为 \(g_1\)，扰动为：

\[
\epsilon = \rho \frac{g_1}{\|g_1\|_2 + \varepsilon_{\mathrm{num}}}
\]

其中 `rho=0.1`，梯度范数必须在全部可训练参数上统一计算，不能按 tensor 分别归一化。

每个训练 step 必须严格执行：

1. 原始参数位置前向和反向，获得 \(g_1\)。
2. 保存原始参数，并施加全局 SAM 扰动。
3. 清除第一次梯度。
4. 在扰动参数位置进行第二次前向和反向，获得 \(g_2\)；冻结 BN running buffers。
5. 精确恢复原始参数。
6. 使用 \(g_2\) 执行一次 SGD 更新。
7. 清除梯度，为下一个 batch 做准备。

注意：

- SGD weight decay 只在最终 base optimizer 更新中正常应用。
- 第二次反向得到的 \(g_2\) 必须保留到 SGD 更新执行完成。
- 不得在更新之前清除 \(g_2\)。
- 不得将扰动后的参数直接作为最终模型参数保存。
- 不得在每个 tensor 上分别计算 SAM 梯度范数。

## 5. 学习率调度

使用 `torch.optim.lr_scheduler.CosineAnnealingLR`，绑定到底层 SGD optimizer。

```yaml
scheduler:
  name: CosineAnnealingLR
  T_max: 200
  eta_min: 0.0
  interval: epoch
  warmup_epochs: 0
```

每个 epoch 完成训练后调用一次 scheduler step，并记录本 epoch 实际使用的学习率。

训练开始时的学习率必须为 0.05。不得额外引入 warmup、分段衰减或根据验证结果改变调度方式。

## 6. Checkpoint 与模型选择

必须区分最佳验证集模型、训练恢复状态和第 200 轮模型。

### `best_val.pt`

这是后续 VPR + Guide 的唯一默认初始化来源。

- 每轮完成训练后，在固定验证集上计算 Top-1 Accuracy。
- 仅当本轮验证准确率严格高于历史最高值时更新。
- 准确率相同则保留最早达到该准确率的轮次。
- 记录最佳 epoch、Best Validation Accuracy 和 Best Validation Loss。
- 保存完整模型权重、优化器状态、scheduler 状态、epoch、配置、随机数状态及数据划分标识。

### `last.pt`

- 每个 epoch 后原子保存，用于断点续训。
- 必须包含恢复训练所需的完整状态。
- 不能默认作为 VPR + Guide 的初始化来源。

### `epoch_200.pt`

- 单独保留第 200 轮结束后的模型和训练状态。
- 用于审计最后一轮与最佳验证集轮次的差异。
- 不替代 `best_val.pt`。

所有 checkpoint 必须采用原子写入，写入完成后重新读取校验，并生成 SHA256。

## 7. 官方测试集评价

完成全部 200 epochs 后：

1. 加载 `best_val.pt`。
2. 在 CIFAR-10 官方 10,000 张测试图片上进行一次最终评价。
3. 记录 Test Top-1 Accuracy、Test Loss、最佳验证 epoch 和对应验证指标。
4. 保存测试评价结果和 checkpoint SHA256。

严禁在训练过程中逐 epoch 使用官方测试集选模。不得根据最终 Test Accuracy 调整学习率、rho、weight decay、Cutout 或 BN 处理方式。

如果 Test Accuracy 低于 96.83%，应如实报告，不得自动搜索参数、挑选其他测试集表现更好的 checkpoint 或删除不理想的运行。

## 8. 随机性、数值精度与环境

```yaml
reproducibility:
  seed: 0
  deterministic: true
  cudnn_benchmark: false
  amp: false
  tf32: false
  torch_compile: false

runtime:
  batch_size: 128
  num_workers: 2
  pin_memory: true
  drop_last: false
```

设置并记录 Python、NumPy、PyTorch CPU、PyTorch CUDA 和 DataLoader worker 的随机种子。

启动前重新检查 GPU UUID、显存、磁盘余量和数据完整性。目标 GPU 为之前审计的 RTX 5090；只有在启动前确认设备身份与空闲状态后才能使用。

优先使用现有的 PyTorch 2.8.0 + CUDA 12.8 环境，不为了运行老论文代码而重装旧版 CUDA 或 PyTorch。

建议在独立 worktree 和独立实验目录中运行，例如：

`artifacts/dense_sam_anchor_20261024/seed0/`

不得删除历史 checkpoint、覆盖旧实验日志、修改冻结的 VPR 数学模块或更改历史 source manifest。

## 9. 执行门禁

### G0：只读预检

- 读取适用的 `AGENTS.md`。
- 检查 Git 分支、工作区状态和未提交文件。
- 检查 GPU、磁盘和 CIFAR-10 数据身份。
- 核对模型结构、配置与数据划分来源。

### G1：最小实现验收

在合成数据上验证：

- SAM 参数扰动与恢复正确。
- 第二次梯度没有丢失。
- BN 仅第一次前向更新 running buffers。
- 所有 loss、梯度和参数有限。
- checkpoint 保存、读取和断点恢复正确。

然后在真实 CIFAR-10 数据上运行少量 batch，检查数据预处理、训练循环和来源记录。

### G2：正式训练

G1 全部通过后，立即执行 seed=0 的完整 200-epoch 训练。不得因为验证准确率暂时停滞而提前停止。

如果出现 NaN、参数恢复失败、数据身份不一致、BN 测试失败或磁盘不足，应停止训练并保留错误证据。修复实现问题后重新执行对应验收，不得自行改变已冻结的算法超参数。

## 10. 必须交付的产物

```text
dense_sam_anchor_20261024/
├── config.yaml
├── resolved_config.yaml
├── run_manifest.json
├── split_manifest.json
├── metrics.csv
├── epoch_metrics.jsonl
├── best_val.pt
├── last.pt
├── epoch_200.pt
├── best_val.json
├── test_eval.json
├── checkpoint_sha256.txt
├── stdout.log
├── training_summary.md
└── tests/
    ├── test_sam_parameter_restore.py
    ├── test_second_gradient.py
    ├── test_bn_running_buffers.py
    └── test_checkpoint_resume.py
```

实际实现可以调整目录细节，但必须保留上述信息与功能。

`training_summary.md` 至少报告：

- 实际运行配置及其 SHA256。
- 模型结构与数据划分身份。
- 200 epochs 是否完整完成。
- Best Validation Accuracy、Best Epoch、Best Validation Loss。
- Epoch 200 的 Validation Accuracy 与 Loss。
- 最佳验证 checkpoint 的最终 Test Accuracy 与 Loss。
- BN 策略和 SAM 更新逻辑的验收结果。
- GPU、峰值显存、训练时间、磁盘情况。
- 所有 checkpoint 的路径、大小与 SHA256。
- 与 SSAM 论文协议的差异。

## 11. 后续 VPR + Guide 的衔接规则

本轮只建立 Dense-SAM 基线，不同时启动 SSAM、CrAM、S²-SAM 或其他新训练。

训练成功并通过产物审计后：

1. 冻结本轮 `resolved_config.yaml`。
2. 将 `best_val.pt` 作为后续 VPR + Guide 的默认稠密模型父节点。
3. 所有后续剪枝与对照分支必须记录相同的父 checkpoint SHA256。
4. 不对该 checkpoint 额外微调后再开始剪枝。
5. 在论文级结论形成前，使用相同配置补齐其他随机种子的配对实验；seed=0 的单次运行不能代替三种子统计证据。

**现在的执行顺序是：预检 → 最小验收 → 200 epochs Dense-SAM → 验收最佳 checkpoint → 冻结基线。** 不要在本轮扩大研究范围。

## 12. 一手参考来源

- SSAM 论文：<https://arxiv.org/abs/2210.05177>
- SSAM 官方代码（固定 commit）：<https://github.com/Mi-Peng/Sparse-Sharpness-Aware-Minimization/tree/76974fd0255cf2f61201e19566cb64ceb429ecdd>
- SSAM 训练入口：<https://github.com/Mi-Peng/Sparse-Sharpness-Aware-Minimization/blob/76974fd0255cf2f61201e19566cb64ceb429ecdd/train.py>
- SSAM CIFAR 数据增强：<https://github.com/Mi-Peng/Sparse-Sharpness-Aware-Minimization/blob/76974fd0255cf2f61201e19566cb64ceb429ecdd/data/dataset.py>
- F-SAM 论文：<https://openaccess.thecvf.com/content/CVPR2024/html/Li_Friendly_Sharpness-Aware_Minimization_CVPR_2024_paper.html>
- F-SAM 官方代码：<https://github.com/nblt/F-SAM>

以上配置是本轮唯一执行版本。任何超参数变更都必须先形成书面差异记录，不得在正式训练过程中自行调整。
