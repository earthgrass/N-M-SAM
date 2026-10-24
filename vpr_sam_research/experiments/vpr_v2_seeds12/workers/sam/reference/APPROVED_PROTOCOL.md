# 任务：建立与 SSAM 文献基线对齐的 Dense-SAM，执行完整 200 Epoch 训练

## 一、任务目标与边界

我的研究目标是研究 SAM 与 N:M 结构化稀疏的结合。现阶段最重要的任务不是运行 VPR、Guide 或剪枝，而是先建立一个尽可能与高水平文献基线对齐的 Dense-SAM。

目标参考包括：

1. SSAM：Sparse-Sharpness-Aware-Minimization，NeurIPS 2022 官方仓库。
2. F-SAM：nblt/F-SAM 官方仓库，作为辅助参考。
3. 目标文献基线准确率约为 CIFAR-10 / ResNet18 的 96.83%。

SSAM 是本轮主要配置来源，因为目标准确率来自 SSAM 相关文献。F-SAM 用于核对实现差异，不要把两套互相冲突的配置拼接成所谓的官方配置。

**本轮只运行一条 Dense-SAM 训练轨迹，200 Epoch、单 Seed。**

本轮不运行：
- VPR-L2 或其他半径重分配方法。
- Geometry-guided、Magnitude 或其他 Mask。
- N:M 剪枝、BN-only 剪枝评价或恢复微调。
- SSAM-F、SSAM-D 等稀疏扰动算法。
- 超参数搜索、额外 Seed 或自动重跑。

训练完成后，向我交付最佳 checkpoint、最终 checkpoint、完整日志、配置、审计记录和 SHA256。后续是否开展 VPR+Guide 实验由我另行安排。

## 二、执行前检查

开始前必须：

1. 阅读当前仓库的 AGENTS.md、研究协议及相关实现。
2. 检查当前 Git 分支、工作树、暂存区和已有实验目录。
3. 检查现有 RTX 5090 服务器连接、GPU 使用情况、Python/PyTorch/CUDA 版本和磁盘可用空间。
4. 检查本地或服务器已有 CIFAR-10 数据的完整性。
5. 核对 SSAM 固定版本的官方配置、模型、数据增强、SAM 实现、学习率调度、BN 行为和 checkpoint 保存逻辑。

SSAM 参考版本：

https://github.com/Mi-Peng/Sparse-Sharpness-Aware-Minimization/tree/76974fd0255cf2f61201e19566cb64ceb429ecdd

F-SAM 参考仓库：

https://github.com/nblt/F-SAM

必须记录 F-SAM 的实际 commit，以及本次审计涉及的文件 SHA256。

不得修改、覆盖或删除此前 VPR-L2、Geometry-guided、三种子实验、恢复轨迹及其他历史产物。新实验必须放在独立目录和隔离的工作分支或 worktree 中。

数据已经存在时不得重新下载或上传。若数据完整性、服务器环境或磁盘空间未通过门禁，先报告阻塞原因，不得擅自清理历史文件。

## 三、冻结的 Dense-SAM 配置

以 SSAM 官方 CIFAR-10 / ResNet18 Dense-SAM 路径为主要参考。以下是本轮冻结配置；任何与官方实现不一致的部分必须记录为兼容性偏差，不得静默修改。

### 3.1 数据集

- 数据集：官方 CIFAR-10。
- 训练集：50,000 张。
- 测试集：10,000 张。
- 禁止再使用 45,000/5,000 的内部划分。
- 不得从训练集中额外抽取 5,000 张充当验证集。
- 不得将训练图片混入测试集或更换测试集。

遵循 SSAM 官方训练 DataLoader 的 `drop_last=True` 设置时：

- Batch size = 128。
- 每个 Epoch 为 390 个完整训练 Batch。
- 每个 Epoch 实际参与训练的样本数为 49,920。
- 每轮随机打乱后丢弃不足一个 Batch 的尾部样本。
- 200 Epoch 总计 78,000 次优化器更新。

必须在日志中记录实际样本数和 step 数，不得简单声称每轮都执行了全部 50,000 张图片的梯度更新。

测试集使用全部 10,000 张图片，评估 DataLoader 不丢弃尾部 Batch。

训练前校验 CIFAR-10 官方数据文件、样本数量、文件身份及已有数据的完整性；禁止重新生成或更改数据划分。

### 3.2 模型

- 模型：CIFAR 专用 ResNet18。
- 优先使用 SSAM 对应版本的模型定义。
- 不得直接替换成 ImageNet 版 torchvision ResNet18。
- 必须核对 stem、stride、max-pooling、残差层数量、分类头和初始化方式。
- 记录可训练参数总数、模型定义来源及模型初始化 Seed。

### 3.3 优化器和 SAM

| 参数 | 冻结值 |
|---|---|
| 优化器 | SGD |
| 初始学习率 | 0.05 |
| Momentum | 0.9 |
| Weight decay | 0.001 |
| Nesterov | False |
| SAM rho | 0.1 |
| Adaptive SAM | False |
| Batch size | 128 |
| Epochs | 200 |
| Random seed | 1234 |
| 损失函数 | 标准 CrossEntropyLoss |
| 混合精度 | 关闭，使用 FP32 |

采用标准两步 SAM：

1. 第一次前向传播和反向传播，计算全局 L2 梯度范数。
2. 按 rho = 0.1 构造参数扰动。
3. 清除第一次梯度。
4. 使用同一个 Batch 的同一份增强后 Tensor 进行第二次前向传播和反向传播。
5. 恢复原参数。
6. 使用第二次梯度执行 SGD 更新。

不得误用逐 Tensor 独立归一化的 SAM 扰动。必须检查全局梯度范数、扰动向量范数、参数恢复和梯度更新均符合冻结实现。

不得在本轮加入 VPR、Guide、Fisher、Mask、EMA、Label Smoothing、MixUp 或其他研究变量。

### 3.4 数据增强与归一化

训练增强按 SSAM 对应的 `CIFAR10_cutout` 定义：

1. RandomCrop，大小 32×32，padding = 4。
2. RandomHorizontalFlip。
3. ToTensor。
4. Normalize。
5. Cutout，size = 16，应用概率 p = 0.5。

归一化采用 SSAM 对应版本的 CIFAR-10 统计量：

- Mean = `[125.3, 123.0, 113.9] / 255`
- Std = `[63.0, 62.1, 66.7] / 255`

Cutout 在归一化之后执行，与参考实现保持一致。

测试集只执行 ToTensor 和相同的 Normalize，不使用随机增强或 Cutout。

必须检查增强顺序、归一化数值、Cutout 的实际概率和实现细节。不得将另一份代码的不同 Normalize 数值或 Cutout 顺序直接混入。

### 3.5 学习率调度

- Scheduler：SSAM 对应的 CosineLRscheduler。
- `eta_min = 0`。
- `warmup_epoch = 0`。
- 遵循官方调度调用顺序和 Epoch 粒度。
- 训练前后记录每个 Epoch 的实际学习率。

不要仅因为 F-SAM 采用不同的调度器，就把 SSAM 配置静默替换为 F-SAM 的默认 StepLR 或逐 Batch Cosine。

## 四、BN 行为必须明确实现并审计

本轮选择 SSAM 官方训练循环的 BN 行为，以匹配主要目标基线。

SSAM 对应训练路径中，第一次和第二次 SAM 前向传播都在 `model.train()` 状态下执行，均更新 BN running statistics。

因此本轮：

- 第一次前向：更新 running mean、running variance 和相应计数器。
- 第二次扰动前向：同样更新 running mean、running variance 和相应计数器。
- 不得擅自调用 `disable_running_stats()`。
- 不得在第二次前向中将 `track_running_stats=False`。
- 不得在训练期间额外执行 BN 重估。
- 验证阶段必须使用 `model.eval()`。

每次训练迭代记录或抽样核验 BN buffers 的变化。对于正常参与训练的 BN 层，验证每个 SAM step 内的两次训练前向是否均产生了预期的 running-stat 更新。

同时，在报告中单独记录 F-SAM 的差异：其官方训练代码在第二次前向前将 BN momentum 设为 0，从而冻结 running mean/variance 更新，但 `num_batches_tracked` 的行为仍需独立核验。

**本轮不混用两种 BN 策略。** 若为了兼容当前 PyTorch 必须改变 BN 行为，先停下来报告差异，不得静默修改后继续。

## 五、Checkpoint 保存与模型选择

本轮遵循 SSAM 官方代码的 checkpoint 选择逻辑。

每个 Epoch 结束后：

1. 使用官方 CIFAR-10 的全部 10,000 张测试图片计算 Accuracy 和 Loss。
2. 如果当前 Accuracy 严格高于历史最佳 Accuracy，保存新的最佳 checkpoint。
3. Accuracy 并列时保留最早达到该 Accuracy 的 Epoch。
4. 同时持续保存最新 checkpoint。

输出：

- `best_test.pt`：200 Epoch 内最高测试 Accuracy 对应的完整 checkpoint。
- `last.pt`：第 200 Epoch 完成后的 checkpoint。
- `epoch_metrics.csv`：每个 Epoch 的训练和测试指标。
- `selection_metadata.json`：最佳 Epoch、最佳 Accuracy、该轮 Loss、最终指标、选择规则。
- `resolved_config.json`：最终实际生效的完整配置。
- `environment.json`：Python、PyTorch、torchvision、CUDA、GPU、驱动及相关环境信息。
- `source_manifest.json`：代码来源、Git commit、文件 SHA256、任何兼容性补丁。
- `SHA256SUMS`：所有关键交付文件的哈希。
- `FINAL_REPORT.md`：训练结果、配置、差异、审计和限制。

Checkpoint 必须保存足够的状态，以核验模型来源、Epoch、Seed、最佳 Accuracy 和配置；尽可能包含优化器、调度器与 RNG 状态。

不要将 `best_test.pt` 命名为 `best_val.pt`，也不要将它误标为第 200 Epoch 的最终模型。

重要：本轮使用官方测试集逐轮选模，因此测试集已经参与模型选择。报告必须明确这一点，最佳 Accuracy 属于文献协议对齐的选择指标，而不是独立、无偏的泛化估计。不得隐瞒测试集选模行为。

## 六、F-SAM 与 SSAM 配置差异审计

正式训练前，生成一份简洁的配置对照表，至少比较：

- ResNet18 的具体结构。
- 数据集规模与划分。
- Batch size 和 `drop_last`。
- LR、Momentum、Weight decay。
- rho。
- Scheduler 及调用粒度。
- Cutout 与 Normalize。
- BN 第二次前向行为。
- Seed 与确定性设置。
- Best/Final checkpoint 保存规则。

SSAM 是本轮主要配置来源；F-SAM 仅作为辅助审计。

发现两者不同之处时，列出来源、实际代码证据及本轮采用的值。不得将两个仓库的默认值混合后宣称严格复现了其中任何一个仓库。

本轮不运行 F-SAM 算法，不运行 SSAM 稀疏算法。

## 七、执行步骤

按以下顺序执行：

### 阶段 A：预检

- 检查 Git、工作树、服务器、GPU、磁盘和数据。
- 核对 SSAM 参考配置。
- 创建独立实验目录。
- 生成冻结配置文件与来源清单。

### 阶段 B：实现和测试

- 实现独立、可审计的 Dense-SAM 训练入口。
- 不修改历史冻结源码。
- 完成 SAM 数学单测、全局梯度范数检查、参数恢复检查、BN 更新检查、Loss 有限性检查、checkpoint 选择规则测试。
- 完成真实数据的少量 Batch CUDA smoke test。
- 确认数据样本数、模型结构、参数量、优化器配置和学习率调度正确。

### 阶段 C：正式训练

上述检查全部通过后，直接在现有 RTX 5090 服务器上运行：

- Seed 1234。
- 200 Epoch。
- 78,000 optimizer steps。
- 不做早停。
- 不搜索超参数。
- 不自动启动额外 Seed。
- 不因中途 Accuracy 未达到 96.83% 而擅自更改配置或重新开始训练。

训练必须能够在 SSH 断开后继续运行，并记录 PID、日志位置、开始时间、心跳、最近完成的 Epoch、GPU 使用情况及异常信息。

不得在没有完成 200 Epoch 的情况下将部分 checkpoint 标记为正式完成。

### 阶段 D：独立审计和交付

训练完成后：

1. 核对 Epoch 数、step 数、每轮样本数和实际配置。
2. 重新加载 `best_test.pt` 和 `last.pt`，核验来源、schema、模型参数及 SHA256。
3. 独立计算最佳 Epoch 和最佳 Accuracy，检查与日志一致。
4. 检查 BN 行为及数值异常记录。
5. 检查原有 VPR、Guide、历史 checkpoint 和历史文件哈希未被修改。
6. 生成最终报告。

最终报告必须分别呈现：

| 指标 | Best checkpoint | Last checkpoint |
|---|---:|---:|
| Epoch | 实际值 | 200 |
| Test Accuracy | 实际值 | 实际值 |
| Test Loss | 实际值 | 实际值 |
| Train Loss | 实际值 | 实际值 |
| Checkpoint SHA256 | 实际值 | 实际值 |

同时报告与 96.83% 目标值的差距，但不得为接近目标而修改已经冻结的实验协议。

如果结果低于 96.83%，保留完整实验，先分析配置执行和数值证据，不自动重训或调参。

## 八、停止条件

遇到以下情况立即停止受影响的阶段并报告：

- 数据完整性或样本数量不符合要求。
- 模型结构与配置无法核验。
- SAM 参数扰动或恢复逻辑不正确。
- BN 行为与冻结方案不一致。
- 发生 NaN、Inf、梯度异常或训练状态损坏。
- 磁盘空间不足或 GPU 环境无法通过 smoke test。
- 必须修改冻结配置才能继续运行。

不得通过删除历史文件、修改研究方法、切换数据集或更换配置绕过门禁。

本任务授权范围仅限建立并运行这条 Dense-SAM 基线。完成交付后停止，不继续启动剪枝、VPR+Guide、Block-SAM 或其他正式实验。

## 九、最终交付

请在最终回复中给出：

1. 训练是否完成 200 Epoch。
2. Best Epoch、Best Test Accuracy、对应 Loss。
3. Epoch 200 的 Accuracy 和 Loss。
4. 实际 optimizer steps 与训练样本数。
5. 实际 BN 行为及 SSAM/F-SAM 差异。
6. `best_test.pt`、`last.pt` 和日志的真实路径。
7. Checkpoint SHA256。
8. 单测、CUDA smoke 和独立审计结果。
9. 是否可以将 `best_test.pt` 冻结为后续 VPR+Guide 的共同 Dense-SAM 起点。
10. 所有偏离冻结协议的事项。

不得把尚未执行的训练写成已完成，不得虚构准确率或文件路径。