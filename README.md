# N:M + SAM — 200 epoch GPU 初版待完善版本

本分支只归档六组 CIFAR-10 / ResNet18-CIFAR / seed 0 / 200 epoch GPU 实验对应的代码、正式配置、回归测试和轻量结果。旧的 2 epoch Phase-1 代码入口、结果、图表和文档已从当前文件树移除；原提交仍在 Git 历史中。

训练协议：batch size 128，SGD momentum 0.9，weight decay 0.0005，初始学习率 0.1，CosineAnnealingLR(T_max=200)，FP32，无 warmup；SAM rho 0.05。稀疏方法为严格完整块 2:4、STE dense master weights，每 20 step 更新 mask。`sparse_scope="phase1"` 是正式实验实际使用的层选择名称（排除首层与 1×1 卷积），不是短期实验入口。

| 实验 | 正式入口 | Final test accuracy |
|---|---|---|
| Dense-SGD | `experiments/dense_sgd_200epoch/nm_sam_cifar10/strong_baseline.py` | 95.85% |
| Dense-SAM | `nm_sam_cifar10/strong_sam.py` | 96.47% |
| 2:4-SGD-Magnitude | `nm_sam_cifar10/strong_sparse_sgd.py` | 96.07% |
| 2:4-SAM-Magnitude | `nm_sam_cifar10/strong_sam.py` | 96.40% |
| 2:4-SAM-Taylor-g1 | `nm_sam_cifar10/strong_sam.py` | 91.96% |
| 2:4-SAM-Taylor-g2 | `nm_sam_cifar10/strong_sam.py` | 94.87% |

六组结果均包含 200 行 epoch 指标，最后一行与 final_metrics.json 一致，global_step=78200。`results/200epoch/` 保存实际配置、CSV、JSON、日志及相关图表。checkpoint 和原始 CIFAR-10 数据不随 Git 归档。

## 来源与范围

Dense-SGD 保留服务器归档中的独立源码和依赖，放在单独的 `experiments/dense_sgd_200epoch/` 中；其数据增强依赖与后续实现存在工程差异，因此未替换为后续源码。另五组代码来自现存 GPU 源码快照。四组 SAM 的关键源码与结果随附 SHA256 清单匹配。稀疏 SGD 的结果包包含 experiment_metadata.json 和 SOURCE_SHA256SUMS.txt；入口、正式配置及归档中保留的依赖均与服务器哈希记录一致（部分纯文本按 LF 换行核对）。完整服务器清单列出的非 200 epoch 文件不随当前归档保存。逐文件来源与 SHA256 见 `SOURCE_PROVENANCE.json`。

保留原有算法实现，没有为归档修改训练源码。此版本仍待完善：SAM step 异常中断时缺少完整参数恢复保障；数据加载失败时的 FakeData fallback 需要在正式实验工作流中额外核实数据集身份。历史 200 epoch 结果不等于本次重跑结果。

## 环境与验证

历史服务器：Python 3.12.3、PyTorch 2.8.0+cu128、torchvision 0.23.0+cu128、RTX 5090。`requirements.txt` 仅声明依赖，并非完整环境锁文件。本次隔离验证在 CPU 环境执行：共享实现 54 个测试通过，独立 Dense-SGD 历史源码 3 个测试通过，三个正式入口的 CLI 导入检查通过。不重跑 200 epoch GPU 训练。

从仓库根目录验证五组共享实现：

```bash
python -m pytest nm_sam_cifar10/tests -q
```

Dense-SGD 的历史依赖需在独立目录验证，避免同名 Python 包互相覆盖：

```bash
cd experiments/dense_sgd_200epoch
python -m pytest nm_sam_cifar10/tests -q
```

## 正式训练命令

以下命令会启动完整 GPU 训练。本次归档核验没有执行这些命令。结果写入新的 `runs/` 目录，避免覆盖历史产物。先准备真实 CIFAR-10 数据和匹配的 CUDA 环境。

从仓库根目录启动五组共享实现：

```bash
python -m nm_sam_cifar10.strong_sam --config nm_sam_cifar10/configs/strong_dense_sam_seed0.yaml --allow-full-training --output-dir-override ./runs/dense_sam_seed0
python -m nm_sam_cifar10.strong_sparse_sgd --config nm_sam_cifar10/configs/strong_2to4_sgd_magnitude_seed0.yaml --allow-full-training --output-dir-override ./runs/nm24_sgd_magnitude_seed0
python -m nm_sam_cifar10.strong_sam --config nm_sam_cifar10/configs/strong_2to4_sam_magnitude_seed0.yaml --allow-full-training --output-dir-override ./runs/nm24_sam_magnitude_seed0
python -m nm_sam_cifar10.strong_sam --config nm_sam_cifar10/configs/strong_2to4_sam_taylor_g1_seed0.yaml --allow-full-training --output-dir-override ./runs/nm24_sam_taylor_g1_seed0
python -m nm_sam_cifar10.strong_sam --config nm_sam_cifar10/configs/strong_2to4_sam_taylor_g2_seed0.yaml --allow-full-training --output-dir-override ./runs/nm24_sam_taylor_g2_seed0
```

从独立历史源码目录启动 Dense-SGD（数据路径按仓库根目录的标准位置计算）：

```bash
cd experiments/dense_sgd_200epoch
python -m nm_sam_cifar10.strong_baseline --mode all --data-root ../../nm_sam_cifar10/data/cifar10 --output-dir ../../runs/dense_sgd_seed0
```
