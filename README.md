# N-M-SAM — N:M Structured Sparsity × Sharpness-Aware Minimization (Initial Research Prototype)

> **Branch:** `初版待完善版本`
> **Status:** 初始研究原型快照 (first-stage research prototype). 已知存在后续重构与正式训练,这里只冻结初版代码与对应轻量实验结果,作为可追溯、可继续重构的基线。
>
> **重要:** 本分支刻意只放初版 Python 代码与初版结果摘要;不包含任何 GPU 长训练 checkpoint、Phase-2+ 正式结果、forensic 审计材料或 archives 打包快照。

---

## 1. 研究目标

在所有合法的 N:M mask 中,挑选使 SAM 鲁棒损失最小的那一个 ——
扰动受限于该 mask 允许的子空间。已实现的求解器是 **first-order SAM-Taylor 近似**
`|W * g₂|`,其中 `g₂` 是 SAM 的第二次梯度。**不是 Hessian/二阶方法。**

本仓库是研究初版,目的是呈现定性现象,而不是追逐顶尖准确率,也不是实现真正的
NVIDIA Sparse Tensor Core kernel。所有 2:4 sparsity 在此都是 PyTorch mask + straight-through
estimator (STE) 的 **逻辑/模拟** 实现。

## 2. Phase 1 研究问题

1. 在 CIFAR-10 上,2:4 weight sparsity 会不会显著伤害 CNN accuracy?
2. SAM 能否在 2:4-sparse 模型上提升性能 / sharpness?
3. 同样 2:4 sparsity 等级下,SAM-aware mask (基于 `|w * g_SAM|` 选择) 是否优于 magnitude mask?

## 3. 五个方法 (Phase 1)

| Method         | Sparsity | Optimizer | Mask-score              |
| -------------- | -------- | --------- | ----------------------- |
| Dense-SGD      | none     | SGD       | n/a                     |
| Dense-SAM      | none     | SAM       | n/a                     |
| 2:4-SGD        | 2:4      | SGD       | magnitude               |
| 2:4-SAM        | 2:4      | SAM       | magnitude               |
| 2:4-SAM-aware  | 2:4      | SAM       | `|w * g_SAM|` (SAM 梯度上的 Taylor) |

五个方法共用同一份 **模型、数据增强、batch size、epoch 数、基础 SGD 超参数、随机种子**。
唯一改变的是 **optimizer + sparsity 策略**。

## 4. 项目结构 (本分支)

```
nm_sam_cifar10/
├── configs/           # 默认运行配置 (JSON/YAML)
├── data/              # CIFAR-10 loader (torchvision + 自定义 fallback);不含 raw 数据
├── models/            # ResNet18-CIFAR、SimpleCNN
├── methods/           # 五个方法的声明式组合
├── optim/             # SAM + optimizer 工厂
├── sparsity/          # canonical mask、layer scope、STE、perturbation geometry
├── trainers/          # 共享的 SGD step 与 SAM step
├── metrics/           # task、landscape、sparsity、efficiency metrics
├── scripts/           # shell 启动脚本
├── results/
│   └── phase1_fixed/  # 初版 phase-1 实际跑出来的 CSV/JSON/log(不含 .pt checkpoint)
├── plots_phase1/      # 初版 PNG 图表 (loss / acc / sharpness)
├── tests/             # 单元 + 运行时 sanity 测试
├── docs/              # 初版架构/审计报告 (不含 docs/audits/)
├── train.py           # 主入口
├── evaluate.py        # 独立评估器
├── plot.py            # CSV → plots + summary
├── README.md
├── EXPERIMENT_REPORT.md   # 初版 Phase-1 报告
├── METHOD_ARCHITECTURE_REPORT.md
└── .gitignore
```

## 5. 运行复现 (CPU smoke)

```bash
# 默认 dense_sgd, 5 epoch, ResNet18-CIFAR, CIFAR-10
python train.py --epochs 5 --batch-size 128 --lr 0.1

# 跑 5 个方法对比
python train.py --method dense_sgd  --epochs 5
python train.py --method dense_sam  --epochs 5
python train.py --method nm_sgd     --epochs 5 --n 2 --m 4
python train.py --method nm_sam     --epochs 5 --n 2 --m 4
python train.py --method nm_sam_aware --epochs 5 --n 2 --m 4
```

数据集会自动下载/复用 `./data/cifar10/cifar-10-batches-py`(本仓库不打包 raw 数据)。

## 6. 已验证 (本分支快照内的内容)

* 五个方法在 ResNet18-CIFAR / CIFAR-10 / 单 seed / 5 epoch 上的 **CSV/JSON/log 结果已就位**:
  `results/phase1_fixed/{dense_sgd,dense_sam,nm_sgd,nm_sam,nm_sam_aware}_seed0/`
* 配套 PNG 图表 (epoch_time / sharpness / test_acc / test_loss / train_loss):
  `plots_phase1/`
* 跨方法汇总: `results/phase1_fixed/summary_phase1_fixed.csv`
* 跨方法稳定性扫描: `results/phase1_fixed/sam_stability_grid_fixed.csv`
  + 单元 + runtime sanity tests (`tests/`)
* 单元测试: `pytest tests/`(关键 mask / STE / SAM 路径)

## 7. 已知问题 / 限制 (本初版已承认)

* 准确率水平只用于呈现定性现象,不作为 SOTA 声明。
* 2:4 mask 是逻辑/模拟实现,没有真实调用 NVIDIA Sparse Tensor Core;
  报告的 epoch time 只是 *algorithm cost*。
* 默认 `phase1` scope 保留 first conv 与 1x1/downsample conv 为 dense,仅对 FC + 主干做
  2:4;其它 layer-scope 选项是显式 CLI,不是研究结论。
* SAM-aware mask 是 first-order 近似,未做 Hessian / 二阶项。
* 长训练 / 多 seed / GPU 上的 full Phase-2+ 结果 **不属于本分支**,需另起分支/工作流。

## 8. 后续重构建议路径

基于 `初版待完善版本` 分支进行正式重构时,推荐:

1. 新建 `phase2/<feature>` 分支(例如 `phase2/strong-baseline`、
   `phase2/sam-aware-taylor-g2`)。
2. 在新分支引入重构脚本与正式训练结果(`results/<新 run>/`),
   不要在初版分支追加。
3. 需要回溯初版行为时,`git checkout 初版待完善版本` 后按 `train.py` 默认
   `configs/default.json` 复跑即可(只跑 5 epoch / seed 0)。
4. 正式结果上 PR 之前,核对 `configs/optim/sparsity` 的关键 hash 与
   `EXPERIMENT_REPORT.md` 中记录一致,避免与初版基线混淆。

## 9. 数据 / 模型权重 / 凭据

* **不** 打包 `data/cifar10/cifar-10-batches-py/`。
* **不** 提交 `*.pt` / `*.pth` / `*.safetensors` 等模型权重。
* **不** 提交任何 token、SSH key、`.env` 或凭据文件。`.gitignore` 已显式忽略
  `*.env`、`*.pem`、`*.key`、`.audit_cache` 等敏感或临时文件。
* 历史 archives (`*.tar.gz`)、forensic 审计与 phase-2+ 结果被显式忽略,见
  `.gitignore`。

## 10. 硬件说明

* 本初版默认在 CPU 上跑 5 epoch 已能复现 Phase-1 现象。
* GPU 上的 full training **不属于本分支**;`GPU_*` / `*STRONG_GPU*` /
  `*SERVER_PHASE2A*` 等报告被 `.gitignore` 排除。
