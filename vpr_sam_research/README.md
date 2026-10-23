# VPR-SAM 独立科研实验工程

**Volume-Preserving Perturbation Redistribution SAM**

本工程围绕“标准 SAM 的球形局部扰动与 N:M 结构化删除之间的尺度、方向失配”开展研究。在保留 SAM 两次梯度、参数恢复和基础优化器更新的前提下，使用全参数空间等体积的动态椭球重新分配扰动半径。

完整的“传统问题 → 方法 → 目标 → 实现”介绍见 [仓库研究概览](../README.md)。本页说明独立工程的组成与使用方式，不展示实验数据，不将研究假设写成已证实的性能结论。

## 核心实现

令 $A_t=\operatorname{diag}(a_{1,t},\ldots,a_{d,t})$，从 $A_0=I$ 开始：

$$
\epsilon_t^*=\rho\frac{A_t^2g_{1,t}}{\lVert A_tg_{1,t}\rVert_2},\qquad
p_{i,t}=\frac{(\epsilon_{i,t}^*)^2}{\lVert\epsilon_t^*\rVert_2^2},
$$

$$
s_{i,t+1}=s_{i,t}+\gamma(1-dp_{i,t}),\qquad
s_i=\log a_i,\quad\gamma=\eta_a/d.
$$

全局 $\sum_i s_i=0$ 对应 $\prod_i a_i=1$：守恒整个扰动集合的体积，允许跨层半径迁移。这里不采用逐层或逐 N:M 块守恒，不把扰动重新归一化为固定每步 L2 长度，也不增加 EMA、半径裁剪或新损失。

训练顺序保持为：计算 $g_1$ → 椭球扰动 → 计算 $g_2$ → 恢复参数 → 基础优化器更新 → 成功步后提交下一步几何。固定扰动参数集合和全局维度；全零扰动跳过几何更新，非有限数值必须显式失败。

## 工程组成

```text
configs/               合成数据 smoke、方法模板与剪枝协议
src/vpr_sam/
  models/              CIFAR ResNet18 与测试模型
  data/                真实数据完整性校验、划分与确定性增强
  optim/               SGD、SAM、VPR 及必要消融的几何与两步流程
  training/            从随机初始化训练、checkpoint、恢复与续训
  pruning/             配置化 Magnitude N:M Mask 和硬剪枝
  evaluation/          任务指标、公共外部球形 probe、输出变化
  analysis/            配对统计、汇总与绘图
tests/                 数学、数值、状态管理和流程验证
scripts/               验证、部署与运行入口
docs/                  数学合同、复现协议和工程说明
data/                  独立数据，不上传 Git
runs/                  独立 checkpoint、日志及分析产物，不上传 Git
artifacts/             验证与交付材料，不上传 Git
```

本目录可独立安装与部署，运行时不依赖父目录历史工程的 Python 路径、配置或 checkpoint。正式比较从配对随机初始化开始；续训只恢复本工程同一运行的健康状态，不继承历史校准权重。N:M、剪枝范围、恢复预算和 checkpoint 选择均通过明确配置约束。

## 安装与本地验证

先创建虚拟环境并安装与平台匹配的 PyTorch/torchvision，再在本目录安装工程及依赖。Linux 示例：

```bash
python -m venv .venv
.venv/bin/python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements-research.txt
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python -m pytest -q --basetemp=artifacts/pytest-temp
.venv/bin/python scripts/smoke_suite.py --run-id local_acceptance --include-resnet
```

Windows 使用 `.venv/Scripts/python`。GPU 部署需安装匹配的 CUDA 版本，不能把 CPU 安装命令当作服务器环境配置。smoke 使用显式合成数据，每次使用新的 `run-id`，其结果仅用于工程验证。

## 全流程接口

以下只展示已有 smoke 配置对应的接口，具体剪枝输出路径由显式 N:M 协议决定：

```bash
python -m vpr_sam train --config configs/smoke/vpr.json
python -m vpr_sam prune --checkpoint runs/smoke/vpr/final.pt
python -m vpr_sam recover --checkpoint <本工程生成的-pruned.pt>
python -m vpr_sam evaluate --checkpoint <待评估-checkpoint> --split validation
python -m vpr_sam summarize --runs <完整配对运行目录> --checkpoint final --recovery-checkpoint final --output runs/paired_analysis
python -m vpr_sam plot --summary runs/paired_analysis/summary.json --output runs/paired_analysis/figures
```

正式实验使用完整、已批准并冻结的配置及相应显式执行许可。正式阶段已经进入多随机种子对照流程；本页的安装和 smoke 命令不会自动启动正式队列。不要在已有队列运行时重复提交任务，也不要为更新说明文档而推进服务器冻结 checkout。

## 最新协议与版本定位

最新审计实现与正式配置以 [vpr-sam-audit-fixes 分支](https://github.com/earthgrass/N-M-SAM/tree/vpr-sam-audit-fixes/vpr_sam_research) 为准：

- [冻结数学与数值合同](https://github.com/earthgrass/N-M-SAM/blob/vpr-sam-audit-fixes/vpr_sam_research/docs/MATHEMATICS.md)
- [已登记的正式研究协议](https://github.com/earthgrass/N-M-SAM/blob/vpr-sam-audit-fixes/vpr_sam_research/docs/MAIN200_REGISTERED_PROTOCOL.md)
- [设备与 checkpoint 一致性审计](https://github.com/earthgrass/N-M-SAM/blob/vpr-sam-audit-fixes/vpr_sam_research/docs/FINAL_AUDIT.md)

当前研究首先检验：在相同训练协议和 Magnitude N:M 剪枝条件下，VPR-SAM 是否产生更小的硬剪枝损伤。几何半径是否能进一步指导 Mask 选择属于后续研究，不属于当前方法已实现或已验证的贡献。

## 当前分支

本分支包含当前独立工程，入口为 [vpr_sam_research](vpr_sam_research/README.md)。正式队列使用冻结提交 `d4c58d66b248b620e0e058b92e79100f173970a6`；后续 README 提交不改变该运行来源，也不推进服务器 checkout。[PR #1](https://github.com/earthgrass/N-M-SAM/pull/1) 保持未合并。
