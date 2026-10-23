# VPR-SAM 独立科研实验工程

V1 基线为 `758d628`。本轮针对性修复见 [审计整改报告](docs/AUDIT_FIXES.md)，
包含剪枝/恢复配对统计、两层公平性校验和 schema 2 产物来源合同。冻结的优化器公式未修改。
PR 1最后一轮设备一致性与绘图checkpoint修复见 [FINAL_AUDIT.md](docs/FINAL_AUDIT.md)。

本目录可单独复制、安装和部署。运行时仅导入 `vpr_sam` 和声明的第三方依赖，
不导入父目录历史 N+SAM 代码、不读取历史配置或 checkpoint。

**当前交付的是供审计的实验基础设施。没有启动正式训练、连接服务器或确定正式 N:M/超参数。**
正式配置中的 null 刻意阻止未批准的实验。所有正式训练必须从随机初始化开始。

## 目录

```text
configs/smoke/          六种方法 + ResNet18 的合成数据短程配置
configs/formal/         六种正式方法的待批准模板（不可直接运行）
configs/pruning/        显式 N:M 剪枝协议示例，1:3 仅用于 smoke
src/vpr_sam/
  models/              自包含 CIFAR ResNet18、smoke TinyNet
  data/                CIFAR-10 校验、划分、逐样本确定性增强
  optim/               SGD/SAM/VPR、全局几何与两步事务
  training/            随机初始化、断点恢复、checkpoint、恢复训练
  pruning/             通用 N:M magnitude mask 和硬剪枝
  evaluation/          任务指标、统一外部 sphere probe、输出变化
  analysis/            配对统计、汇总 CSV/JSON、论文图
tests/                 数学、数值、算法、恢复与完整流程
scripts/               smoke、微基准、协议展开、服务器安装/启动
docs/                  数学合同、协议、审计与部署说明
data/                  本工程独立数据目录，不随 Git 上传
runs/                  本工程新 checkpoint、日志、图表，不随 Git 上传
artifacts/             测试和环境审计产物，不随 Git 上传
```

## 本地安装与验证

Python 3.12。先安装匹配平台的 PyTorch/torchvision；服务器使用后述 CUDA 固定环境。
在本目录执行（Windows 将 `.venv/bin/python` 替换为 `.venv/Scripts/python`）：

```bash
python -m venv .venv
.venv/bin/python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements-research.txt
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python -m pytest -q --basetemp=artifacts/pytest-temp --junitxml=artifacts/pytest.xml
.venv/bin/python scripts/smoke_suite.py --run-id local_acceptance --include-resnet
.venv/bin/python scripts/benchmark_geometry.py --output artifacts/geometry_benchmark.json
```

`run-id` 必须是新目录。不会覆盖已有 run。smoke 全部使用显式合成数据，
不能将其准确率或方法排序当作研究结论。测试临时文件统一放在 artifacts 内。
测试过程中不下载 CIFAR-10。性能微基准不执行模型训练。

## 完整运行命令

以下示例先用 smoke 展示可运行接口；正式命令需要先补全并审核配置。
每个训练配置有独立 output，产物含 `config.json`、`data_manifest.json`、环境与源码 SHA256、
`steps.json`、`epochs.json`、`last.pt`、`best_validation.pt`、`final.pt` 和 `summary.json`。

```bash
# 随机初始化训练；不接受外部初始化 checkpoint
python -m vpr_sam train --config configs/smoke/vpr.json
# smoke 模拟中断；之后严格恢复本项目同配置 checkpoint
python -m vpr_sam train --config configs/smoke/sam.json --stop-after 2
python -m vpr_sam train --config configs/smoke/sam.json --resume runs/smoke/sam/last.pt
# 硬剪枝，N:M/scope/tail 来自显式协议
python -m vpr_sam prune --checkpoint runs/smoke/vpr/final.pt --protocol configs/pruning/example_1of3_smoke.json
# 固定 mask、共同预算的恢复；默认结果写入该剪枝产物的 recovery 子目录
python -m vpr_sam recover --checkpoint runs/smoke/vpr/pruning/n1_m3_all_magnitude_dense_final/pruned.pt
# 恢复中断后必须同时指定原 pruned checkpoint 和 recovery checkpoint
python -m vpr_sam recover --checkpoint runs/smoke/vpr/pruning/n1_m3_all_magnitude_dense_final/pruned.pt --resume runs/smoke/vpr/pruning/n1_m3_all_magnitude_dense_final/recovery/last.pt
# 显式评估；训练本身只用 validation，test 不能用于选超参数
python -m vpr_sam evaluate --checkpoint runs/smoke/vpr/final.pt --split validation
python -m vpr_sam evaluate --checkpoint runs/smoke/vpr/final.pt --split test
# 汇总同协议、同 seed 对照（至少先完成这些 run）
python -m vpr_sam summarize --runs runs/smoke/sgd runs/smoke/sam runs/smoke/vpr --output runs/smoke/analysis
python -m vpr_sam plot --summary runs/smoke/analysis/summary.json --output runs/smoke/analysis/figures
# 所有方法已完成显式 test 评估后，按同一选择规则汇总最终 test
python -m vpr_sam summarize --runs runs/smoke/sgd runs/smoke/sam runs/smoke/vpr --split test --checkpoint final --output runs/smoke/test_analysis
# 多 N:M 结果并存时显式选一个共同协议；所有方法/seed 必须具备对应结果
python -m vpr_sam summarize --runs runs/smoke/sgd runs/smoke/sam runs/smoke/vpr --pruning-protocol configs/pruning/example_1of3_smoke.json --checkpoint final --recovery-checkpoint final --output runs/smoke/paired_pruning
```

PNG 和 PDF 输出包括训练/验证、扰动长度、log-volume、半径分布、剪枝损伤、seed 统计、跨参数预算迁移及恢复结果。
新汇总在 `postprocessing.hard` 和 `postprocessing.recovery` 中输出各研究指标的 seed 级统计、
同 seed 差值和 bootstrap 区间，并输出 `postprocessing.csv` 及两组配对剪枝/恢复图。
同方法不同 seed 的 gamma、warm-up、精度等必须一致；各方法之间仅算法专有几何可以不同。
若有剪枝结果，要求完整匹配的 seed 集合；部分恢复或不一致的 N:M/scope/tail/规则会拒绝汇总。
`--split test` 要求所有被比较阶段都有绑定其 checkpoint 的显式 test 评估。
指标包含两次损失/梯度范数、投影归一化误差、`d*p_max`、最大几何步长、半径极值、
log-radius RMS、每参数张量 log-volume、加法舍入误差、每步耗时和训练峰值 CUDA 显存。

## 正式协议完成后

1. 复制 `configs/formal/vpr.json` 为 `configs/common_approved.json`。
2. 共同填写所有 null，并确认 seed 列表、N:M、层范围、预算、学习率、精度、gamma、warm-up、恢复及探针；device必须改为批准的显式cuda:N。
3. 用同一配置展开全部方法。`--seeds` 与 `--methods` 由批准后的协议填写；以下数字仅展示接口，不是正式选择。

```bash
python scripts/materialize_protocol.py --common configs/common_approved.json --seeds 0 1 2 --methods sgd sam vpr --output configs/approved
python -m vpr_sam validate --config configs/approved/vpr_seed0.json --allow-formal
python -m vpr_sam prepare-data --data-root data/cifar10
python scripts/preflight.py --config configs/approved/vpr_seed0.json --require-5090
# 只有确认之后才执行；从随机初始化开始
VPR_FORMAL_APPROVED=YES bash scripts/run_one.sh configs/approved/vpr_seed0.json
```

同一 CLI 的正式恢复另需 `--allow-formal`。项目不提供批量服务器任务提交器，
不自动 SSH、不内置服务器凭据，不会仅因安装或导入而启动训练。

## 重要文件

- [数学定义与数值合同](docs/MATHEMATICS.md)
- [公平性、剪枝和复现协议](docs/PROTOCOL.md)
- [部署与正式实验执行计划](docs/DEPLOYMENT.md)
- [工程审计与验证结果](docs/AUDIT.md)
- [本轮审计整改与验收状态](docs/AUDIT_FIXES.md)

本轮 schema 2 checkpoint 加入模型/配置/Mask 身份，剪枝、恢复、评估及汇总统一核对源文件身份。
旧 V1 产物保留，但不会被自动迁移或绕过校验；请用新代码生成新的 smoke 产物。
正式训练门禁要求完整填写 N:M 与恢复协议，采用全流程冻结，不能只批准稠密训练参数。

复现限于一致软件/设备及确定性设置。旧 checkpoint 完全不作为初始化；
断点恢复仅指本工程新生成的健康状态。checkpoint 为 pickle，只加载可信产物。
