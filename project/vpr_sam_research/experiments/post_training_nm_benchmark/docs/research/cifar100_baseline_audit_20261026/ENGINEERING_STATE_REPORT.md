# 工程状态报告（审计起点：2026-10-26）

## 仓库与保护状态

- 实际 Git 根：`C:/Users/earthgrass73/Desktop/NM稀疏/APBR-post-training-benchmark`。
- 当前分支：`research/post-training-nm-benchmark`；审计起点 HEAD：`9ff061a009e19e619a8a8ae3f5b08b494c83b8d0`。
- 开始本任务时 `git status --short --untracked-files=all` 无输出，即工作区干净。未切换 commit、未 reset/clean、未覆盖历史资产。
- 本轮新增内容隔离在 `baseline_adapters/`、本目录、一个 smoke 脚本和一个测试文件；`benchmark/obc.py`、APBR-SAM 核心及历史产物均未修改。
- 项目级 `AGENTS.md` 明示：历史资产只读、正式 N:M/超参数需确认、不得未经批准启动长程任务。本轮只执行 CPU 合成数值测试。

## 当前工程与历史资产

- 当前模型 `benchmark/model.py` 是 CIFAR BasicBlock ResNet18，分类头为 10 类；它不是可直接用于 CIFAR-100 的已验收模型。把 `num_classes` 改为 100 是必要接口适配，但尚未形成 CIFAR-100 checkpoint。
- `configs/existing_inputs.json` 记录 15 个历史输入：3 seeds ×（Dense-SAM、APBR dense-with-geometry、3 种恢复产物），均有 SHA256、producer commit、epoch/step 和相对路径。它们属于 CIFAR-10 历史资产。
- OBC 正式三种子交付位于 `artifacts/obc_seeds012_final_20261025/`。`FINAL_RESULTS.csv` 的 Final accuracy 是 96.38/96.33/96.38%，报告汇总 96.3633±0.0289 个百分点；这是 CIFAR-10 OBC 结果，不是本轮复现结果，也不是 CIFAR-100 证据。
- `docs/research/evidence/historical_statistics.json` 核对出 Dense-SAM recovery Final 96.2133±0.0961%、APBR-SAM (`vpr_guided`) 96.3900±0.0854%；这与任务说明中的四舍五入结果一致。历史 Test 被用于 checkpoint 选择，报告已披露选择偏差。
- 当前正式 OBC 配置冻结为 2:4、`oihw_output_row_contiguous_v1`、首尾层排除、BN affine 排除、不完整尾部保持 dense；tie 规则区分 OBC 原生 argmin 与 magnitude stable lower index。

## 数据身份

- 工作区只有 `data/cifar10/cifar-10-python.tar.gz`，SHA256 `7ccff612802c13b600173bb0b06c7f86514ca14e8efcdce7ff176f100355b6c9`。
- `data/cifar100` 不存在；因此本轮不能核验 CIFAR-100 archive、train/test 样本身份或生成正式 CIFAR-100 结果。
- CIFAR-10 当前归一化、增强、校准 ID 和 BN ID 已有冻结记录；它们不得自动继承为 CIFAR-100 协议。

## 既有测试状态

- 任务开始前最近保存的主测试 XML 日期为 2026-10-25，包括 `cpu_tests.xml`、`acceptance_tests.xml`、`final_gate_tests.xml` 和 `cuda_closed_gate_tests.xml`。
- 本轮最终新增测试见 `TEST_EVIDENCE/adapter_tests_v2.xml`：14 passed（含 100 类 ResNet18 单层 Wanda 路径）；补齐 requirements 中已锁定但环境漏装的 joblib 后，最终项目 `tests/` 完整回归为 105 passed（`project_suite_v3.xml`）。未执行 GPU 或正式分类准确率测试。

## 起点结论

当前仓库适合作为隔离适配宿主，但尚不具备 CIFAR-100 正式实验的三项关键输入：已核验 CIFAR-100 数据、100 类 ResNet18 配置、匹配 seed 的 Dense-SAM/APBR-SAM checkpoint。故状态为 `STATIC_AUDIT_PASS`，正式 CIFAR-100 状态必须保持 `FULL_EXPERIMENT_NOT_RUN`。
