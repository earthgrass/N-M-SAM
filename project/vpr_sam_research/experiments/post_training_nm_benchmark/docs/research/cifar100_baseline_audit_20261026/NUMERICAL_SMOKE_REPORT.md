# CPU 数值 Smoke 报告

## 状态

| 验收项 | 状态 | 证据 |
|---|---|---|
| 静态来源与工程审计 | `STATIC_AUDIT_PASS` | source commits、文件 SHA、工程报告 |
| Conv2d 数学映射 | `MATHEMATICAL_MAPPING_PASS` | 3 类卷积前向/反向 |
| N:M mask 与坐标 | `MASK_VALIDATION_PASS` | 13 个 pytest；tail/组卷积/逆置换 |
| Wanda adapter | `NUMERICAL_SMOKE_PASS` | 官方 Linear 同输入 score/mask 对照 |
| SparseGPT adapter | `NUMERICAL_SMOKE_PASS` | 官方 Linear 同输入 mask/补偿对照 |
| ROSE pinned N:M | `OBSERVED_PINNED_SOURCE_FAILURE` | `UnboundLocalError` traceback |
| ROSE 论文语义修复 | `NUMERICAL_SMOKE_PASS` | 排序、逆排序、2:4、有限值 |
| CIFAR-100 accuracy | `FULL_EXPERIMENT_NOT_RUN` | 数据/checkpoint 不存在；本轮禁止正式训练 |

## 环境与命令

- Windows 11；Python 3.12.10；PyTorch 2.13.0+cpu；CUDA unavailable。
- 随机种子：20261026；合成输入，无 CIFAR 图像、无历史 checkpoint。
- 核心命令：

```powershell
python scripts/run_cifar100_baseline_smoke.py `
  --wanda-source <pinned-Wanda> `
  --sparsegpt-source <pinned-SparseGPT> `
  --rose-source artifacts/upstream/ROSE `
  --output docs/research/cifar100_baseline_audit_20261026/TEST_EVIDENCE/smoke_results_v2.json

python -m pytest -q tests/test_cifar100_baseline_adapters.py `
  --basetemp=artifacts/pytest-cifar100-adapters-v2 `
  --junitxml=docs/research/cifar100_baseline_audit_20261026/TEST_EVIDENCE/adapter_tests.xml
```

两条命令退出码均为 0；首轮 adapter pytest 为 `13 passed in 8.22s`。加入 100 类 ResNet18 单层 Wanda 技术路径后，最终为 `14 passed in 7.36s`，见 `adapter_tests_v2.xml`。

项目规定范围的完整回归命令 `python -m pytest -q tests` 首次得到 99 passed、5 failed，五项均因环境漏装 `requirements-cpu.txt` 已锁定的 `joblib==1.5.2`；失败 XML 和环境保存为 `project_suite.xml` / `project_suite.environment.json`。安装该锁定依赖、不修改测试后重跑得到 104 passed；加入最终 ResNet18 单层用例后再跑得到 **105 passed in 22.20s、exit 0**，最终证据为 `project_suite_v3.xml` / `project_suite_v3.environment.json`。曾误运行无 `tests` 范围的 pytest，收集了 ignored 第三方 clone 自带测试并被人工中止；该次不计入本工程验收。

## 关键数值

- Conv2d：三种配置 forward max error 均 0；input-grad max `1.11e-16`；weight-grad max `1.07e-14`。
- Wanda 官方 Linear vs adapter：mask 完全一致；score max difference `1.08e-7`（官方统计 buffer 为 float32、适配累计为 float64）；幸存权重变化 0。
- SparseGPT 官方 Linear vs adapter：mask 完全一致；补偿权重 max difference `1.25e-6`。CPU 诊断仅临时 neutralize 官方 `torch.cuda.synchronize()` timing fence，finally 恢复，官方源码字节未改。
- SparseGPT Conv：H 形状与 `2XXᵀ/T` 独立计算完全对齐；删除位为零、dense tail 保留、无 NaN/Inf。
- ROSE pinned source：构造 columnar 2:4 层后触发 `reorder_column` 未赋值的 `UnboundLocalError`。失败不是容差问题，完整 traceback 已保存。
- ROSE controlled repair：置换为双射、inverse 精确恢复坐标、16 权重中 8 个删除、补偿后有限。

## 证据清单

- `TEST_EVIDENCE/smoke_results_v2.json`：首选原始记录、环境、官方文件 SHA256、失败 traceback。
- `TEST_EVIDENCE/smoke_results.json`：第一次成功运行记录，因后续增加完整 traceback 而保留、不覆盖。
- `TEST_EVIDENCE/adapter_tests.xml`：JUnit 原始测试报告。
- `TEST_EVIDENCE/adapter_tests_v2.xml`：加入 CIFAR-100 头 ResNet18 单层技术路径后的最终 adapter 报告。
- `TEST_EVIDENCE/project_suite.xml`：缺 joblib 时的 5 项既有 SNOWS 环境失败；没有隐藏或覆盖。
- `TEST_EVIDENCE/project_suite_v2.xml`：补齐锁定依赖后的 104 项完整项目回归 PASS。
- `TEST_EVIDENCE/project_suite_v3.xml`：最终 105 项完整项目回归 PASS。
- `TEST_EVIDENCE/SHA256SUMS.json`：交付证据与实现文件哈希。

这些测试只证明实现路径和坐标语义，不证明 CIFAR-100 精度、SOTA 或硬件加速。
