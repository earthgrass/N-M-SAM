# 工程审计报告

本文件保留 `758d628` 的 V1 验证记录（40项测试）。当前整改与新验证记录请见 [AUDIT_FIXES.md](AUDIT_FIXES.md)。

审计日期：2026-10-23（Asia/Shanghai）。本报告描述本次本地验证，不将历史实验记为新结果。

## 交付范围

本工程为独立 src-layout Python 项目，具有完整模型、数据、优化器、训练、断点恢复、
N:M 硬剪枝、统一恢复、评估、配对统计、绘图、配置及部署脚本。
历史工程仅用于只读了解已验证的 ResNet18/BN 协议；新项目内部没有历史 Python 模块导入。
不使用历史 checkpoint 或数据路径，不改动历史代码。所有新运行输出位于本项目内。
原论文文档未复制进项目，也不构成运行时依赖。

## 数学审查

- 全参数绝对坐标空间，固定 d，跨层全局归约；不逐层或逐块守恒。
- 正确使用 A^2 g / ||Ag||，而不是 Ag / ||Ag|| 或最终统一 L2 归一化。
- 反馈来自同一已构造的 FP32 epsilon 缓冲，平方占比全局归一化。
- s_next=s+gamma(1-dp)，eta_a=d gamma，无 EMA、裁剪、附加损失或更新后体积回正。
- 固定点、全局守恒和 Jacobian 均有独立数学参考测试。
- 固定梯度的局部线性稳定条件不被解释成随机训练的全局保证。
- fixed_l2、positive 和 static 明确标记为消融，主方法 vpr 保持冻结定义。

## 已执行验证

实际机器环境由 [机器可读结果](validation/results.json) 记录：Windows、Python3.12.10、
PyTorch2.13.0 CPU、torchvision0.28.0；与计划服务器环境2.8.0+cu128不同。

1. **40 项 pytest 测试全部通过**；无 skip、无 error。原始测试项与结果见 [JUnit](validation/pytest.xml)。
2. 覆盖固定点闭式解、有限差分 Jacobian、零和特征值、局部稳定/不稳定、边界反例、
   两种步长表达、椭球边界、全局跨张量反馈、零/缺失梯度、非有限值、FP32/FP64 漂移与能量集中。
3. 手工参考外层更新与 SGD/SAM/VPR 对照一致；两次前向复用输入、BN 只计一次、基础优化器只执行一次。
4. 第二次前向异常、第二次梯度 NaN 均恢复临时参数和 BN buffers，不提交几何；失败对象禁止保存/继续。
5. 稠密 VPR 中途恢复与连续训练逐项一致：模型、几何、SGD 状态、scheduler、RNG、epoch 指标和数据顺序。
6. 固定 Mask 的 SAM 恢复训练也通过中断/连续轨迹一致性测试。workers=1 与 workers=0 的数据/增强一致；
   完整轨迹一致性当前在 workers=0 上验证，GPU/多 worker 完整轨迹仍需服务器补验。
7. N:M 测试使用1:3、3:8和1:2，检查尾块、ties 和无权重回生。没有将2:4作为正式默认。
8. 真实数据失败显式报错，测试验证不会进入 FakeData fallback。
9. 六种方法各完成6个训练步、3个恢复步、硬剪枝和合成 test 评估，并完成统计与5组 PNG/PDF 图。
10. CIFAR ResNet18 完成2个 FP32 VPR-SAM 训练步，使用8个合成样本（其中4个validation）。
11. 通过 wheel 构建；在另一个目录、另一个非 editable 安装环境中，使用 Python `-I` 运行六方法完整 smoke。
    导入位置确认来自 wheel 的 site-packages，不是历史工程或原 src 路径。
12. AST 扫描源模块，没有历史工程模块导入。Git 变更限定于 `vpr_sam_research/`。

smoke 完成记录不代表任何方法优于另一个方法，不报告其准确率作为研究发现。
完整 checkpoint 和图形留在本地 runs/artifacts，Git 仅保存源码及轻量验证证据。

## 精度与性能评估

[几何微基准](validation/geometry_benchmark.json) 仅使用固定预生成梯度，没有模型训练。
CPU、单线程、10次更新：d=100000 时 FP64 状态800000字节、FP32状态400000字节；
FP64 log-volume 约 -6.0e-16，FP32约3.9e-9。
这些小规模短序列结果不足以选择正式精度；时间受机器负载影响，不推断 RTX5090 表现。
FP64 临时张量、log/exp 与同步归约可能显著增加时间和显存，必须实测。

## 可复现性与产物

每次运行保存配置、数据split索引/指纹、源文件SHA256、软件/硬件、初始化hash和数据顺序hash。
恢复严格检查配置、源代码、数据、参数映射、状态精度、Python、平台和运行环境，不允许无声转换。
checkpoint 原子替换，只在健康完整步后保存；没有自动加载已有文件的隐式行为。
`train` 不提供外部初始化checkpoint参数；正式训练只能随机初始化或恢复本项目同一run。
只读评估与剪枝产物记录父checkpoint SHA256。
结果汇总拒绝重复method/seed和不公平协议；test汇总要求显式test评估并核对checkpoint hash，
统一选择 final 或 best_validation，剪枝统计不混用不同来源checkpoint。

## 未执行与已知限制

- **没有连接服务器、运行CUDA、下载真实CIFAR-10或开始正式长程训练。**
- 服务器目标软件版本与本地验证版本不同，必须重跑测试、CUDA preflight、GPU smoke和GPU断点恢复。
- 正式N:M、scope、tail、数据split、seed数量、超参数、精度、恢复预算、探针与选择规则均待批准。
  formal模板故意保留null，不是可直接启动的正式配置。
- 逻辑N:M分块不代表使用稀疏硬件kernel；不承诺2:4硬件吞吐收益。
- 第一版无BN校准、无几何Mask评分、无Taylor剪枝、新Predictor或Compensation；统一magnitude用于隔离优化器作用。
- 遇到半径极端增长或守恒超限直接失败，不静默稳定化。局部稳定区间不能消除此风险。
- 基础优化器内部部分失败后不宣称状态自动回滚；终止并从上一个健康checkpoint恢复。
- 精确复现不跨不同设备/驱动/依赖版本保证。checkpoint使用pickle，只接受可信文件。
- 第一步先审计基础设施，服务器阶段再按 [部署计划](DEPLOYMENT.md) 验收和决定执行顺序。

## 建议 GPT 审计顺序

1. `optim/geometry.py` 对照数学文档，检查全局维度、指数稳定性与原始反馈。
2. `optim/step.py` 检查两次梯度、BN、恢复顺序、失败边界与固定mask。
3. `data/`、`training/engine.py` 检查随机流、随机初始化、断点状态和数据泄漏。
4. `pruning/`、`evaluation/` 检查分块语义、立即硬剪枝损伤与统一外部探针。
5. `config.py`、`analysis/report.py` 检查公平性校验、统计单位和最终test证据。
6. 检查40项测试与独立wheel运行证据，再讨论尚未批准的正式协议。
