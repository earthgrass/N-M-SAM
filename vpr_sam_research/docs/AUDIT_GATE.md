# 实验审批门禁

|门禁|状态|证据|
|---|---|---|
|源码版本与分支一致|PASS|validation/server_acceptance/audit_gate.json；相关专题报告|
|数学核心未改动|PASS|validation/server_acceptance/audit_gate.json；相关专题报告|
|5090硬件验收|PASS|validation/server_acceptance/audit_gate.json；相关专题报告|
|CUDA Smoke|PASS|validation/server_acceptance/audit_gate.json；相关专题报告|
|真实CIFAR-10|PASS|validation/server_acceptance/audit_gate.json；相关专题报告|
|精确断点恢复|PASS|validation/server_acceptance/audit_gate.json；相关专题报告|
|固定Gamma短程校准|PASS|validation/server_acceptance/audit_gate.json；相关专题报告|
|完整协议冻结|PENDING|validation/server_acceptance/audit_gate.json；相关专题报告|
|200 epoch正式训练|NOT STARTED|validation/server_acceptance/audit_gate.json；相关专题报告|

测量代码 `bf4f123e01a9b0819376a4def7d2e82e5c56975d`，父审计基线 `ee57bc68b9a2f5f71dffb067f67416a9461f8db1`。

- geometry.py: SHA256 `8e0e6d42bcf34a7a8c3b79dc4b76081e9d8c0d6313b183ce0f59109bd39058b5`
- step.py: SHA256 `3c022f2815d577ef3d94ffeed40cb93c06e2359a2734c857a06f1a0c1f738ac2`

短程固定gamma提名：5.482145661475941e-08；仅FP64有限预算证据。FP32门禁失败不以放宽容差处理。

可以提交GPT审查本轮证据与待批准协议；这不构成正式实验执行许可。未批准项：验证划分/seed列表、SGD/LR/增强、rho/gamma/warm-up/精度、N:M/scope/tail、恢复预算/优化器、checkpoint选择、probe、日志频率与GPU资源。正式JSON中的null保持阻断。
