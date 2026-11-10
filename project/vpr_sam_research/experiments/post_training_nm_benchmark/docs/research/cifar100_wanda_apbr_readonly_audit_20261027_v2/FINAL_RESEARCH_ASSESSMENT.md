# 最终科研判断

## 六个问题的答案

1. **Q1 — `VERIFIED`**：Wanda 官方核心评分、逐行 N:M 选择与不更新幸存权重在 CNN adapter 中保留；Conv2d patch 映射是必要接口适配。名称应写 `Wanda-CNN-unified2:4`，不是“官方 ResNet18 Wanda”。
2. **Q2 — `VERIFIED`**：Seed 0 的 19 层 Wanda Mask/checkpoint 通过结构、零值、幸存权重、父 SHA 和 Mask SHA 审计。运行时 score 张量未保存这一项仍是 `UNVERIFIED`，不影响 checkpoint 的结构验收。
3. **Q3 — `SUPPORTED_HYPOTHESIS`**：APBR Raw 高 0.68 pp，且相对父模型的 accuracy drop 少 0.84 pp；但父模型不同、BN-only 未评估，而且 APBR Raw cross-entropy 比 Wanda 高。因此不能把 0.68 pp 全部归因于几何评分。
4. **Q4 — `UNVERIFIED`（测试轨迹）/ `VERIFIED`（最终点）**：Wanda Recovery-20 Final 高 0.21 pp，起点和协议正确；日志只有训练轨迹，没有逐 epoch Test，无法观察反超发生时间或将其归因于优化地形。
5. **Q5 — `VERIFIED_WITH_SCOPE_LIMITS`**：Seed 0 全模型 calibration+pruning：Wanda 1.159s、SparseGPT 9.273s、ROSE 9.889s。纯二阶全模型峰值显存和逐层细分计时未单独记录。
6. **Q6 — `SUPPORTED_HYPOTHESIS_SEED0_ONLY`**：可以说“Seed 0 中 APBR 与更小的即时 accuracy 损伤相关”；不能说“APBR 已被证明更擅长减少即时剪枝损伤”。需等待 Seeds 1、2 的同 seed 配对结果。

## APBR 的现有优势与不足

优势：Seed 0 Raw accuracy 和相对父模型 accuracy drop 均优于 Wanda，Mask-only 语义清楚。不足：Recovery-20 Final 没有优于 Wanda；Raw cross-entropy 也不优；完整流程承担 APBR 几何训练与 20 epochs Recovery 的额外成本。

## 实现问题

本审计没有发现 Seed 0 Wanda/APBR 的父 checkpoint、mask、BN-only 隔离、Recovery 起点或固定 mask 协议错误。证据缺口不等于实现失败。

## 最低成本下一步

不修改当前三 Seed 计划。等待 Seed 1/2 正常完成后，只复用已经生成的 Dense/Raw/BN-only/Recovery checkpoint 与最终评估：

1. 计算 Wanda 与 APBR 的 Raw accuracy/loss 同 seed 配对差及 mean ± sample SD。
2. 如要检验 BN 假设，最小新增工作是对既有 BN-only checkpoint 做固定 Test 评估；需另行批准，且不得选 epoch/调参。
3. 不建议补做逐 epoch Test 曲线；它会扩大 Test 访问且不影响预注册 Final 结论。若研究优化轨迹，优先用现有 train loss，并明确它不是泛化轨迹。

当前正常三 Seed 队列无需修改、无需重启。
