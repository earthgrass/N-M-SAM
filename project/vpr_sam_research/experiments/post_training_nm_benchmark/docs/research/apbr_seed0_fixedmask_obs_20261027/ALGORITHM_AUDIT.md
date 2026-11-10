# Fixed-Mask OBS 算法审计

## 原生 SparseGPT 与本实验的差异

二者复用相同的卷积 `unfold`、近似 Hessian、相对阻尼、Cholesky/逆因子和逐列 OBS 误差传播结构。唯一方法性改变是 Mask 来源：原生 SparseGPT 在补偿过程中动态选择删除坐标；本实验在进入补偿前加载由 APBR 几何评分 `|w| exp(-s)` 生成的 bool keep-mask，逐列处理时直接使用外部删除坐标，禁止动态重选。

补偿从 APBR-SAM 稠密 Epoch 200 checkpoint 开始，而不是从已置零的历史 Raw Hard 权重开始。被删除坐标严格写零，幸存坐标允许接收 OBS 更新；补偿后不会再通过乘 Mask 冒充固定 Mask OBS。

## Mask 身份和 2:4 约束

- 19 个 eligible Conv2d 的 Mask 均与历史 Seed 0 APBR Raw Hard 逐元素相等。
- Mask-only 重建权重与历史 Raw Hard 权重一致。
- 总 eligible 参数：11,157,504；删除 5,578,752；每个完整四元组恰好保留 2 个。
- Mask SHA256：`79d9c410c67e931101cf46112e732c68133a9935e4f99a41503a0de8ee7b40f4`。
- OIHW output-row contiguous、stable tie、dense tail、排除层语义均沿用冻结实现。

## 数值验证

针对性测试共 6 项并全部通过，覆盖外部 Mask 决定支持、无动态选择、幸存权重可改变、删除权重严格为零、有限值、无补偿分支等价 Raw Hard、局部误差计算和小型正定实例参考。相关 adapter 测试共 25 项通过；项目完整回归为 126 passed（另有 1 个 CPU pin-memory warning）。

全模型结果：

- 5,575,732 个幸存权重发生变化。
- 删除权重全部为零；参数全部 finite。
- 排除参数和 BN buffers 与稠密父 checkpoint 相同。
- 所有 19 层 `mask_equal_historical=true`、`deleted_zero=true`、`finite=true`。
- 所有层的局部二次误差比 `OBS / Raw` 小于 1，范围约 0.0643–0.5190。
- 最大层 `layer4.0.conv2`（K=4608）的误差比为 0.22570。

## 语义边界

该实现没有修改受保护的 `benchmark/model.py`、`benchmark/nm.py`、`benchmark/obc.py`，也没有修改原生 SparseGPT/ROSE 锁定源码。新实现位于独立 `baseline_adapters/fixedmask_obs/` 和独立 runner 中。它应命名为 **APBR + Fixed-Mask OBS**，不能称作原生 SparseGPT。

完整逐层数值见：

`artifacts/apbr_seed0_fixedmask_obs_20261027_blackwell_remote/apbr_seed0_fixedmask_obs_20261027_blackwell_attempt2/compensation_evidence.json`
