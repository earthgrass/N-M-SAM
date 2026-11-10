# 同父模型、同固定 Mask OBS：CPU 全模型对照

## 执行身份

- 原版 APBR Seed 0 Epoch-200 Dense 父模型 SHA256：`4332079becdb58fb50f0e68b8bd0dcc251db0184a6050facb524817520a51d82`。
- 两套历史 Mask 分别来自 APBR Raw Hard 和该父模型上的原生 SparseGPT；19 层、11,157,504 个可剪坐标，每套删除 5,578,752 个，严格 2:4。
- 两个分支均从原始 Dense 权重出发，使用同一批固定的 1,024 张 CIFAR-100 Train 图像构造同一个逐层 Hessian；对各自固定 Mask 使用同一 `fixedmask_obs_matrix`：FP32、相对阻尼 0.01、block size 128。无动态重新选 Mask、无 BN 更新、无训练或 Recovery。
- 两份 checkpoint 先写入并记录 SHA256，再各执行一次 10,000 张固定 Final Test。本次 CPU 运行，不宣称 GPU 性能测量。

## 结果

| 同一 APBR Dense 父模型 | Mask-only Test | 相同固定 Mask OBS 后 Test | OBS 后 CE |
|---|---:|---:|---:|
| APBR Mask | 59.22%（历史） | **76.37%** | 0.903978 |
| 原生 GPT Mask | 48.66%（历史） | **76.99%** | 0.858095 |

GPT Mask 在无补偿时低 10.56 个百分点；接受相同固定 Mask OBS 后高 0.62 个百分点。两套 Mask 有 2,384,158/11,157,504（21.37%）位置不同、1,130,692/2,789,376（40.54%）四元组不同。

在相同 Train 校准激活上的逐层物理输出二次误差之和：APBR Mask-only `8.73414`、GPT Mask-only `9.00438`；OBS 后分别为 `1.50534` 和 `1.04726`。两种 Mask 的 19 层误差都降低；GPT Mask 在 OBS 后的 19 层误差均低于 APBR Mask。此“逐层求和”仅作局部诊断，不等于整网输出 MSE。

## 判断与边界

这次对照支持一个具体结论：在**该 APBR Dense Seed 0 父模型、该固定 Mask OBS 实现与冻结校准集**上，GPT 选出的支持集比 APBR 支持集更容易通过相同补偿恢复。无补偿时 APBR Mask 较好，补偿后次序反转，说明仅比较 Raw Hard 会掩盖 Mask 与补偿器的相互作用。

这不证明 APBR 几何信息整体无效，也不证明所有 seed 或其他补偿器上同样如此。原生 SparseGPT 是动态选择与顺序补偿；本实验先取其**最终 Mask**，然后从原始 Dense 权重独立做固定 Mask OBS，因此它是支持集对照，不是原生动态过程的逐步因果分解。CPU 结果在两位小数上与历史 APBR Fixed-Mask OBS 76.37% 和该父模型 GPT Native 76.99% 一致，但没有据此宣称 checkpoint 位级相同。

## 证据

- 完整输出与逐层日志：`artifacts/apbr_gpt_mask_fixed_obs_cpu_seed0_20261103/`。
- 两份冻结 checkpoint：`apbr_fixed_obs.pt`、`gpt_fixed_obs.pt`；Test 前哈希：`pre_test_freeze.json`；最终结果：`results.json`。
- checkpoint 文件 SHA256 分别为 `c522900cf634d27731f418627b4f80a15d1905e93a0050b8e5641e49e695e97f`、`75fe12050a69ccffc22e29a00bcd43751cfbb8c3fb195161fdbce668c8ae051f`，已重新计算并匹配。两份 Mask 为 bool，所有删除权重严格为零。
