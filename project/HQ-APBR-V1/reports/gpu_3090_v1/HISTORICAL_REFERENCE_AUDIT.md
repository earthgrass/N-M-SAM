# HQ-APBR V1 历史参考审计（只读）

日期：2026-11-05。本审计只读取既有评测 JSON、checkpoint 元数据和权重/Mask，并重新计算文件 SHA256。没有运行模型推理，没有打开 CIFAR-100 Test 数据。HQ 的正式 200 Epoch 训练尚无最终评测，本表全部标为 **REFERENCE_ONLY**，不产生 HQ 的准确率或排名。

| 历史记录 | Dense Accuracy | Native SparseGPT Accuracy | Native CE | 相对 Dense |
|---|---:|---:|---:|---:|
| sam_formal_original_sparsegpt | 78.84% | 77.47% | 0.85025973 | -1.37 pp |
| sam_paired_rerun_original_sparsegpt | 78.84% | 77.51% | 0.85023241 | -1.33 pp |
| apbr_original_sparsegpt | 78.68% | 76.99% | 0.85810039 | -1.69 pp |

## 身份与选择依据

第一行是最早正式 Seed 0 SAM + SparseGPT 的 `raw_hard.pt`，包含原生动态 Mask 与 OBS 补偿；同目录虽另有 BN-only 和 Recovery-20，表中没有使用那些结果。第二、三行来自后来同一次 `apbr_weights_original_sparsegpt_seed012` 对照，其两个原始 Dense 父模型就是同一正式 Seed 0 SAM / APBR Epoch-200 checkpoint。当前逐文件哈希、父哈希、1,024 个校准 ID、19 层 bool Mask、11,157,504 坐标、5,578,752 删除位置、严格 2:4 与删除位置零值全部通过只读复核。

SAM 两次原生 SparseGPT 的实际结果分别是 **77.47%** 和 **77.51%**，checkpoint 与 Mask 哈希不同。它们不是同一文件的重复别名。既有正式三 Seed 表继续使用原正式记录；与后来的 APBR 原生 GPT 同批次参考时使用 77.51% 行，且明确注明批次。不因数值高低替换记录。本审计没有定位这 0.04 pp 差异的首个数值来源，不能直接断言仅由硬件导致。

SAM Dense SHA256：`3a0716555d33ec92efe8362e24abec9be27625d24ffb6dfb1d51fddb5b91bd13`。

原版 APBR Dense SHA256：`4332079becdb58fb50f0e68b8bd0dcc251db0184a6050facb524817520a51d82`。

原版 APBR Native SHA256：`0ff0eb2f8a9371ab47f39a8a8d0ea8b237e4f0ed89eb6ff583f05a4d4d1d6e04`。

完整评测文件绝对路径、源码/配置来源、逐文件 SHA256、Dense 初始模型身份和剪枝算法记录保存在 [reference_results.json](reference_results.json)。评测 JSON 中 Test 样本数均为 10,000，`selection_performed=false`、`hyperparameter_change_performed=false`，记录的 Test pickle SHA256 一致；此次未重新打开该 Test pickle。

## 与 HQ 后续比较的边界

名义协议一致：官方 50k/10k、100 类 CIFAR ResNet18、200 Epoch、SGD/cosine、batch128、rho0.05、原生 GPT 的 FP32/block128/damping0.01、固定 1,024 Train 图片及顺序、19 层 OIHW 2:4。原生 GPT 均为 `2XXᵀ/n` 和原生逐列补偿，无 Recovery、无 BN-only。

HQ 有意改变训练期几何；原版 APBR 的能量反馈与椭球扰动不等于 HQ 的历史 Q 与固定全局 L2。HQ 当前设备为 RTX3090，历史 Dense 训练记录为 RTX4080SUPER，同批次 Native 重剪枝记录为 RTX5090。HQ 正式启动的 [初始化证明](formal_start/INITIALIZATION_PROOF.json) 已与两份历史 Dense 身份核对：canonical model tensor SHA256 均为 `87aa9dd7492a89adf9e3ab78ca4a3227f30bc8de53b18a3bd73cd61ab0bd759d`。因此三者具有相同模型初始张量；HQ Geometry 从全零开始，未复用 smoke 状态。初始化证明文件 SHA256 记录在 reference JSON。当前仍未建立全网络跨环境 replay。HQ 逐层采集与历史全层 hooks 采集虽使用相同冻结校准 ID 和批大小，也要在最终记录中保留实现差异。

因此，本次提供可追溯的历史参照；HQ Final 完成前不能宣称 HQ 相对这些记录的收益。届时只有明确的 **HQ + Original SparseGPT Native** 可与本表的 Native 终点比较；HQ Final-Q Mask-only 和固定 Mask OBS 要分别列出，并保留单 Seed、跨训练父模型/硬件的局限。
