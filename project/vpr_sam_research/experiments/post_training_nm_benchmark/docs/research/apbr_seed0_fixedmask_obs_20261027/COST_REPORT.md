# APBR + Fixed-Mask OBS 成本报告

## Blackwell 实测

| 阶段 | 秒 |
|---|---:|
| checkpoint 加载 | 0.3452 |
| geometry Mask 构建与身份核验 | 0.1443 |
| calibration wall（含 Hessian） | 0.6647 |
| Hessian CUDA event（calibration 子集） | 0.1949 |
| 分解/逆因子 | 0.1333 |
| OBS 补偿 | 3.4816 |
| pruning wall（含分解和补偿） | 4.0213 |
| post-compensation 验证 | 0.2093 |
| calibration + pruning（不含验证） | **4.4767** |
| Mask + calibration + pruning + 验证 | 4.8549 |
| 独立 Test 评估 | 0.6580 |

峰值 CUDA allocated 为 1,157,752,320 bytes（约 1.078 GiB）；峰值 reserved 为 1,516,240,896 bytes（约 1.412 GiB）。未发生 OOM、Cholesky 失败或非有限值。

## 历史对照

历史 Seed 0 在 RTX 4080 SUPER 上记录的剪枝/原生补偿时间：Wanda 1.159 s、SparseGPT Native 9.273 s、ROSE Native 9.889 s。APBR Recovery-20 为 356.347 s。Fixed-Mask OBS 本次运行在 RTX PRO 6000 Blackwell 上，因此 **不能把 4.477 s 与 4080 数值作严格硬件无关速度排名**。

即使只作成本边界解释，Fixed-Mask OBS 也没有 20 epochs 的梯度训练、优化器状态更新和 7,820 recovery steps；它在本机仅执行一次校准和逐层二阶补偿。该结果支持“显著低于 Recovery-20 训练成本”的工程结论，但精确加速倍数必须在同一 GPU 上复测后才能报告。
