# Post-pruning 成本只读审计

## Seed 0 实测剪枝阶段

| 方法 | Calibration (s) | Pruning/compensation (s) | Sum (s) | Whole-process peak alloc (GiB) | Whole-process peak reserved (GiB) |
|---|---:|---:|---:|---:|---:|
| magnitude | 0.000001 | 0.136342 | 0.136343 | 1.082 | 1.555 |
| wanda | 0.975107 | 0.184185 | 1.159292 | 1.575 | 2.387 |
| sparsegpt | 1.006849 | 8.266102 | 9.272951 | 1.082 | 1.873 |
| rose | 0.908881 | 8.980367 | 9.889247 | 1.082 | 1.873 |

计时范围：`calibration_seconds` 覆盖 1024 张 Train 图像的前向统计；`pruning_seconds` 覆盖所有 19 层 mask 生成及 SparseGPT/ROSE 原生补偿。代码在逐层调用内部做 CUDA synchronize，因此 SparseGPT/ROSE 的分解与补偿完成后才返回。状态文件的显存峰值从剪枝前 reset 后持续覆盖剪枝、BN-only 与 Recovery-20，不能当作“纯二阶补偿峰值”。

SparseGPT 的 calibration+pruning 为 9.273s，比 Wanda 多 8.114s（8.00×）。ROSE 为 9.889s，比 SparseGPT 多 0.616s（总阶段 1.066×）。这是本项目同一硬件/Seed 0 的实际差额，不是论文计时。

## 二阶算法路径

SparseGPT 为每层构造 `H=2XX^T/T`，相对阻尼 `0.01*mean(diag(H))`，执行 Cholesky → inverse → upper Cholesky，随后在补偿后的当前 block 中动态确定 2:4 mask，进行块内和块间 OBS 误差传播。ROSE 先由 Wanda 分数估计 column/block loss，只在原四元组内部及完整四元组之间重排，同步置换 W/H，调用同一 SparseGPT 核心，再逆置换回原坐标。当前 ROSE 是论文语义受控修复版，不是锁定官方 N:M 分支的原样执行。

## 单层最大卷积 GPU 证据

最大层 `layer4.0.conv2`（K=4608）的既有 GPU gate 在 RTX 4080 SUPER 上记录：Hessian build 0.012786s、decomposition 0.075125s、compensation 1.370275s、Hessian+prune total 1.551802s；峰值 allocated 495.01 MiB、reserved 586.00 MiB。该 gate 是固定 32 张图像的单层资源测试，不能替代 Seed 0 全模型 1024-image 成本。

## 完整方法成本边界

- Dense-SAM 200 epochs 的 epoch-time 合计 10709.1s。
- APBR-SAM 200 epochs 的 epoch-time 合计 15714.8s，比 Dense-SAM 多 5005.7s（1.467×）。
- APBR pruning 本身只有 0.204s，但完整 APBR 方法还包括几何训练的额外成本和 Recovery-20 356.3s，不能只报打分时间。
- Wanda Recovery-20 为 398.2s；SparseGPT/ROSE 的原生比较终点是补偿后 Raw，若另报 Recovery-20 必须作为补充结果。

## 不能从现有日志得到的量

全模型日志没有逐层保存 Hessian build/decomposition/compensation 的细分计时，也没有把剪枝阶段显存峰值与 Recovery 峰值分离。因此这些字段保持 `UNVERIFIED`；不能用单层 gate 或整个 postprocess 的时间冒充全模型纯二阶补偿成本。
