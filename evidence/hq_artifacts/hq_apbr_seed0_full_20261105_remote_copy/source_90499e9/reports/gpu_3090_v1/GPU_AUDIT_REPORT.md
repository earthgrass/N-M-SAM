# HQ-APBR V1：RTX 3090 GPU 门禁审计

用户已选择“保持暂停，先看报告”；当前执行状态为 `USER_PAUSED_BEFORE_FORMAL_TRAINING`。原始性能门禁仍为 `GPU_SMOKE_BLOCKED`，没有修改阈值或自动重跑。临时 smoke checkpoint 已回传本地（179,290,175 bytes），SHA256 `ad09999870de610ccd48b875c50ebf8a4fb19f9061148dcfb9e956ac423fe134` 与远端一致；18 个已下载文件均通过校验，详见 `remote_run_v1/LOCAL_VERIFICATION.json`。

本次真实 GPU 数值、显存和三轮临时训练均通过；**整体状态仍为 `GPU_SMOKE_BLOCKED`**。每轮完整校准耗时与训练耗时之比为 **30.5526%**，超过事先冻结的 30% 上限 0.5526 个百分点，程序按规则停止。未启动正式 200 Epoch，未访问 Test，未执行剪枝或 Recovery。

| 检查 | 实测结果 |
|---|---|
| GPU / 软件 | RTX 3090，标称 24 GiB，Compute Capability 8.6；Driver 580.105.08；Python 3.12.3；PyTorch 2.8.0+cu128；CUDA build 12.8 |
| CPU/GPU inverse diagonal | FP32 relative L2 1.653e-07；FP64 3.784e-16 |
| CPU/GPU Geometry / epsilon | FP32 最大绝对误差 1.964e-07 / 2.794e-09；FP64 6.384e-16 / 0.000e+00 |
| Epoch 0 标准 SAM 方向 | 与直接全局 L2 SAM 方向最大绝对误差 2.837e-11 |
| 全局扰动范数 | rho=0.05；三轮最大相对误差 2.704e-09；zero steps=0 |
| 19 层映射 | 11,157,504 eligible 坐标，固定 1,024 Train 图片；完整 19 层刷新 |
| Cholesky / finite | 单步门禁及三轮 smoke 共 76 次层级分解全部 PASS；无 NaN、Inf、OOM 或分解失败记录 |
| Geometry 中心化 | 全部刷新最大绝对坐标和 6.112e-10，通过按坐标绝对量加权的预定容差 |
| BN / 权重隔离 | 每次校准前后 buffers 与参数 SHA256 完全相同；校准 RNG 恢复通过 |
| Resume | 真实网络 GPU 单步中断恢复后，模型、Optimizer、Scheduler、Geometry、RNG 逐位相同 |
| Peak allocated / reserved | 1.5528 / 2.1270 GiB，低于冻结的 20 / 22 GiB 上限 |
| 临时训练 | Epoch index 0、1、2；每轮 50,000 样本、391 batches；累计 1,173 步 |

Epoch 0 使用零 Geometry，随后保存历史 Q 更新；Epoch 1 使用前一轮已保存 Geometry，哈希链逐轮一致。模型参数更新与恢复检查均通过。三轮 Train CE 从 4.02159 降至 3.01648，仅表示 smoke 中优化正常，不构成准确率或泛化结论。

失败证据保留在 `remote_run_v1/run_v1/failure_1793865727158228278.json`。其 RuntimeError 明确为“Calibration overhead exceeds frozen limit; no adaptive fallback”，属于性能规则阻塞，没有自动重试或放宽阈值。若接受实测 30.55% 的校准开销，需要明确修订资源门禁；也可提出独立优化方案并重新实测。当前不可标记完整 GPU 门禁 PASS。

追溯：代码 commit `ec63a173a02b9e319b99abc251f3cb7c3494c96d`；config SHA256 `56ff197b0860dfd0f4a0768c5609031214e6405c51557009a9977582b38400e7`；source manifest SHA256 `5ba28563953e14ee1f5137545f9ddd35feae3afc41ca96833afd961e08b94aa8`。数值、BN 哈希、GPU UUID、失败 traceback 和逐层记录汇总于 [SUMMARY.json](SUMMARY.json)，原始证据保存在 `remote_run_v1/run_v1/`。所有 checkpoint 均为临时门禁/smoke 产物，没有 Epoch 199 正式父模型。
