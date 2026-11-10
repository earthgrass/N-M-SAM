# HQ-APBR V1：三轮 Geometry 演化记录

用户已选择“保持暂停，先看报告”；当前执行状态为 `USER_PAUSED_BEFORE_FORMAL_TRAINING`。原始性能门禁仍为 `GPU_SMOKE_BLOCKED`，没有修改阈值或自动重跑。临时 smoke checkpoint 已回传本地（179,290,175 bytes），SHA256 `ad09999870de610ccd48b875c50ebf8a4fb19f9061148dcfb9e956ac423fe134` 与远端一致；18 个已下载文件均通过校验，详见 `remote_run_v1/LOCAL_VERIFICATION.json`。

本报告只描述临时 smoke 的训练侧状态，未访问 Test。历史 EMA 系数固定 eta=0.1，Geometry 为 FP64；每轮结束后使用 1,024 张固定 Train 图片更新一次，未跳过任何刷新。

| Epoch index（轮末更新） | log Geometry 漂移 RMS | 新 s 最小 / 最大 | 全局坐标和绝对值 | Q 排名稳定性均值 | 本轮非 eligible 扰动能量 |
|---|---:|---:|---:|---:|---:|
| 0 | 0.041054 | -0.134165 / 0.086020 | 9.641e-11 | 未定义（首次刷新） | 48.02% |
| 1 | 0.035007 | -0.268357 / 0.153847 | 4.256e-10 | 0.7521 | 38.20% |
| 2 | 0.029757 | -0.412836 / 0.182535 | 6.112e-10 | 0.8441 | 34.89% |

Epoch 0 的训练使用零 Geometry，因此方向与标准全局 L2 SAM 一致；表中 Epoch 0 的新状态用于 Epoch 1。Epoch 1 与 2 的 `current_geometry_hash` 分别等于上一轮 `next_geometry_hash`；没有使用未来 Q。最终临时 Geometry 哈希为 `b9973891ca1db03e6e14fca9c20158e8098eeb8ec03deb063e496ce4e61ba054`。

Q 对角全层范围依次为 [8.33057, 681.04541]、[5.39916, 692.16846]、[2.31445, 662.38818]，均为正且有限。第二次和第三次刷新各层 Q 的 Spearman 范围分别为 [0.6506, 0.8649] 与 [0.7866, 0.9445]。首次刷新没有前一轮 Q，稳定性为 null，未伪造为零或一。

三轮的非 eligible 参数能量占比从 48.02% 变为 34.89%；最大单个 eligible 层占比分别为 6.75%、9.70%、10.10%。这是随权重、梯度和 Geometry 同时变化的观测，不能仅凭这些占比把变化因果归属于历史 Q，也不能据此宣称剪枝性能改善。

每次校准的参数与 BN buffers 前后 SHA256 一致；RNG 恢复通过，Geometry 全局中心化满足预设数值容差。全局 epsilon L2 半径始终约 0.05，最大相对误差 2.704e-09。本轮只有 3 Epoch/1,173 步，不足以证明 200 Epoch 稳定性。性能规则阻塞及原始失败记录均保留，尚无 Dense Final、Mask-only、OBS 或 SparseGPT 的最终测试结果。

逐层 Q/Geometry 统计、哈希链、能量分布及 BN 哈希见 `remote_run_v1/run_v1/smoke/history.json`、各轮 calibration JSON 和 [SUMMARY.json](SUMMARY.json)。
