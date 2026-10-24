# BN mechanism audit

Frozen step.py：center forward 为 model.train，更新 running stats；第二个对抗 forward 在 freeze_bn 中将 track_running_stats=False，使用当前 batch 统计但不更新 running buffers。退出恢复标志；原文件未改。
Raw Hard 仅权重乘固定 Mask，保留所有 BN buffers。BN-only 是独立模型副本，reset_running_stats、momentum=None、仅 BN train、其他层 eval、no_grad、全部参数 requires_grad=False；校准前后参数 tensor_hash 相同。
三方法使用相同 4096 原始 train IDs、无增强、32×128、验证 5000/batch256。Raw Recovery 的父 checkpoint 为未经校准 pruned.pt；SGD 新优化器、固定 Mask，20 epoch/7040 steps；数据 manifest 和 order_hash 相同。
VPR-L2 中 BN affine 参与全空间几何；Mask 仅 Conv/Linear weight。geometry_allocation.csv 给出各参数 log-volume 与 RMS，用于审查跨空间预算迁移。预算偏移不能仅由参数数量推断为真实性能原因。
Guided−Magnitude Raw Accuracy +0.62pp，BN-only 后 −0.10pp；Raw loss 改善 0.01469609，BN 后 loss 反而增加 0.00026513。BN 校准显著减弱 Raw 收益；不能据此断言 BN 是唯一机制。
Dense-SAM BN-only 本轮补充评价：Accuracy 0.931，Loss 0.2070142692565918；它是新非训练诊断，未追溯替换 V1 正式 Raw/Final。
未发现需要更改冻结历史数据的 BN 规则缺陷。若未来比较其他 BN 算法/样本量，必须预登记并保留本口径；本轮未搜索 BN 样本量。

几何分配：Mask-eligible 11157504 坐标的 log-volume +3342.6406452187、s RMS 0.00756526；非剪枝16458坐标总量 −3342.6406452187、s RMS 0.26969966，互相抵消保持全局体积。这证明有显著空间间几何迁移，但不证明该迁移造成剪枝损伤或 BN 效应。
