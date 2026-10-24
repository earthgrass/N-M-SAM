# N:M + VPR-SAM research archive

最新可核验状态：V1 SAM/VPR 三种子 200 epoch 主实验及固定 Mask 恢复已完成；VPR-L2 seed 0、Geometry-guided 2:4 和 20 epoch 恢复已完成。

同预算 seed 0 的 Best Validation Accuracy 为 Dense-SAM+Magnitude 95.90%、VPR-L2+Magnitude 96.00%、VPR-L2+Geometry-guided 96.08%。这是事后验证集选优；Final 主结果保留，尚无多 seed VPR-L2/Guided 与独立 Test 支持，不能宣称普遍优于 SAM。

[完整独立工程](vpr_sam_research/README.md) · [研究交付](vpr_sam_research/results/research_delivery/RESEARCH_STATUS.md) · [实验协议](vpr_sam_research/docs/EXPERIMENT_PROTOCOL.md)

Block-wise VPR 仍是独立候选设计和 CPU 原型，没有正式训练结果。此归档分支不覆盖 main 或历史冻结分支，也不自动合并 PR。旧工程和实验留存于历史分支。
