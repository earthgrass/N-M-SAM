# Research status | 2026-10-24

V1 SAM/VPR seeds 0/1/2 的六条 200 epoch 轨迹及六条 20 epoch 恢复完成。VPR-L2 seed 0、三 Mask 无训练评价和两条 Magnitude/Guided 恢复完成。
同预算 seed 0 Best Validation Accuracy：Dense-SAM+Magnitude 95.90%（20）；VPR-L2+Magnitude 96.00%（8）；VPR-L2+Guided 96.08%（15）。
Final 分别为 95.90%、95.84%、95.88%；不能用探索性 Best 改写 Final 主结果。Guided 的验证 Loss 不优于 SAM。尚无多 seed VPR-L2/Guided 或独立 Test 泛化证据。
本轮完成 32 coarse + 32 fine matched-random Mask 无训练评价。Block-wise 仅独立原型、CPU 测试与数学设计，没有正式训练结果。
