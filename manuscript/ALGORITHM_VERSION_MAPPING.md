# 算法版本映射

| 版本 | 首删 | 次删 | 补偿 | 现有结果/状态 |
|---|---|---|---|---|
| Physical conditional OBS | 当前物理单点OBS argmin | 虚拟条件OBS argmin | 组入口物理状态的一次联合OBS | 有历史SAM/APBR运行 |
| Near-optimal APBR conditional OBS | τ=.01物理窗口内几何argmin | τ=.01条件物理窗口内几何argmin | 同一物理联合OBS | APBR-VPR Seed0 77.83；Seed1 77.43；均值77.63±.28 |
| 本轮论文：APBR-guided conditional OBS | 当前组直接 `argmin |w|exp(-s)` | 剩余组三个候选中条件OBS argmin | 同一活动集上的一次联合OBS | PENDING；本文定义，未修改实现或运行实验 |

最后一行不能通过旧adapter设置τ=0得到；该设置会转为第一行。本文版本的两个阶段都不是阈值混合，第二阶段也不再使用几何破平局。两个阶段的精确平局均由低存储索引确定。

训练身份同样分开：本文保留固定实际L2半径的APBR定义；77.63对应的历史父模型是VPR/椭球约束机制。两者的公式和实验身份不能互换。

新Algorithm1的数学活动集 F 包含当前组和后续所有未冻结坐标。R为阻尼Hessian在F上的主块的逆，不是全逆矩阵任意主块，也不是孤立4×4矩阵。补偿使用R[:,D]影响全部F；当前组完成后冻结，下一组使用新F的有效逆曲率。

出处与哈希见 `TAU_ORIGIN_AUDIT.md`、`RESULT_PROVENANCE_CHECK.md`。这两个版本的区分已用于新正文；旧论文和历史代码未改。
