# 适配决策与算法语义边界

## Wanda

保留：`|W|·||X_j||₂`、每 output row/每合法 2:4 组删除最小两项、幸存权重不更新、无 Hessian/恢复训练依赖。

接口适配：通过 Conv2d patch 收集真实输入列统计；group/depthwise 独立处理；bool keep-mask；dense tail；固定 tie。

语义影响：dense tail 与官方 LLM partial-slice 行为不同，但这是统一项目约束且只影响非完整组；ResNet18 首层本来排除。其余完整组评分目标不变。建议名称：`Wanda-CNN-unified2:4`，不要写“官方 CIFAR-100 Wanda”。

结论：**建议进入正式实验准备**。它是三者中风险最低、运行成本最低的基线。

## SparseGPT

保留：`H∝XXᵀ`、relative damping、逆 Hessian Cholesky 因子、动态逐列 mask、块内/块间误差补偿和最终写回。

接口适配：Linear/Conv2d 统一矩阵接口；每个卷积 group 独立 H；dense tail；blocksize 必须整除 M；移除与算法无关的 CUDA timing fence；本轮 smoke 用 FP64 求解以放大坐标错误可见性。

语义影响：CNN patch 分布取代 token 激活，这是将同一局部重构目标应用到 Conv2d 的必要数学映射，不是仅 reshape 权重。统一 tail/tie 仍是约束变化。建议名称：`SparseGPT-CNN-unified2:4`。

结论：**建议推进单层 GPU/内存验收后进入正式实验**。不得在未做峰值显存实测前启动全模型。

## ROSE

保留（按论文）：Wanda 预剪损失、column/block 两级降序、Eq.(7) mean denominator/0.5 阈值、W/H 同步置换、SparseGPT 补偿、逆置换。

受控修复：N:M 中只在原四元组内重排列、以完整四元组为 block 重排；修复 pinned source 的未定义变量；采用论文 mean 而不是源码 max；Conv2d 使用正确 patch。

语义影响：这是**论文语义实现**，不是锁定官方源码的直接端到端结果。官方 N:M 触发路径已实测失败，因此没有可成立的“官方原生 vs CNN 适配数值等价”。如果论文作者后续修复官方代码，应重新锁 commit 并重复对照。

结论：**暂停正式 accuracy 队列，只保留研究性候选**。先要求：(1) 作者源码修复或作者确认；(2) 受控实现与修复官方 Linear N:M 对齐；(3) 对 ResNet18 层的 columnar-trigger 比例与收益做预注册单层检查。不能为凑齐三基线直接投入正式三 seed。

## 未触及部分

- 未修改 APBR-SAM 训练、几何状态、mask 或恢复流程。
- 未修改 OBC 实现和结果。
- 未下载/训练 CIFAR-100，不做 BN 重估，不运行 recovery。
- 三个 adapter 是独立模块，不共享一个混合评分器；ROSE 仅复用 SparseGPT 的经测试更新函数，算法差异保留在显式 permutation 阶段。
