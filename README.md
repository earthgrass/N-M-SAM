# VPR-SAM：面向 N:M 结构化稀疏的全局等体积扰动重分配

**Volume-Preserving Perturbation Redistribution for Sharpness-Aware Minimization**

本项目研究一个问题：**在保持 SAM 扰动集合总体积不变的条件下，动态调整不同参数方向的扰动半径，能否使模型更好地适应后续 N:M 结构化剪枝？**

研究主线是“传统 SAM 与结构化删除的几何失配 → 全局等体积扰动重分配 → 相同剪枝规则下的硬剪枝损伤验证”。当前已建立独立实验工程，完成工程与短程数值验收，进入多随机种子的正式对照阶段。本页介绍研究思想与实现，不展示实验数据，也不预先宣称性能收益。

## 1. 传统方法的问题：局部球形扰动与结构化删除并不等价

标准 SAM 在参数周围的欧氏球内构造一阶近似最坏扰动：

$$
g_{1,t}=\nabla L_t(\theta_t),\qquad
\epsilon_t^{\mathrm{SAM}}=\rho\frac{g_{1,t}}{\lVert g_{1,t}\rVert_2}.
$$

N:M 结构化剪枝则要求在每个规定的权重块中保留 N 个、删除其余权重。用二值 Mask $M$ 表示，实际参数变化是：

$$
\theta_M=M\odot\theta,\qquad
\delta_M=(M-1)\odot\theta.
$$

两者之间存在两个需要研究的失配：

- **尺度失配**：硬剪枝会把选中的权重直接置零，其有限幅度变化可能超出 SAM 的局部扰动范围。
- **方向失配**：SAM 的扰动由当前任务梯度决定，删除方向则由 Mask 和被删除的权重决定，两者通常不重合。

标准球形集合对所有单位方向提供相同的最大允许半径，但不同参数方向对训练与后续删除的作用可能不同。因此，问题不只在于全局半径大小，还在于**扰动空间如何在参数方向之间分配**。

## 2. 我们的方法：改变扰动几何，保留 SAM 训练骨架

VPR-SAM 为每个参与扰动的参数坐标引入正尺度 $a_{i,t}$，令：

$$
A_t=\operatorname{diag}(a_{1,t},\ldots,a_{d,t}),\qquad A_0=I,
$$

$$
\mathcal E_t=\{\epsilon:\lVert A_t^{-1}\epsilon\rVert_2\leq\rho\}.
$$

扰动集合由固定球体变为动态的、以零扰动为中心的各向异性椭球。第 $i$ 个参数轴的允许半径为 $\rho a_{i,t}$。初始化时全部尺度为 1，因此起点与标准 SAM 相同。

### 全参数空间等体积约束

$$
\prod_{i=1}^{d}a_{i,t}=1,\qquad
\operatorname{Vol}(\mathcal E_t)=V_d\rho^d\det A_t=V_d\rho^d.
$$

守恒范围是**整个参与 SAM 扰动的参数空间**，不是逐层、逐参数张量或逐 N:M 块独立守恒。这允许某些方向收缩、另一些方向扩张，包括跨层的半径重分配。

### 动态椭球中的一阶最坏扰动

对当前损失作一阶线性化，在上述椭球内最大化 $g_{1,t}^{\mathsf T}\epsilon$，得到：

$$
\epsilon_t^*=\rho\frac{A_t^2g_{1,t}}{\lVert A_tg_{1,t}\rVert_2}.
$$

当 $A_t=I$ 时，该式退化为标准 SAM。第二次梯度仍为：

$$
g_{2,t}=\nabla L_t(\theta_t+\epsilon_t^*).
$$

随后恢复原参数，再由原基础优化器使用 $g_{2,t}$ 完成更新。VPR-SAM 不增加新的训练损失，也不改变这两次梯度的定义。

## 3. 核心机制：扰动投影的全局负反馈

使用本轮实际扰动在各坐标上的平方投影，定义全局能量占比：

$$
p_{i,t}=\frac{(\epsilon_{i,t}^*)^2}{\lVert\epsilon_t^*\rVert_2^2},\qquad
\sum_{i=1}^{d}p_{i,t}=1.
$$

下一轮的半径尺度按照冻结的指数负反馈更新：

$$
a_{i,t+1}=a_{i,t}\exp\!\left[\eta_a\left(\frac1d-p_{i,t}\right)\right].
$$

- 高于平均占比 $1/d$ 的方向，下一轮允许半径减小。
- 低于平均占比的方向，下一轮允许半径增大。
- 全局指数增量之和为零，因此从单位初始化出发，在精确算术下自然保持等体积，不需要额外的体积回正。

实现中使用 $s_i=\log a_i$ 和 $\gamma=\eta_a/d$，得到完全等价的更新：

$$
s_{i,t+1}=s_{i,t}+\gamma(1-dp_{i,t}),\qquad \sum_i s_{i,t}=0.
$$

该反馈调整的是**下一步扰动集合**。它复用已有扰动，不需要第三次反向传播，也不需要构造 Hessian 或 Jacobian。

## 4. 解决什么问题，以及如何验证

在方法层面，VPR-SAM 使原本固定、各向同性的扰动集合获得了**全局等体积的动态方向分配机制**。研究目标是借此改善训练中的参数扰动探索，使模型更能适应后续结构化删除。

这一机制并不自动保证扰动与删除方向对齐，也不保证完整覆盖硬剪枝变化。**等体积不等于每步扰动的 L2 长度相同，半径大小也不直接等于权重可删除性。** 是否减少剪枝损伤，需要通过受控实验检验。

当前正式研究只比较 Dense-SAM 与 VPR-SAM：从同一 seed 对应的随机初始化出发，统一数据划分、样本顺序、训练预算与非研究变量；训练完成后，对各自的 final checkpoint 使用相同 Magnitude N:M 规则，再进行相同预算的固定 Mask 恢复。

核心观察是剪枝前后的任务损失与准确率变化，其中硬剪枝损伤定义为：

$$
\Delta L_{\mathrm{hard}}=L(M\odot\theta)-L(\theta).
$$

同时检查公共外部球形 probe、几何演化以及时间与显存成本。当前不同时引入新的 Mask Predictor、Compensation 或几何引导剪枝评分，以便将方法差异归因于扰动几何本身。N:M 由配置指定，算法不硬编码某一稀疏比例。

## 5. 大致实现流程

```text
随机初始化模型，初始化全局 log-radius 为零
    ↓
原参数上计算 g1
    ↓
用当前全局椭球构造 epsilon*
    ↓
临时扰动参数，计算 g2
    ↓
恢复原参数，由基础优化器完成一次更新
    ↓
成功步后，根据本轮 epsilon* 更新下一步全局 log-radius
    ↓
完成稠密训练 → 相同 Magnitude N:M 硬剪枝 → 固定 Mask 恢复 → 配对评估
```

独立工程包含模型、数据、优化器、训练、通用 N:M 剪枝、恢复、评估、统计和绘图。实现固定参与扰动的参数集合与全局维度；全零扰动跳过几何更新，非有限数值显式失败；checkpoint 保存模型、优化器、几何、随机状态及来源信息，以支持严格恢复与结果溯源。首版不加入 EMA、半径裁剪或额外优化目标。

## 6. 分支与代码入口

| 分支 | 定位 |
|---|---|
| [main](https://github.com/earthgrass/N-M-SAM/tree/main) | 研究介绍入口；基于原默认分支建立，未合并整改 PR 的实现代码 |
| [vpr-sam-audit-fixes](https://github.com/earthgrass/N-M-SAM/tree/vpr-sam-audit-fixes) | 当前 VPR-SAM 独立工程、审计修复与正式流程配置 |
| [VPR-SAM](https://github.com/earthgrass/N-M-SAM/tree/VPR-SAM) | 独立工程的早期基线，保留其版本定位 |
| [research/implementation-v1](https://github.com/earthgrass/N-M-SAM/tree/research/implementation-v1) | 早期 N:M × SAM 实现分支 |
| [research/rebuild-v1](https://github.com/earthgrass/N-M-SAM/tree/research/rebuild-v1) | 早期工程重建分支 |
| [初版待完善版本](https://github.com/earthgrass/N-M-SAM/tree/初版待完善版本) | 历史 N:M + SAM 工程归档，当前仓库默认分支 |

最新实现请从 [vpr_sam_research](https://github.com/earthgrass/N-M-SAM/tree/vpr-sam-audit-fixes/vpr_sam_research) 进入；研究数学、复现流程与已登记协议均在该独立工程中。历史分支的具体实现不因 README 更新而变成最新实现。

## 当前分支说明：vpr-sam-audit-fixes

本分支包含当前独立工程，入口为 [vpr_sam_research](vpr_sam_research/README.md)。正式队列使用冻结提交 `d4c58d66b248b620e0e058b92e79100f173970a6`；后续 README 提交不改变该运行来源，也不推进服务器 checkout。[PR #1](https://github.com/earthgrass/N-M-SAM/pull/1) 保持未合并。
