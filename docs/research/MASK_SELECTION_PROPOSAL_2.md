# 方案二：以鲁棒上界和旧 Mask 下界控制结构切换

状态：DESIGN_ONLY；选择器未实现、未执行实验、未确认新颖性。
本方案不估计 Hessian，也不复原缺失残差。它把最终鲁棒目标的一个可计算上界
作为选择对象，以保守接受条件换取低计算量和清晰的条件保证。

## 1. 原始目标与候选限制

对固定任务损失 L(Z;B,s)，定义：

```text
Fρ(P_M W,M)=max_{P_Mδ=δ, ||δ||₂≤ρ} L(P_M W+δ)。
Mask 决策：min_{M∈合法2:4候选} Fρ(P_M W,M)。
```

不稀疏化参数的投影恒为 I；扰动球是整个允许参数空间的一次全局 L2 球，
不是每个四元素块各自一个 ρ。候选可限制在预算内，但每个候选评价的仍是全网络目标。
这里的保证针对固定 L；单个 minibatch 的安全下降不等于全数据或测试性能下降。
E_B max L_B 与 max E_B L_B 不能混为一谈。

## 2. 为什么用上界而不是经验分数

若能计算 U(M')≥F_new，而旧目标有 LB_old≤F_old，那么

```text
U(M') < LB_old−margin  ⇒  F_new < F_old−margin。
```

这是严格的接受逻辑。选择准则应由一个覆盖新支撑的鲁棒上界推导，
不能先用 |W*g2| 排名，再把它解释为目标一致。
该逻辑并不保证找到所有有利切换：上界和下界间隙大时，正确结果可以是拒绝全部候选。

## 3. 可复用的观测及假设

完整 Sparse-SAM step 在旧 M 下给出 W0、ε0、raw G2 和第二遍 task CE。
取 Z0=P_M W0、x0=Z0+ε0、ell=L(x0)。ε0 可以由 optimizer J 构造，
但 P_Mε0=ε0、||ε0||≤ρ 已使它成为任务目标中的**可行点**，无需假定它是精确最坏点。

默认 identity STE + master L2 + 同批次/状态函数时，端点梯度数值可恢复为：

```text
b=∇_Z L(x0)=G2_STE_J−λD(W0+ε0)。
```

这是链式法则的条件结论，不把 STE 二阶导数当成真实 Hessian。
D 必须符合实际正则集合；effective-space L2 需另用其正则梯度。
PureMasked 不提供旧 inactive 坐标的 b，新增支撑不能用缺失梯度冒充完整 b。
若比较 eval_fixed_buffers 的 F，则训练 g2 不能直接复用：需额外一次该状态下的
functional loss/gradient。两个目标口径必须分别记录。

关键数学假设是：在包含 x0、旧比较点、新候选球及相关线段的凸区域中，
∇L 是 β-Lipschitz，且 β≥0 是有效的全网络界。因此：

```text
|L(x0+t)−ell−bᵀt| ≤ β||t||₂²/2。
```

这是 smoothness 引理；不需要驻点，也不假定 Hessian 半正定。
ReLU 跨激活边界可能破坏梯度 Lipschitz 假设。块内 λmax、少量随机 HVP 或
观察到的梯度差都不能自动证明全网络 β 的上界。

## 4. 从 smoothness 推导新 Mask 的全局鲁棒上界

设实际待比较的候选中心为 Z0+v，候选支撑为 M'，定义 t=v−ε0。
对任何合法 δ∈S_M'：

```text
L(Z0+v+δ)
≤ ell+bᵀ(t+δ)+β||t+δ||²/2
= ell+bᵀt+β||t||²/2 +(b+βt)ᵀδ+β||δ||²/2。
```

由于 β≥0 且允许空间非空，球上最后两项的最大值为
ρ||P_M'(b+βt)||₂+βρ²/2（非空条件在实际网络的 dense 坐标中成立）。所以：

```text
U(v,M') = ell+bᵀ(v−ε0)+β||v−ε0||²/2
          +ρ||P_M'[b+β(v−ε0)]||₂+βρ²/2。
Fρ(Z0+v,M') ≤ U(v,M')。
```

这里 b+βt 是上界模型中的线性系数，不是假称求到了新中心真实梯度。
支撑变化显式进入 P_M'，普通损失变化和最坏邻域增加来自同一上界。
没有省略 g 非零的一阶项，没有把曲率标量当成 sharpness 本身。

## 5. 旧目标下界与更新时点

必须在优化器完整更新后、其他 Mask 调度前比较相同的 W1。
令 ΔW=W1−W0，v_old=P_MΔW，v_new=P_M'W1−P_MW0。
则 q_post=v_new−v_old=(P_M'−P_M)W1。

旧支撑中仍可使用 ε0，故旧真实目标至少达到 L(x0+v_old)。由 smoothness 下界：

```text
LB_old = ell+bᵀv_old−β||v_old||²/2
       ≤ L(P_MW1+ε0) ≤ Fρ(P_MW1,M)。
```

于是 **U(v_new,M')<LB_old−margin** 足以保证该固定状态下真实 Fρ 降低。
若 W1=W0，下界直接是已观测 ell；不需要第二次完整网络损失评价。
这也解释了为何不允许把 W_post*g_pre 作为没有时点修正的 Taylor 项。

多个块可以在暂存 Mask 中贪心减小同一个全局 U，但只在最后一次接受完整候选：
U_final<LB_original。不能把多个块独立上界差相加后假称联合目标降低。
若希望每个中间改动也有保证，需用当前支撑可行的 δ_A=P_Aε0
重新构造 LB_A，而非继续假定旧 ε0 属于所有新支撑。

浮点版本还需要 ell/梯度/范数归约误差的有效裕量。
仅加一个任意 epsilon 不构成数学误差证书。

## 6. 可执行设计伪代码（未实现）

```text
capture the fixed loss context before the step; retain W0/M/raw endpoint data
complete existing Sparse-SAM with the mask frozen, obtaining W1
on a declared mask-update frame:
    recover ell,b only if STE/L2/state contracts match the target L
    obtain a valid beta bound, or explicitly enter unguaranteed heuristic mode
    compute global old lower bound and anchor statistics once
    temporary_mask = M
    for at most K predeclared eligible blocks:
        enumerate exactly six legal masks in stable order
        compute their full-network U using changed-coordinate cache updates
        retain the candidate with minimum U (current mask remains an option)
        reject candidates outside the region in which beta is valid
    if U(temporary_mask)+numerical_margin < LB_old−margin:
        commit temporary_mask for the NEXT step
    else:
        keep M and record rejection reason / bound gap
```

算法无须改变 SAM epsilon、g2 裁剪、Nesterov、BN 首遍状态提交或更新 inactive master 的语义。
ρ=0 时 ε0=0，可用单遍任务梯度，准则自然退化到任务损失上界选择。

## 7. 严格计算预算和可扩展性

设 P 为全参数数，每次访问 K≤1024 块、每块6候选，更新间隔 τ≥20。
缓存 t_base=v_old−ε0、b、||t_base||²、bᵀt_base，以及
||P_M(b+βt_base)||²。新候选只改变局部四坐标：

```text
||t_base+q||² = ||t_base||²+2 t_baseᵀq+||q||²。
```

新投影范数也用删除/添加的四坐标贡献更新；累计联合修改仍更新同一个全局缓存。
这里是在 O(4) 成本下更新全网络统计，不是把全网络问题变成块损失。

| 项目 | 默认训练兼容版本额外成本 |
| --- | --- |
| 额外网络操作 | 每步0 F /0 B；不评估六次网络 |
| 额外梯度/HVP | 0；只使用已计算的一个任务梯度端点 |
| 更新帧算术 | O(P)+O(6K·4)，无矩阵分解 |
| 平均每步 | 上式除以τ |
| 状态 | O(P)+O(K)；至多约2个额外FP32向量，可流式归约 |
| P≈1100万示例 | 向量预算约88 MB，尚未GPU测量；不构造H或块Hessian |

eval 目标或缺失完整 g2 的 PureMasked 版本，每更新帧需1 F /1 B；
τ=20 时反向次数增量约2.5%。必须单独列为该版本成本，不声称仍零反向。
对 β 的科研校准是有限 snapshot 的独立验证预算，不允许变成逐块 HVP。
O(P) 归约、显存带宽、packing 和 host/device 同步仍须 profiling；选择逻辑应批处理。
维持严格2:4保留了潜在稀疏kernel价值，但 dense master/STE 训练不自动加速。

## 8. 改善保证与无法保证的情形

**可以保证**：在固定 L、有效 β、正确完整端点梯度、比较同一 W1、足够数值裕量时，
每次最终被接受的切换降低 Fρ 至少 margin。

**无法保证**：β仅经验估计；目标模式/BN/RNG不一致；新增坐标梯度不可用；
把局部或平均曲率当成全邻域界；ReLU切换导致假设失败；只在一个训练 batch 上验收
却声称期望风险或泛化下降。也不保证必有可接受候选或找到全局最佳 Mask。

即使界有效，也可能因 β 太大而从不改变 Mask。该保守性是主要实际风险。
本方案对“中心相同、梯度接近零、但新旧支撑曲率不同”的情形分辨力有限；
它没有神奇地补齐二阶信息，只是在拒绝未能证明安全的动作。

## 9. 验证计划和失败判据（只设计，不执行）

1. 用已知光滑常数的四维/双块二次函数，验证 U≥F、LB≤F、接受蕴含真实改善，
   包含不定H、g≠0、q=0但支撑不同，以及优化器漂移。
2. 构造低估β的反例，确认不再具有证书；加入激活边界例子检验假设边界。
3. GPU批准后，在训练内部固定校准 snapshot 比较上界间隙、接受率、错误接受率、
   ΔL/ΔS/ΔF排序、mask翻转和额外耗时；验证集单独，测试集不选Mask。
4. 与Magnitude、Taylor-g1/g2、同预算普通局部代理比较，不能只报告被接受的有利样本。

若有效界导致接受率接近零且不能改善目标，或经验界反复错误接受，停止将其作为
主选择器；可以保留为审阅后用于其他策略的安全控制，但不能把组合包装为新理论。
真实网络的有限步内层值只提供可行下界/参考proxy，不是F的精确参考。

## 10. 已有理论与潜在贡献

Smoothness/Cauchy不等式和上下界接受逻辑是已有优化工具。
这里独立推导的是：从同一 SAM 端点对**新中心、新支撑、实际优化器漂移**形成
廉价全网络鲁棒切换准则。是否属于新贡献须与已有 sharpness-aware pruning
工作逐项比较，不能仅因写出一个界就宣布创新。
目标基准见 [SAM](https://arxiv.org/abs/2010.01412)；已有相关方向见
[AdaSAP](https://arxiv.org/abs/2306.14306)。本方案未宣称优于这些工作。

未来集成位置为独立的 mask-scoring/acceptance policy；当前代码只保留通用诊断，
不得在未经审阅时替换生产 Magnitude/Taylor 调度。
