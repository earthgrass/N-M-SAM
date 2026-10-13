# 方案一：两梯度对称割线模型上的全网络鲁棒结构选择

状态：DESIGN_ONLY。未实现选择器，未运行科研实验，未确认新颖性。
本方案用 SAM 的两个端点梯度建立低秩二次模型，在候选新支撑中求全局
参数球的鲁棒代理；不把逐元素 g2 分数当成最终目标。

## 1. 从最终目标开始

固定输入 B、状态 s、参数顺序和损失口径，令 L 是无 L2 的全网络任务损失。
P_M 在稀疏权重上是合法 2:4 投影，在 bias/BN/排除层上是单位映射：

```text
Z=P_M W
Fρ(Z,M)=max { L(Z+δ): P_M δ=δ, ||δ||₂≤ρ }
目标：在给定 W 下，选择合法 M 以降低 Fρ(P_M W,M)。
```

限制候选集合或使用局部更新，不改变评价函数，只放弃对全部 Mask 的全局最优保证。
以下结论针对同一个固定 L。用一个 minibatch 选择 Mask 不自动降低全数据目标；
E_B maxδ L_B 一般不等于 maxδ E_B L_B。推广到期望目标需要独立校准和统计验证，
不能把批次改进当作测试泛化证明。

## 2. 数据来源及明确假设

默认目标状态为 train_fixed_batch，以复用训练端点。默认 master-space L2、
identity STE、同一 materialized batch、相同 RNG、相同可微任务函数。
若 STE 前向为 Z=P_M W、替代 Jacobian 恰为 I，则端点梯度的**数值**满足：

```text
G1_STE_J = g1_L + λ D W0
G2_STE_J = g2_L + λ D (W0+ε0)
g1_L = G1_STE_J − λ D W0
g2_L = G2_STE_J − λ D (W0+ε0)
```

D 是明确的正则化坐标集合，不能凭名字猜 BN 排除规则。必须使用未裁剪的
raw g1/g2。上述识别由链式法则成立，**不等于 STE 图上的二阶导数是真 Hessian**。
有效空间诊断仍用纯 functional L(Z)。effective-space L2 需要另减其真实正则梯度；
PureMasked 没有 inactive 坐标梯度，不能直接使用此零额外反向版本。
其他 surrogate、状态依赖前向、跨框架随机差异均需重新核验这一等式。

训练 ε0 可以由 J 的梯度构造；本方案只把它当作一个已观测的合法方向，
不把它叫作任务 Fρ 的精确内层最优。令 α=||ε0||₂，α>0 时 u=ε0/α。
若 α=0，报告 undefined_direction，使用不改 Mask 或另一个已审阅方案，不能除零。

还需区分：

- **严格推导假设**：L 在覆盖两个端点及候选球的区域中二阶可微，H 对称；
  误差界进一步要求 Hessian 有已知 Lipschitz 常数 K_H。
- **待验证模型假设**：未观测方向的残差曲率足够小，低秩模型能保持候选排序。
- **未知量**：残差算子范数界不是由一次 HVP 或两个梯度自动得到的。

ResNet ReLU 跨激活边界时不一定满足这些光滑假设，不能把条件定理无条件用于真实网络。

## 3. 端点信息如何变成模型

由沿参数路径对梯度积分，严格地有：

```text
h=(g2_L−g1_L)/α = [∫₀¹ H(Z0+t ε0)dt]u。
若 H 是 K_H-Lipschitz，||h−H(Z0)u||≤K_H α/2。
```

这是路径平均曲率。定义 c=uᵀh，构造：

```text
Hhat = h uᵀ + u hᵀ − c u uᵀ。
Hhatᵀ=Hhat；Hhat u=h；rank(Hhat)≤2。
```

为什么是这一形式？在以 u 为第一基向量的正交坐标中，约束 A=Aᵀ、Au=h
固定第一行和第一列，剩余子矩阵自由。令剩余块为零唯一最小化 ||A||_F，
得到上述式子。这是已有对称割线/最小改变量思想，不是本课题发明了新 Hessian。
它没有从一个方向“恢复完整 Hessian”。

若 h_exact=H(Z0)u，Q=I−uuᵀ，则严格分解为：

```text
H(Z0)=A(h_exact)+R，R=Q H(Z0) Q，
A(v)=v uᵀ+u vᵀ−(uᵀv)uuᵀ。
```

因此 rᵀHr 缺失问题没有消失：Hhat 只是显式地把剩余块设为零。
即使真实 H 半正定，Hhat 也可能不定；其负特征值不能当成网络真实负曲率证据。

## 4. 从二次模型推导 Mask 准则

在旧中心 Z0 用同一个模型，同时展开中心变化和新支撑的内层扰动：

```text
Qhatρ(v,M') = L(Z0)+gᵀv+½vᵀHhat v
             + max_{P_M'δ=δ, ||δ||≤ρ}
                 [ (g+Hhat v)ᵀδ+½δᵀHhatδ ]，g=g1_L。
选择准则：在合法候选中最小化 Qhatρ(v,M')，而非最小化普通 ΔL2。
```

这保留了中心损失、候选梯度变化、新扰动支撑和模型内的跨块耦合。
不要求 g≈0，不用 λmax 无条件代替 sharpness。

**内层最多三维，仍是全网络目标。** 令
V=span{P_M'g, P_M'u, P_M'h}。对任何 δ∈S_M'，正交拆成 δ_V+δ_⊥。
线性项对 δ_⊥ 为零，Hhat δ_⊥=0，且 ||δ_V||≤||δ||。
所以存在一个内层最优解位于 V；限制到 V 不损失这个代理的最优值。
这是对 Hhat 的严格降维，绝非真实网络 H 的三维充分表示。

构造三个投影向量的 3×3 Gram 矩阵，建立正交基，求至多三维的球约束二次最大化。
支持改变时只更新被改坐标的 Gram 贡献；不能只在四元素块中分配一个独立 ρ。

## 5. 误差、证书及改善条件

假设 ||R||op≤β_R、||h−H(Z0)u||≤η，则
||H(Z0)−Hhat||op≤β_R+3η，由 A 的三项范数上界得到。
对中心位移 v，所有允许 δ 的 ||v+δ||≤||v||+ρ，故：

```text
E(v)=½(β_R+3η)(||v||+ρ)² + (K_H/6)(||v||+ρ)³
|Fρ(Z0+v,M')−Qhatρ(v,M')|≤E(v)。
```

最后一步使用 |max f−max q|≤sup|f−q|。这是严格的**条件误差界**。
β_R、K_H 未知时，直接用经验数值替代只能算启发式，不能宣称有改善保证。

未来生产内层不能调用逐候选、逐迭代 GPU 同步的 toy 求解器。应批量求解，
给出代理的上下界 Q_lo、Q_hi。对于约化矩阵 H_r、向量 b_r，任取
λ≥max(0,λmax(H_r)) 且 λI−H_r 正定，内层有对偶上界：

```text
U_inner(λ)=λρ²/2 + ½ b_rᵀ(λI−H_r)⁻¹b_r。
```

任意可行 δ 给出下界；差值是代理内层数值误差证书。可规定最多 32 次批量迭代，
预算内不能缩小间隙则放弃该候选，不冒充精确求解。浮点舍入也需加入 ε_num。

充分接受条件为：

```text
Q_hi(new)+E(v_new)+ε_num < Q_lo(old)−E(v_old)−ε_num−margin。
```

在上述有效界成立时，新 Mask 的真实固定损失鲁棒目标至少降低 margin。
没有有效 β_R/K_H 界时，只能报告代理下降，无法保证 Fρ 下降。

## 6. 必须正确处理优化器更新时点

观察发生在 W0，Mask 只能在完整 step 后、任何其他 Mask 调度前决定。
令 W1 为实际更新后的 master、ΔW=W1−W0、Z0=P_M W0：

```text
v_old=P_M ΔW
v_new=P_M' W1−P_M W0
v_new−v_old=(P_M'−P_M)W1=q_post。
```

比较 Qhatρ(v_new,M') 与 Qhatρ(v_old,M)，避免把 W_post*g_pre 当成同一点展开。
固定的 B/s 必须覆盖这次比较；不能一边更新 BN/RNG 一边声称它是同一个 L。

## 7. 可执行设计伪代码（未实现）

```text
before step: capture fixed B/s and W0/M/version
run the existing Sparse-SAM step with M frozen; obtain raw G1/G2/ε0 and W1
if this is not a mask-update frame: return
recover endpoint task-gradient values under the explicit STE/L2/state contract
if α=0 or state contract fails: keep M
build/cache g,u,h, post-step anchor statistics and global projected Gram terms
choose at most K eligible blocks by a predeclared coverage schedule
for each block, enumerate its six exact legal masks (includes current choice)
    update scalar/Gram statistics for each candidate at W1
    solve global rank-model inner problem in ≤3 dimensions, with budget/certificate
    select the smallest upper score; apply only if its bound beats current lower score
    update accumulated joint mask/statistics; reject if trust region is exceeded
commit one final legal M_next; use it only in the next SAM step
```

逐块贪心只保证已接受局部步骤的条件改善，不保证全部 Mask 的联合最优。
多个接受块必须用累计位移和更新后的全局模型重新评价，不能相加独立分数。

## 8. 硬计算预算

P=全参数数，B_s=稀疏块数，K≤1024 是每次更新访问块数，τ≥20 是更新间隔。

| 项目 | 默认训练兼容版本的额外成本 |
| --- | --- |
| 网络前向/反向 | 每步 0 F / 0 B；只重用已有 SAM 两端点 |
| 梯度/HVP | 无额外梯度，无训练期逐候选 HVP |
| 更新帧算术 | O(P)+O(6K I d³)，d≤3、I≤32；需 GPU 批处理 |
| 平均每步算术 | 上式除以 τ；禁止六候选完整网络前后向 |
| 状态 | O(P)+O(K)；可增量保留约 2 个 FP32 向量，约 8P 字节，不构建 P×P H |
| 参数规模例子 | P≈1100万时两个向量约88 MB；另有已有 trace/state 与小矩阵缓存 |

若选择 eval_fixed_buffers 为实际选择目标，则不能直接复用训练梯度：需要明确的
两端点 eval 计算，每更新帧额外 2 F / 2 B，τ=20 时约5%的反向次数增量。
这不是默认零开销版本。PureMasked 同样需要完整有效空间端点计算。
验证 β_R/误差的少量 HVP 是未来单独批准的诊断预算，不塞入百万块训练循环。
一个 HVP 不能证明残差算子范数上界。

时间/显存尚未实测；O(P) 的带宽和小矩阵同步也可能成为瓶颈。
预算门：按最终 GPU profiling 限制更新帧的摊销开销，例如≤5%，超限就减 K 或停止。
所有提交掩码仍严格2:4；当前 dense master/STE 没有自动硬件加速，native kernel/layout
及重新打包成本必须独立测量。

## 9. 局限、反例与否定条件

- 残差 R 大时，观测方向完全不能解释新激活方向；代理可选出更坏 Mask。
- 低秩补全的人造负曲率、ReLU 切换、BN 小方差、α过小的差分噪声都可能破坏模型。
- q_post 或优化器漂移过大，三阶误差盖过候选差异；小 ρ 不救大 q。
- 同一批次的改善不能自动泛化到另一批次、全数据鲁棒目标或测试精度。
- 有效误差界可能过于保守，导致从不接受更新；这也是失败，不应隐藏。

未来批准后先验证：常/变 Hessian解析函数→少量新工程 checkpoint；比较真实候选
ΔF参考代理排序、regret、与普通ΔL排序的差异，按 q/ρ、漂移、层、激活模式变化分桶。
必须独立报告残差占比、证书覆盖率、接受率、求解间隙与实际成本。
若残差补充没有稳定价值、排序不优于 Magnitude/Taylor，或模型开销超过预算，暂缓。
有限步网络最大化仅作为参考 proxy，不能伪称真实 F 的精确 max。

## 10. 潜在贡献及边界

可研究的是：如何把已有全网络割线信息、候选新支撑和可计算的低维鲁棒问题结合，
并控制 Mask 与 SGD 更新的时点误差；不是发明梯度路径积分、拟牛顿或信赖域理论。
有关对称模型最小 Frobenius 改变量的既有工作见
[Powell 的 symmetric Broyden 讨论](https://optimization-online.org/2010/06/2653/)。
SAM 目标来源见 [SAM 原论文](https://arxiv.org/abs/2010.01412)。新颖性需进一步逐篇核验。

集成位置：未来审阅后可新增独立 scoring policy，组合 registry、raw trace、
functional context；不改 SAM 内核。不把本方案接入当前生产 schedule。
