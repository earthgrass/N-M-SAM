# HQ-APBR V1：数学定义与仓库语义审计

**范围与状态。** 本文只定义训练期历史二阶可补偿性目标、固定全局 L2 扰动以及训练后静态 Final-Q Mask。它不定义新的权重补偿器，也不主张准确率收益。下述恒等式在明确的正定二次目标和精确算术下成立；CIFAR-100 上的效果仍待实验。HQ-APBR 是独立方法，不能用原 APBR-SAM 的能量反馈或椭球约束结果冒充验证。

## 1. 符号、张量与时间

- \(e=0,\ldots,T-1\) 表示 Epoch；第 \(e\) 轮 mini-batch 使用此前已提交的 \(s^{[e-1]}\)，其中 \(s^{[-1]}=0\)。第 \(e\) 轮所有优化器更新成功完成后，才在固定的模型和训练侧校准输入上计算 \(Q_e\) 并提交 \(s^{[e]}\)。因此不能使用未来激活来构造本轮扰动。
- \(\ell\) 为 eligible Conv2d；若有 grouped Conv2d，再以 \(h\) 指其连接组。该组权重展平为 \(W_{\ell h}\in\mathbb R^{m_{\ell h}\times d_{\ell h}}\)，校准输入展开为 \(X_{\ell h,e}\in\mathbb R^{d_{\ell h}\times n_{\ell h}}\)。列是一个图像的一个输出空间位置，行是 OIHW 展平后对应的输入通道、kernel 行和 kernel 列。
- \(o\) 为输出行，\(j\) 为输入特征坐标；权重坐标记 \(i=(\ell,h,o,j)\)。\(E\) 是所有 eligible 权重坐标组成的集合，\(D=|E|\)。当前 CIFAR-100 ResNet-18 实际代码检查得到 19 层、\(D=11{,}157{,}504\)，全模型可训练参数 11,220,132 个，非 eligible 可训练坐标 62,628 个。不能把这 62,628 个坐标从 SAM 全局扰动分母中去掉。
- \(H_{\ell h,e}\) 是输入重建 Gram；\(Q_{\ell h,e}\) 是带阻尼的逆 Gram；\(u_{\ell h,o,j,e}\) 是目标 log 尺度；\(s_{\ell h,o,j}^{[e]}\) 是持久 log 几何状态；\(b_i=\exp(2s_i)\) 是扰动方向的正尺度。\(\rho>0\) 是全模型共享的 SAM L2 半径，\(\eta\in[0,1]\) 是历史更新系数。

当前 [OIHW 展平与 unfold 实现](../../APBR-post-training-benchmark/vpr_sam_research/experiments/post_training_nm_benchmark/baseline_adapters/common/conv2d.py)、[eligible 范围](../../APBR-post-training-benchmark/vpr_sam_research/experiments/post_training_nm_benchmark/benchmark/nm.py)、[100 类模型入口](../../APBR-post-training-benchmark/vpr_sam_research/experiments/post_training_nm_benchmark/baseline_adapters/cifar100_model.py)分别规定上述映射。当前 19 层都是 `groups=1`，但实现仍应按真实 `groups` 分配 Hessian，不可把不同连接组混成一个矩阵。

## 2. 激活重建目标与 SparseGPT 口径

取定义明确的 HQ 目标

\[
 H_{\ell h,e}=\frac{1}{n_{\ell h}}X_{\ell h,e}X_{\ell h,e}^{\top},\qquad
 H^d_{\ell h,e}=H_{\ell h,e}+\lambda_{\ell h,e}I,\qquad
 Q_{\ell h,e}=(H^d_{\ell h,e})^{-1}.
\]

若输出行的局部物理误差为 \(\frac12\|\Delta wX\|_2^2/n\)，则其二次矩阵是 \(H\)。\(Q\) 来自**激活/层输出重建近似**，不是交叉熵 Hessian，也不是网络全局 Hessian。要求阻尼后严格正定，且所有被使用的 \(Q_{jj}>0\) 且有限。数值实现允许矩阵乘法造成的微小非对称舍入，先按预设容差检查，再显式取 \((H+H^\top)/2\)，并记录实际分解的矩阵和最大修正幅度；明显非对称输入会被拒绝。

仓库的 [SparseGPTAdapter](../../APBR-post-training-benchmark/vpr_sam_research/experiments/post_training_nm_benchmark/baseline_adapters/sparsegpt/adapter.py) 在 `add_batch` 中累计 \(2XX^\top\)，`hessians()` 再除以展开后的 patch 向量总数，所以现有输出是 \(H^{SG}=2H\)。`sparsegpt_matrix` 先把死特征的零对角设为 1，并把其列权重设零，然后加相对阻尼 \(\delta^{SG}=p\,\operatorname{mean}\operatorname{diag}(H^{SG}_{\rm after\ dead})\)，冻结配置中常用 \(p=0.01\)。其后使用 Cholesky、`cholesky_inverse` 与逆矩阵的上三角 Cholesky 因子；不能把这一过程称为通用逆矩阵调用。

**常数尺度换算。** 当不存在死特征特殊处理，并保持相对阻尼规则时，令 \(\lambda=p\operatorname{mean}\operatorname{diag}(H)\)，便有

\[
 H^{SG}+\delta^{SG}I=2(H+\lambda I),\quad
 Q^{SG}=\tfrac12 Q,\quad
 \tfrac12\log Q^{SG}_{jj}=\tfrac12\log Q_{jj}-\tfrac12\log2.
\]

此时一次**全局**中心化会消去统一的 \(-\frac12\log2\)；Final-Q 静态分数仅被统一乘以 \(\sqrt2\)，2:4 组内排序不变。但不能只把 \(H^{SG}\) 除以 2 而保持原绝对阻尼。若各层采用彼此不同的常数缩放，一次全局中心化无法消去层间差异；这会改变跨层扰动预算。若某层有死特征或其替代对角，以上统一尺度关系也不再自动成立；正式 HQ 实现须先决定并测试与现有处理相同的死特征规范，逐层记录其数量与最终缩放。

对 grouped Conv，\(X_{\ell h,e}\) 与 \(Q_{\ell h,e}\) 分组独立；每组 \(Q\) 仅供该组的输出行使用。对当前非 grouped 模型，一个输入特征的 \(Q_{jj}\) 在同层所有输出行间共享。

## 3. 单坐标 OBS 删除代价的适用条件

在固定的正定 \(H^d\) 下，给定**当前**完整输出行 \(w\)，求

\[
 \min_{\Delta w}\ \frac12\Delta w^{\top}H^d\Delta w
 \quad\text{s.t.}\quad e_i^{\top}\Delta w=-w_i.
\]

拉格朗日条件 \(H^d\Delta w+\alpha e_i=0\) 给出 \(\Delta w=-\alpha Qe_i\)。代入约束，\(\alpha=w_i/Q_{ii}\)，所以

\[
 \Delta w^*=-\frac{w_i}{Q_{ii}}Q_{:i},\qquad
 C_i=\frac12(\Delta w^*)^{\top}H^d\Delta w^*
     =\frac{w_i^2}{2Q_{ii}}.
\]

这是**指定带阻尼二次目标**中的单坐标精确成本。若已有固定删除集合，其他仅存活坐标可调整，应在当前自由子空间构造相应的条件逆矩阵，而不能始终使用最初的完整 \(Q\)。若同时删除集合 \(S\)，精确局部成本是 \(\frac12w_S^\top(Q_{SS})^{-1}w_S\)（其余坐标自由）；它通常不是各个 \(C_i\) 的和。因而单坐标评分选 2:4 Mask 不保证四元组内联合最优，更不保证全网络最优。原生 SparseGPT 在补偿传播期间动态挑选，不等于本任务的静态 Final-Q 评分。

## 4. 目标尺度、全局中心化及广播

对每个 eligible 坐标 \(i=(\ell,h,o,j)\)，定义

\[
 u_{i,e}=\tfrac12\log Q_{\ell h,e,jj},\qquad
 \bar u_e=\frac1D\sum_{i\in E}u_{i,e},\qquad
 \widehat u_{i,e}=u_{i,e}-\bar u_e.
\]

中心化按**权重坐标个数**加权。为了省内存，可以按共享特征汇总：

\[
 \bar u_e=\frac{1}{2D}\sum_{\ell,h}m_{\ell h}
 \sum_{j=1}^{d_{\ell h}}\log Q_{\ell h,e,jj},\qquad
 D=\sum_{\ell,h}m_{\ell h}d_{\ell h}.
\]

不能把各层均值简单平均；输出通道数较多、输入维度较大的层应按真实坐标数贡献。精确算术下 \(\sum_{i\in E}\widehat u_{i,e}=0\)。若 \(G_e=\exp(D^{-1}\sum_{i\in E}\log Q_{e,ii})\)，其中每个共享对角按输出行重复计数，则 \(\exp(2\widehat u_{i,e})=Q_{e,ii}/G_e\)，即中心化不改变组内尺度比。因为非 eligible 坐标的 \(b_i\) 被固定为 1，这一全局中心化还固定了 eligible 与非 eligible 扰动方向的相对尺度；不能把它任意改成逐层中心化。

**关键结构结果。** \(Q_{\ell h,e,jj}\) 不含输出行 \(o\)。若所有行从 \(s^{[-1]}=0\) 起步，且只执行本文下一节的统一 EMA，则对任何同组输出行 \(o,o'\)，始终有 \(s^{[e]}_{\ell h,o,j}=s^{[e]}_{\ell h,o',j}\)。因此 HQ 的持久状态是**层/连接组的输入特征尺度，广播到逐权重坐标**；并非每个权重独立学习的几何值。梯度 \(g_{o,j}\) 仍因输出行而异，故实际扰动并非行间相同。若论文声称 HQ 学到了逐输出行独立的二阶几何，便与此 V1 公式不符。

## 5. 历史状态与乘积解释

初始化 \(s^{[-1]}=0\)，每轮成功完成后执行

\[
 s_i^{[e]}=(1-\eta)s_i^{[e-1]}+\eta\widehat u_{i,e}.
\]

展开递推得到

\[
 s_i^{[e]}=\eta\sum_{k=0}^{e}(1-\eta)^{e-k}\widehat u_{i,k}.
\]

令 \(G_k\) 为上述在全部 eligible 坐标上计算的 \(Q_k\) 对角几何均值，则

\[
 b_i^{[e]}=\exp(2s_i^{[e]})=
 \prod_{k=0}^{e}\left(\frac{Q_{k,ii}}{G_k}\right)^{\eta(1-\eta)^{e-k}}.
\]

每个 \(\widehat u_e\) 的 eligible 坐标和为零，所以精确算术下任意 \(e\) 都有 \(\sum_{i\in E}s_i^{[e]}=0\)，即 eligible 几何尺度的乘积为 1。实际浮点实现须检查并报告漂移，不可悄悄添加未定义的重中心化。\(0<\eta<1\) 才保留对过去 \(Q\) 的指数记忆；\(\eta=1\) 为仅使用最近目标的 Direct-Q；\(\eta=0\) 始终为 \(s=0\)，等于普通 SAM 的几何控制，不得标为 Historical-Q。此更新**不含**原始 APBR 的 \(p_i=\epsilon_i^2/\sum_j\epsilon_j^2\) 能量反馈，也不学习额外损失。

## 6. 固定全局 L2 半径的两步优化

令 \(P\) 为当前模型全部可训练参数坐标，包含首层、分类头、BN affine、bias 等非 eligible 坐标。第 \(e\) 轮任一 mini-batch 的第一遍梯度是 \(g_{1,i}\)。定义

\[
 \widetilde s_i=\begin{cases}s_i^{[e-1]},&i\in E,\\0,&i\in P\setminus E,\end{cases}
 \quad v_i=\exp(2\widetilde s_i)g_{1,i},\quad
 \epsilon_i=\rho\frac{v_i}{\sqrt{\sum_{j\in P}v_j^2}}.
\]

若 \(v\ne0\)，则精确算术下 \(\|\epsilon\|_2^2=\rho^2(\sum_i v_i^2)/(\sum_i v_i^2)=\rho^2\)。若所有有效梯度为零，定义 \(\epsilon=0\) 并记录 zero-step；不能声称半径等于 \(\rho\)。缺失的 `p.grad` 在扰动计算中作为零贡献，但不得改变优化器对 `None` 梯度的原有语义。Geometry 在整个 Epoch 固定，并从 autograd 图中分离。

这是**固定全模型 L2 球面**，与原始 APBR 的 \(\epsilon^{APBR}=\rho A^2g/\|Ag\|_2\) 椭球边界不同。现有 `fixed_l2` 消融在 FP32 写回后再次做全局 L2 缩放；HQ 可借鉴其稳定数值策略，但不得继承其能量反馈更新。对数域求方向、FP64 求范数、FP32 写回后核验相对半径误差是实现建议；数值容差必须预注册。不能逐层单独归一化，也不能只对 \(E\) 归一化。当前 [Stepper](../../APBR-post-training-benchmark/vpr_sam_research/src/vpr_sam/optim/step.py) 第一遍在 train 模式反向传播，第二遍通过 `freeze_bn` 阻止 BN running statistics 二次更新，参数在第二遍后恢复再由 SGD 更新。HQ 训练适配器必须保持该时序；Epoch 后校准须用 eval 模式，禁止改变 BN buffers。

**Direct-Q 等价的精确范围。** 当 \(\eta=1\)，eligible 上 \(\exp(2s_i)=Q_{ii}/G_e\)，非 eligible 上恒为 1，因此

\[
 v_i=\begin{cases}(Q_{ii}/G_e)g_i,&i\in E,\\g_i,&i\notin E.\end{cases}
\]

若仅在 eligible 子空间归一化，公共的 \(1/G_e\) 会抵消，得到 \(\rho\operatorname{diag}(Q)g_E/\|\operatorname{diag}(Q)g_E\|_2\)。但全模型归一化且非 eligible 梯度非零时，该等式一般**不成立**：非 eligible 坐标相对 eligible 坐标的尺度不同。只有参照 Direct-Q 使用相同的 eligible 中心化、非 eligible 等于 1 的规范，才有全参数等价。另一个充分条件是所有非 eligible 梯度恰好为零。

## 7. 训练后的静态 Final-Q Mask

冻结 Epoch \(T\) 的 Dense Parent 后，在固定训练侧校准输入上以相同映射、阻尼和死特征规则重新计算 \(Q_T\)。定义

\[
 I_{\ell h,o,j}^{\rm final}=
 \frac{|w_{\ell h,o,j,T}|}{\sqrt{Q_{\ell h,T,jj}}}.
\]

每个输出行按 OIHW 连续展平，每个完整的 4 元组保留 \(I\) 最大的 2 个；不完整尾组保持稠密；完全相同的分数使用已有 stable tie 规则。首层 Conv、末层 Linear、BN affine、bias 等原协议排除项不生成剪枝 Mask。按现有模型结构预计 5,578,752 个删除位置，但运行时要重新核验。\(I^2/2\) 恰为第 3 节的单坐标带阻尼 OBS 成本；它**没有**评估两个同时删除位置的联合成本。

该 Mask 在冻结 Dense Parent 上**一次性**计算。原生 SparseGPT 的 `sparsegpt_matrix` 则在每个块的当前补偿权重上动态计算 \(w^2/U_{jj}^2\)、选删并立即传播误差。因此静态 Final-Q Mask、原生 SparseGPT 动态 Mask、固定 Mask OBS 补偿是三个不同操作，不能互称等价。本文的最终 Mask 不直接使用历史 \(s\)；HQ 相对 SAM 的任何收益必须通过训练改变最终权重/激活后由公平对照验证，不能仅凭 Mask 公式推断。

## 8. 已建立事实与尚待证实的问题

**数学已建立：** 带阻尼正定二次目标中的单坐标 OBS 代价；全局中心化和 EMA 的零和/乘积恒等式；非零方向上的固定 L2 扰动半径；共享 \(Q\) 到输出行的广播结构；静态 2:4 单坐标评分的含义。

**实现仍需验证：** 每层卷积展开与真实输出的等价、死特征约定、Cholesky 数值、FP32 写回半径、Epoch 边界/BN/SGD 状态、resume RNG、Final-Q Mask 哈希以及 GPU 耗时和显存。当前原 APBR 与 SparseGPT 的通过记录不自动构成 HQ 实现通过。

**经验上未知：** HQ 是否提高 Dense Accuracy、Mask-only Accuracy、OBS 补偿后的 Accuracy 或计算效率。完整 \(Q_{ii}\) 允许其他所有坐标自由调节，未必能准确预测 2:4 多坐标共同删除后的可补偿性；其大小还会受阻尼与校准样本影响。共享行尺度会降低与原 APBR 逐权重状态的表达能力。任何收益及其因果归因都必须留给同 Seed、同协议的交叉对照。

**需在实现前冻结的协议细节：** 每个 Epoch 校准图片/patch 数量与顺序；死特征替代规则；用于 HQ 的 Hessian 采用 \(H\) 还是仓库 \(2H\) 并同步阻尼；\(\eta\)、刷新频率及 rho；FP32/FP64、AMP 与半径容差；异常时是否保留上一状态而停止 run；Final-Q 的静态 Mask 与固定 Mask OBS 的补偿入口。这些选择不能根据 Final Test 调整。
