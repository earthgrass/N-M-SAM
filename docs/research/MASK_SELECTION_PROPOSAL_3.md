# 方案三：直接使用鲁棒反馈的合法 Mask 策略 Bandit

状态：DESIGN_ONLY；未实现新的选择器或运行实验。
本方案不构造剪枝 Taylor 分数，也不近似整个 Hessian。它把严格合法的全网
Mask 提案视为动作，用已经执行的 SAM 计算提供目标反馈，通过低成本在线决策
学习哪个提案值得使用。保证类型是条件期望 regret，不是每次切换的安全下降。

## 1. 从离散目标到概率决策：严格等价的起点

对固定 W、B、s，原目标为：

```text
min_{M∈合法2:4} Fρ(P_M W,M)，
Fρ(Z,M)=max_{P_Mδ=δ, ||δ||₂≤ρ} L(Z+δ)。
```

给定有限合法候选集 Ω={M^1,...,M^A}，有严格等式：

```text
min_{a=1..A} Fρ(P_Ma W,M^a)
= min_{p∈概率单纯形} Σ_a p_a Fρ(P_Ma W,M^a)。
```

原因是右边对 p 线性，最小值在某个顶点取得。没有把二值 Mask 替换成
概率平均 Mask；实际每次采样一个合法二值 M，所有组仍恰有2个保留位。
Ω 小于全部 Mask 集合会产生未知的候选类差距，这不是全局 Mask 最优解保证。

训练中 W/B/s 持续改变，因此在线问题是对当时上下文定义的动作损失序列，
不能把静态等价关系直接升级成整个训练轨迹的全局 min-max 收敛证明。

## 2. 大网络中动作如何定义

每 τ 个完整 optimizer step 安排一次选择帧；动作仅在 step 之间提交。
预先用与当前动作抽样独立的 coverage schedule 选 K≤1024 个完整块。

- 动作0：保持旧 M 不变，作为明确的 no-change 比较对象。
- 其余6个动作：对每个选中块，按当前 master 幅值稳定排序形成四个坐标的排列，
  然后分别保留六个合法的 rank-pair。一个动作共同修改这批 K 块，其余块不变。
- 可加入预声明的总 ||q||/翻转数预算；动作生成器超预算时按固定规则减少修改或
  返回旧 Mask。规则对所有反事实动作都要事先定义，不能看见当前损失后改规则。

这是最多7个**全网络合法提案生成策略**，不是独立给 K 块分配各6个概率再猜联合损失。
单次观察的全网络损失已经包含所选块间耦合，无需假设块间可加。
代价是候选族表达能力有限：不能覆盖全部6^K组合，可能遗漏真正有利的异质模式。
重复动作可以保留为重复策略，避免悄悄改变固定动作空间及 regret 比较对象。
幅值排序只生成合法动作，不是最终优化评分；选择依据是实际鲁棒反馈。

## 3. 反馈必须对应哪个目标

第 r 帧在固定 W_r 和旧状态下生成所有反事实提案，再抽样 a_r。
下一步在 M^{a_r} 中正常完成两遍 SAM，采集该步**优化器更新前**的状态/梯度。
定义真实动作损失 f_r(a)=Fρ(P_Ma W_r,M^a;B_r,s_r)。

只能用这一帧新 Mask 下的端点，不能沿用旧 Mask 的 ε。
默认 train_fixed_batch 状态与训练兼容；侧向观测不得更新 BN、推进 RNG、
更改 optimizer 或干扰正常两遍更新。

为了让反馈代理来自任务 L 而不是 optimizer J，令：

```text
d_L=P_Ma(G1_train_J−∇_W regularizer)
δ_L=ρ d_L/||d_L||（d_L=0时δ_L=0）
ell_r(a_r)=L(P_Ma W_r+δ_L;B_r,s_r)。
```

identity STE 下得到全有效梯度；PureMasked 也提供所需的 active 投影梯度，
所以构造选中动作的内层方向不需要旧 inactive 坐标梯度。
默认 master L2 的 regularizer 梯度是 λDW；其他正则口径必须另算。
不能拿未修正 g1_J 当 g1_L。

现有 SAM 的 ε_J 若恰等于 δ_L，可直接使用第二遍 task CE；一般需要对 δ_L
增加一次**无反向** functional forward。使用捕获的 W_r/pre-BN/pre-RNG，
不是 optimizer 已更新后的 W，也不是受当前动作影响的后提交 buffers。
这是原目标的可行点值，故 ell_r(a)≤f_r(a)，但不是精确最坏损失。

如果 L 在每个候选中心的 ρ 球附近有统一有效 smoothness 常数 β_r，线性内层方向给出：

```text
Fρ(Z,M) ≤ L(Z)+ρ||P_Mg||+β_rρ²/2
L(Z+δ_L) ≥ L(Z)+ρ||P_Mg||−β_rρ²/2
0 ≤ f_r(a)−ell_r(a) ≤ β_rρ²。
```

该界不要求 q 小，不在旧中心对大 Mask 位移做 Taylor 展开；它要求新中心邻域
的有效 β_r。g投影为0时取δ=0，也有该保守界。
若不愿增加那一次 task forward而使用ε_J，需要额外加入
ρ||P_Mg_L||−g_Lᵀε_J 的角度损失项，不能默认两个反馈相同。

## 4. 在线更新从实际目标反馈推导

假设：动作损失矩阵在抽取当前动作前确定（可依赖历史，但不预知本次随机数）；
0≤ell_r(a)≤f_r(a)≤C，C 是统一有效常数；inner gap≤ε_inner,r。
取 y_r(a)=ell_r(a)/C∈[0,1]，使用 loss 形式的指数权重：

```text
q_r(a)=w_r(a)/Σ_b w_r(b)
p_r(a)=(1−γ)q_r(a)+γ/A
hat_y_r(a)=1[a=a_r] y_r(a_r)/p_r(a_r)
w_{r+1}(a)=w_r(a) exp(−η_exp hat_y_r(a))。
```

条件期望 E[hat_y_r(a)|历史,当前损失矩阵]=y_r(a)，这是重要性加权无偏性。
全信息需要逐个计算候选损失，而此处只观察实际抽样的动作。

下面给出所需 regret 界的推导，而非仅引用其名字。
用 exp(−x)≤1−x+x²/2、log(1+x)≤x 和权重势函数，有对每个固定策略 a：

```text
Σ_r q_rᵀhat_y_r − Σ_r hat_y_r(a)
≤ log(A)/η_exp + (η_exp/2)Σ_r Σ_b q_r(b)hat_y_r(b)²。
```

条件期望二次项为 Σ_b q_r(b)y_r(b)²/p_r(b)≤A/(1−γ)；
混合 p 与 q 的损失差每帧至多 γ。因此 N 帧后：

```text
E Σ_r ell_r(a_r) − E Σ_r ell_r(a)
≤ C[log(A)/η_exp + η_exp N A/(2(1−γ)) + γN] = R_N。
```

取 γ=min(1/2,sqrt(A log(A)/N))、
η_exp=sqrt(2(1−γ)log(A)/(NA))，得到 R_N=O(C sqrt(N A log A))。
这是已有指数权重/bandit 原理的条件推导，不是原创 regret 理论。

可行点代理的误差是单侧的，所以进一步严格有：

```text
E Σ_r f_r(a_r) ≤ min_a E Σ_r f_r(a) + R_N + Σ_r ε_inner,r。
```

注意是“固定策略在已实现的上下文序列上的期望比较”，不是 E[min_a 实际损失]，
也不是反事实执行另一整个训练过程后的比较。特别是动作0提供 no-change comparator，
但 W_r 本身取决于过去动作，不能把这个界当成全训练轨迹的 policy regret。
它没有承诺每帧目标下降或最后准确率提高。

## 5. 假设为何可能不成立

CE 在无限制权重空间没有天然统一上界 C。若所有相关 logits 有已知界 |logit|≤S，
可取 C=log(类别数)+2S；但现有训练没有自动提供这种全轨迹证书。
随意裁剪 CE、使用随时间变化的归一化范围，会改变所优化的损失，不能继续声称
上式对应原始未裁剪 F。若仅采用经验归一化，该版本是启发式，须明确失去此保证。

真实网络也可能没有有效统一 β_r；inner gap 可能大过 Mask 间差异。
若 ε_inner,r 无界，最多能证明针对所定义 one-step proxy 的 regret，不能转移到真 F。
多步内层会增加成本；必须作为预声明预算版本，而非偷偷对某些有利动作加算力。

τ步内的全部平均奖励不能直接套上非预知损失假设，因为这些中间 W 已受本次动作影响。
本设计只使用动作提交后的**第一步、更新前**的观测作为该帧反馈，其余步不计入证明。
探索随机数必须与 sampler/augmentation/dropout 的 RNG 流分离；Mask不能在g1/g2中途改变。

## 6. 可执行设计伪代码（未实现）

```text
initialize standard legal masks; keep a private bandit RNG and A<=7 log-weights
after each declared full-step selection frame:
    fix W and old state, choose a coverage block pool independent of action draw
    define no-change + six legal proposal generators and any q/flip constraint
    sample action using logged p_r; commit its legal mask for the NEXT step
before that next step:
    cache pre-step batch/state/master so counterfactual target is well-defined
    execute the existing two-pass Sparse-SAM with this mask frozen
    recover allowed task gradient from raw first gradient under regularizer contract
    construct supported task delta_L; evaluate feasible CE proxy (at most one extra F)
    update log-weights by the importance-weighted observed loss
hold this mask through the declared interval; do not relabel later-step rewards
record arm/probability/context/inner-gap assumptions and restore all side-probe state
```

初始、有限步和高概率版本须在将来审阅后选择；上面不是一段已交付的算法实现。
数值上需要 log-sum-exp、概率下限与稳定记录，不能让 underflow 把探索概率置零。

## 7. 硬成本、状态与规模

以默认 train_fixed_batch 任务反馈为准，标准 Sparse-SAM 仍为每步2 F/2 B：

| 项目 | 额外成本 |
| --- | --- |
| 每选择帧 | 最多1个无梯度F，0个B，0个HVP；只观察选中动作 |
| 平均每步 | ≤1/τ 个F、0个B；τ=50时前向次数约增加1% |
| 梯度 | 复用当前选中Mask的已有g1；不算6/7次候选梯度 |
| 算术 | O(P)的投影/统计或流式处理 + O(K·4)合法提案生成 + O(A)更新 |
| 状态 | A≤7的概率/log-weights、O(K)块索引；侧probe复用trace并需O(P)临时向量/固定上下文 |
| 额外显存估计 | 约1–2个FP32向量（P≈1100万时44–88MB）+无梯度F的临时激活；需GPU实测 |

eval_fixed_buffers 反馈不能用训练g1，需每帧额外1 B/2 F；τ=50时约1%的反向
次数和2%的前向次数增量。这是独立成本版本，不能混称默认零额外反向。
若反馈做m步内层，默认已有g1后还需m−1个额外任务梯度，成本须除以τ后披露。

K=1024、τ=20可在约数千选择帧内访问数百万块，不需要储存每个块的6个统计量，
也不枚举6^K个联合mask。候选族的有限表达力、访问覆盖率和可能的大q是代价。
所有 sampled masks严格2:4；推理native稀疏kernel仍需布局/打包验证，STE训练没有
自动稀疏加速。F/B次数小不代表wall-time已证廉价，O(P)带宽和额外F须实测。

## 8. 与已有Baseline及前两方案不同

Magnitude/Taylor依据当前权重/梯度局部量确定性选择；这里优化实际观察的
全网络鲁棒损失反馈分布。没有 residual-Hessian 模型，也没有β上界逐候选接受证书。
与方案一的二次模型、方案二的安全上界机制不同，本方案允许带预算的探索，
以累计反馈改变动作概率，容忍单次不良切换。
Magnitude可以只是候选动作之一；若最终仅回到Magnitude或no-change，必须报告，
不能把在线采样本身当作科学收益。

## 9. 验证计划与失败判据（只设计）

未来批准后依次检查：已知有界动作表的无偏估计/regret→已知精确F的二次函数
及内层gap→新工程固定snapshot→短期在线轨迹。对batch噪声、inner gap、
探索概率、q预算、动作覆盖和数据模式作预声明对照，不用测试集选arm或超参数。

必须分开记录：proxy regret、可算参考F regret、实际Mask变更、完整训练目标、
验证性能与总计算量；冻结snapshot的收益不能替代在线训练轨迹收益。

失败判据：feedback无法分辨动作；重要性估计方差/探索代价过高；inner gap掩盖
候选差异；动作族表达力不足；短窗口选择看似变好却损害验证表现；
公平成本下不能超越Magnitude/Taylor或仅选择no-change。任何一项都可否定继续投入。
没有C/β/非预知条件时，不能把上述条件定理拿来充当真实ResNet保证。

## 10. 先行理论和潜在贡献

Bandit/指数权重来自已有理论，见
[Auer等：The Nonstochastic Multiarmed Bandit Problem](https://epubs.siam.org/doi/10.1137/S0097539701398375)。
动态稀疏结构选择已有先行工作，例如 [RigL](https://proceedings.mlr.press/v119/evci20a.html)。
本设计可能有价值的部分是严格2:4、全局SAM任务反馈、合法动作族和训练时点的
低成本结合；是否新颖、是否真的有效均待独立文献审计与实验。
不得把采用Exp3或产生随机Mask本身包装为新理论贡献。

未来实现应为独立selection controller，调用通用state/registry/probe接口；
不改变官方Dense-SAM，也不在当前代码中启用此策略。
