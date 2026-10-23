# VPR-SAM 数学定义与数值合同

## 冻结算法

在固定的全参数绝对坐标空间 R^d，s_i=log(a_i)，初始化 s=0。
所有可训练 FP32 参数参与，包括 bias 和 BatchNorm affine 参数；不含 BN buffers。
参数映射按对象身份去重并固定，grad=None 不移除坐标。

```text
g1 = gradient L(theta)
epsilon = rho A^2 g1 / ||A g1||_2
g2 = gradient L(theta + epsilon)       # epsilon treated as constant
restore theta
theta_next = base_SGD(theta, g2)
p_i = epsilon_i^2 / sum_j epsilon_j^2
s_next_i = s_i + gamma (1 - d p_i)     # eta_a = d gamma
```

精确算术下 sum(s_next)=sum(s)，因此 prod(a)=1。分母是所有坐标的同一个全局分母。
不使用 EMA、半径裁剪、逐层/逐块守恒、附加损失或更新后体积回正。
几何状态是逐参数张量列表，不构建 d×d 矩阵。

## 固定梯度理论

假设 d>=2、rho>0、g 固定且每个分量非零、gamma>0。

```text
p(s) = softmax(4s + 2log|g|)
p_i*=1/d
s_i* = -0.5log|g_i| + (1/(2d))sum_j log|g_j|
a_i* = (prod_j |g_j|)^(1/(2d)) / sqrt(|g_i|)
J(s) = I - 4 gamma d (diag(p)-pp^T)
J(s*) = I - 4 gamma (I - 11^T/d)
```

零和子空间上特征值均为 1-4gamma。0<gamma<1/2 是严格线性收缩区间，
并保证该固定点局部渐近稳定；不是任意初始点或随机训练的全局保证。
gamma=1/2 的特征值为 -1，需高阶分析；d=2 存在高阶局部衰减，不能直接宣称边界不稳定。
常数方向特征值为 1，但固定体积排除了这一方向。

零梯度坐标每次增加 gamma，可能无有限固定点。
高维能量集中时最大收缩接近 gamma(d-1)，即使 gamma<1/2 也可能数值失效。
随机梯度对应移动均衡；E[p]=1/d 不等价于把 E[g] 代入固定点公式。
等体积不等于实际 L2 长度相同，更不保证覆盖真实删除方向或证明删除安全性。

## 数值行为

log-radius 存储可选 FP32/FP64；首轮 smoke 使用 FP64，仅是工程验证选择。
梯度归约、反馈能量及数值诊断使用 FP64。
扰动采用全局 log-domain 范数计算，生成实际施加的 FP32 epsilon 缓冲。
反馈使用同一个 epsilon 缓冲，不以 FP32 加法后的 theta 差替换。
记录加法舍入误差。零分量的 log(0)=-inf 是计算中的合法中间值，不是可持久状态。
全零扰动跳过几何更新；非零数学扰动完全下溢为零时报告数值错误。
所有损失、梯度、参数、优化器状态及半径状态非有限时显式失败。

浮点守恒采用显式容差：

```text
abs(sum_i s_i) <= volume_atol + volume_rtol * sum_i abs(s_i)
```

检查失败停止，不修正状态。普通 VPR/SAM/static/positive 检查椭球边界，
relative tolerance=2e-5、absolute tolerance=1e-12；数值合同需在 GPU 审计确认。
fixed_l2 消融明确不满足原椭球边界，不宣称它属于冻结主算法。

## 事务与异常

成功优化步提交一次几何。warm-up 时保持当前球形状态；不预先积累隐藏反馈。
g2 阶段失败恢复原参数及 BN buffers，候选几何丢弃。
任何失败都会使 Stepper 实例不可继续或保存 checkpoint。
基础优化器部分更新失败不宣称 momentum 已回滚；必须加载之前健康 checkpoint。

CPU 数值微基准不能预测 RTX5090 FP64 开销。未在正式实验中决定精度、gamma 或 warm-up。
