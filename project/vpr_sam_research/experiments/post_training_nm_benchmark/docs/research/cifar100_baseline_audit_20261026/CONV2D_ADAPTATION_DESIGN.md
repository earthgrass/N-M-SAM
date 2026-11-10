# Conv2d 局部线性映射与三种适配设计

## 精确映射

对单组卷积，令 `K=(Cin/G)*kH*kW`、输出空间位置数为 L。PyTorch `unfold` 生成

\[
X_{patch}\in\mathbb{R}^{K\times (N L)},\qquad
W_{mat}\in\mathbb{R}^{(Cout/G)\times K},
\]

且 `Ymat = Wmat Xpatch`。`F.unfold` 的列内顺序为 channel→kernel-row→kernel-column，恰好与 OIHW `weight.reshape(Cout,-1)` 相同。组卷积对每个 g 独立计算，再按 output-channel 顺序拼接；bias 在乘法后添加。

输出尺寸逐维为

\[
\left\lfloor\frac{H+2p-d(k-1)-1}{s}+1\right\rfloor.
\]

因此 stride/padding/dilation 被 `unfold` 精确吸收。映射不改变 dtype、参数数量或原始坐标。

## 激活统计

- Wanda：对每个局部输入坐标 `j` 累积 `a_j=Σ_t X[j,t]^2`，评分 `|W_ij| sqrt(a_j)`。除以统一样本数只会乘公共正数，不改变组内排序。
- SparseGPT：对每个卷积 group 构造 `H=2 XXᵀ/T`。全局正比例缩放在相对 damping 与更新式中抵消，但实现仍固定该定义便于复核。
- ROSE：预剪分数复用 Wanda norm；H 与 SparseGPT 相同。置换 P 必须同时作用为 `W←WP`、`H←PᵀHP`，完成补偿后用 `P^{-1}` 写回。

## SparseGPT 更新保持项

适配保留锁定源码的核心顺序：dead-column 处理→相对阻尼→Cholesky(`H`)→inverse→upper Cholesky→从左到右动态选择当前 N:M mask→`err=(w-q)/d`→更新块内未来列→块间矩阵补偿。改变的只有合法 CNN 输入矩阵、dense tail、明确 tie 及 CPU-safe timing；没有加入 APBR 几何、训练或新的 Hessian 目标。

## 内存边界

`conv2d.py` 对矩阵乘法和后续统计可按向量列分块，但当前 `F.unfold` 仍一次物化 CIFAR-32 patch。对 CIFAR-32 该张量可控；不能把它表述成普适峰值内存有界。正式大图适配必须增加空间 tile/halo 的 streaming unfold，并重新验证边界位置。本轮没有以一次性全模型 patch cache 运行。

SparseGPT 的主峰值来自每组 `K×K` FP64 smoke Hessian及分解；正式实现应使用经审计的 FP32、逐层立即释放。ResNet18 3×3 层 K 最大为 4608，单个 FP32 H 约 81 MiB，分解/副本/workspace 会数倍于此，须 GPU 单层实测后冻结预算。

## 数学验收

`tests/test_cifar100_baseline_adapters.py` 覆盖普通、groups=2、depthwise 三类卷积；包括 1×1/3×3、stride 1/2、padding、dilation、bias/no-bias、不同 batch/空间尺寸与非连续输入，并构造 `num_classes=100` 的项目 ResNet18 完成一层真实 Wanda 采集/剪枝后全模型前向。`smoke_results_v2.json` 的最大前向误差为 0，输入梯度最大误差 `1.11e-16`，权重梯度最大误差 `1.07e-14`（float64）。状态为 `MATHEMATICAL_MAPPING_PASS`。

## 明确限制

- 本轮没有验证 transposed convolution、非零 padding mode 或稀疏 kernel 加速；不支持这些情况。
- depthwise 1×1 的 K=1 无完整 2:4 组，按 dense-tail 规则不剪。
- 正确映射不等于分类性能有效；CIFAR-100 accuracy 仍为 `FULL_EXPERIMENT_NOT_RUN`。
