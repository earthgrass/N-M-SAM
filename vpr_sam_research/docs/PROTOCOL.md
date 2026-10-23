# 实验协议与待批准项

## 公平性

正式 Dense-SGD、Dense-SAM、VPR-SAM 和消融全部随机初始化，不支持历史 checkpoint warm start。
同 seed 的初始模型 state_dict hash 必须相同。训练/验证划分、每 epoch 样本排列、
每样本增强由命名独立随机流决定，不受优化器消耗随机数影响。
SGD 执行一次前向反向；SAM 系列两次使用同一个增强后输入。
两次 SAM 前向仅第一次更新 BN running buffers，第二次仍使用 batch statistics。
保持同模型、数据、seed、训练 epoch/batch、基础 SGD 参数、scheduler 和增强。
固定 FP32 模型、禁用 TF32/AMP/torch.compile，启用确定性算法；不满足的 CUDA kernel 显式失败。
复现承诺限于相同源代码、依赖、设备和确定性设置；不声称跨 CPU/GPU 或版本逐位一致。

## 数据

真实 CIFAR-10 必须通过 torchvision 官方 MD5 校验。下载是独立显式动作。
训练配置默认不下载，无 FakeData fallback。合成数据仅用于 smoke，产物显式标记。
从训练集按 seed 生成固定不重叠 validation split。测试集不参与训练、调参和 checkpoint 选择。
训练只评估 validation。best_validation 用 validation accuracy，平局保留较早 checkpoint。
final 是预算结束 checkpoint；两者都保留。对照必须采用同一选择规则。
最终 test 只能在协议与模型选择冻结后通过 evaluate --split test 显式执行。
smoke 中可测试 test 路径，但合成数据没有研究意义。

## N:M 硬剪枝

N、M、scope、tail 必须显式配置，没有正式默认 2:4。
首版仅实现统一 magnitude 评分，稳定排序处理 ties（保留较低索引）。
Linear 沿 in_features；Conv2d 每个输出通道展平 [in,kh,kw] 后连续 M 分块。
这一逻辑分块不声称自动满足特定硬件稀疏 kernel 布局，也不宣称硬件加速。
tail=dense 保留不足 M 的尾块；tail=error 拒绝尾块。统计实际全模型稀疏率与层报告。
scope 支持 all、exclude_first、exclude_first_last、residual_and_head（排除首层及1×1卷积）。
剪枝输出是实际参数归零，不是仅做评分。立即评估不做 BN 校准，报告 delta loss、accuracy drop、logit L2 和 KL。
不把 checkpoint 中自然出现的数值零计为结构化 mask 删除数。

## 恢复

从各方法自己的硬剪枝结果开始，固定 mask、不 regrow，不使用历史实验产物。
恢复采用共同的显式 SGD 或 SAM 配置，新建基础优化器，统一重置 momentum 和 scheduler。
投影参数、梯度及 momentum 防止权重回生。SAM 恢复在保留子空间使用公共球形扰动。
不默认使用各自训练方法恢复，以免混淆训练几何效应与恢复方法效应。
恢复 epoch、lr、momentum、weight decay、rho、scheduler 需批准并保持一致。

## 鲁棒性和消融

外部探针统一使用固定 rho 的一阶 L2 sphere，剪枝后统一为固定 mask 子空间。
它是可比较探针，不是精确最坏损失或认证鲁棒界。另行记录训练时各自几何下的损失增量。
static：seeded、独立于数据/训练的高斯 log-radius，初始化一次全局去均值，之后固定；static_std 待定。
positive：仅反转冻结反馈的符号，标记为消融。
fixed_l2：先构造 VPR 方向再固定其 L2 长度，明确改变主算法的扰动集合解释。
不加入逐层/逐块守恒、EMA、裁剪或新评分。

## Checkpoint

train 仅允许随机初始化或显式恢复本项目同配置健康 checkpoint。
没有隐式自动寻找 checkpoint。schema、参数映射/精度、源文件哈希、配置、数据和环境严格核验。
恢复基于 epoch+next_batch 重建确定性排列，包含 epoch 累计量、所有 RNG、SGD momentum、scheduler、几何及历史。
checkpoint 文件通过同目录临时文件 atomic replace，未完成优化步不能保存。
mid-epoch 恢复支持 workers>0，因为增强按样本/epoch 寻址且没有 persistent workers。
源代码发生变化后精确恢复被拒绝；不提供忽略校验的快捷选项。
checkpoint 为 Python pickle，只加载可信的本项目文件。

## 统计

独立随机种子是统计单位，不能把 batch/epoch 当独立重复。
汇总首先检查同 seed 初始化、完整数据顺序和数据指纹，拒绝非研究协议差异及重复 method/seed。
报告 n、均值、样本标准差和同 seed 配对差值。n=1 不提供 CI。
至少两对 seed 提供固定随机源的 paired bootstrap percentile interval，少量 seed 只能探索性解释。
正式比较仍需决定 seed 数量、调参预算及多重比较策略。

## 尚未批准的正式协议

N:M、剪枝层范围/尾块、验证集大小、随机种子列表、训练预算与基础超参数、
gamma/精度/warm-up、恢复预算、static_std、最终 checkpoint 选择规则、probe 预算、GPU 分配均未确定。
configs/formal/*.json 中 null 是刻意的执行阻断，不是缺失的工程实现。
先填写一个共同配置，再用 materialize_protocol.py 展开方法/seed，审核后才能执行。
