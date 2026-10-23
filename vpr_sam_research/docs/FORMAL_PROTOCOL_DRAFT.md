# 200 epoch 共同协议草案（PENDING，禁止启动）

只比较 Dense-SAM + Magnitude N:M 与 VPR-SAM + Magnitude N:M。
以下候选值不是用户的正式批准；现有 `configs/formal/` 的未决字段保持阻断。
本轮校准 checkpoint 不得作为正式初始化。

| 字段 | 候选/已有能力 | 状态 |
|---|---|---|
| 数据 | 官方 CIFAR-10，50,000 原始训练、10,000 测试；验证从训练集无放回划分 | 实测状态见 SERVER_ACCEPTANCE |
| validation | 建议 5,000；按 seed 命名 split 流生成；是否固定跨 seed 的 split 需决定 | PENDING |
| 模型 | CIFAR ResNet18，[2,2,2,2]，3×3 stride1 stem，无初始 maxpool，10 类 | 已实现；正式选择 PENDING |
| 初始化、seed | 全新随机初始化，同 seed 完全相同权重、增强、排列；建议至少 3 个配对 seed | 列表/统计预算 PENDING |
| 预算 | 每组 200 epoch，batch128（最后不足 batch 保留） | PENDING |
| SGD | 建议 lr0.1、momentum0.9、weight_decay0.0005、Nesterov=false | PENDING |
| 调度 | 每 epoch CosineAnnealingLR，eta_min=0 | PENDING |
| 增强 | 已有零填充4后32×32随机裁剪、0.5概率水平翻转；mean=(.4914,.4822,.4465)、std=(.2470,.2435,.2616)；可配置Cutout；校准Cutout=0 | Cutout及共同增强 PENDING |
| SAM rho | 建议0.05，两方法一致；VPR 实际 L2 长度单独记录 | PENDING |
| 固定 gamma | 根据真实 g1 能量与短程几何结果提名，见 GAMMA_CALIBRATION | PENDING |
| 几何 warm-up | 校准0步；正式是否使用及步数需批准 | PENDING |
| 精度 | FP32 模型；log-radius FP32/FP64 实测后推荐 | PENDING |
| 执行 | 确定性开启；AMP/TF32/compile关闭，不以降低标准换速度 | 现有实现；GPU实测见验收报告 |
| N:M | N、M 均待定；1:3 仅用于 smoke | PENDING |
| 剪枝范围/尾块 | canonical scope 待定；tail=dense或error；需列完整权重名单 | PENDING |
| Mask | magnitude稳定降序，平局保留低索引；Conv每输出通道展平[in,kh,kw]后分块；Linear沿输入维 | 已实现，正式协议 PENDING |
| Dense选择 | 建议 final；best_validation只能预注册为次要分析，图表绑定同一checkpoint | PENDING |
| 恢复 | 两方法共用新SGD与重置momentum/scheduler、固定mask；epoch、lr及其他参数待定 | PENDING |
| 恢复选择 | 建议 final；选择规则须共同冻结 | PENDING |
| 外部 probe | 共同一阶L2球；剪枝后同mask子空间；rho、batch数待定；不比较各自训练椭球损失作为公平鲁棒性 | PENDING |
| 日志/保存 | 每步核心指标、每epoch验证/几何层贡献；保存last、final、best；步间隔及存储预算待定 | PENDING |
| GPU资源 | 明确cuda:N与物理UUID映射、线程/worker数、是否独占及时间/磁盘预算 | 实测后仍需批准分配 |

主要指标：稠密验证accuracy/loss、硬剪枝accuracy/loss、hard loss增量、accuracy下降、固定预算恢复结果、共同球形probe、时间/显存与几何轨迹。
统计单位为独立配对seed，不能把batch或epoch当重复；报告配对差值、样本标准差与预注册区间方案。
测试集只在最终协议与模型选择冻结后使用。校准不使用验证或测试准确率选gamma。

当前基线具备常规 CIFAR ResNet18 + SGD momentum + cosine + SAM 组件，但短程验收不能证明200 epoch基线已达到充分收敛或强基线水平。
最终还须审查完整学习率/增强/恢复预算与先验公平性；不擅自补入MixUp、EMA、AutoAugment或Label Smoothing。
