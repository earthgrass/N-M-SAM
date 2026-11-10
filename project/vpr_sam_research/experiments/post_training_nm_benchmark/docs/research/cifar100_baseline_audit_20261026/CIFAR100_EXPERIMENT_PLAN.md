# CIFAR-100 / ResNet18 正式实验前置计划（本轮不执行）

## 推荐队列

1. Wanda-CNN-unified2:4：可进入正式配置评审。
2. SparseGPT-CNN-unified2:4：先做单层 GPU 峰值/数值与 32-image 技术 smoke，再决定全模型。
3. ROSE-CNN-paper-repair-unified2:4：暂不进入三 seed；先解决官方 N:M 缺陷的外部确认与 Linear 对齐。

## 必须补齐的输入

- 下载并核验官方 CIFAR-100 Python archive，保存 URL、archive SHA256、解包 batch 哈希、50k/10k 标签分布和 torchvision 版本；不得 fallback synthetic。
- 冻结 100 类 CIFAR ResNet18：stem、avg-pool、shortcut、分类头、参数名必须与后续 checkpoint 一致；当前 `build_model()` 的 10 类头不能使用。
- 按论文主设计训练匹配 seed 的 CIFAR-100 Dense-SAM 与 APBR-SAM 200-epoch 资产。APBR 几何必须由训练得到，不能由三种后处理评分替代。训练需要另行批准，本轮没有启动。
- 冻结三 seed、dense 选模规则、校准训练样本 ID、是否增强、BN 处理、recovery 超参数与报告的 primary endpoint。

## 公平性协议草案

- 同一 seed 的三种后处理基线只加载同一 Dense-SAM 父 checkpoint；APBR-SAM 仍使用自己的训练所得 dense-with-geometry 父模型。
- 所有方法相同层范围：排除首 Conv/末 Linear/BN/bias；相同 OIHW output-row contiguous 2:4、dense tail、bool keep-mask。
- 校准数据只能来自 CIFAR-100 train，不得访问 test；先冻结 index 文件及 SHA。建议技术 smoke 32 张，正式候选 1024 张；这是提案，未批准。
- 报告 Dense、Raw Hard/原生输出、可选统一 BN copy、固定 mask recovery Final。Wanda 原生输出就是 hard zeros；SparseGPT/ROSE 原生输出包含补偿，不能标为 Raw Hard。若需要纯 mask damage，应另存无补偿分支且不冒充原方法。
- recovery 若采用现有 20-epoch 固定 mask SGD，应对所有基线统一，并同时报告零恢复的训练后结果。test 不应用于正式 checkpoint 选择；建议预留 validation 或固定 Final。
- 每层保存 calibration count、H/score 摘要、mask SHA、父 checkpoint SHA、输出 SHA、耗时和峰值显存。

## 分阶段门禁

1. 数据/模型身份测试；随机初始化模型只做 shape smoke。
2. 32-image CPU/GPU 单层：Wanda score；SparseGPT H/Cholesky/补偿；ROSE permutation。
3. 一个 seed、一个 dense checkpoint 的全模型 pruning 技术 pilot；不报告为正式结果。
4. 审阅 pilot 的 mask、输出、显存、耗时和 BN 行为后冻结配置 commit。
5. 用户另行批准后才运行三 seed 正式剪枝/recovery；最后一次性汇总，不改变失败 seed 参数。

## 资源估计与不确定性

- 现有 CIFAR-10 OBC 4090 记录约 33.4–33.6 分钟/seed，其中 recovery 约 7 分钟、Torch allocated 峰值 5.741 GiB。它只可作为同工程量级参考，不能外推为 SparseGPT/ROSE 保证。
- ResNet18 最大 3×3 层 K=4608，单个 FP32 H 约 81 MiB；Cholesky、inverse、factor 与 workspace 会增加数倍。逐层释放可避免所有层 H 同驻留，但 GPU 峰值必须实测。
- Wanda 无 K² Hessian，预计显著便宜；ROSE 在 SparseGPT 上增加 score、排序和置换，实际 overhead 待测。
- CIFAR-100 Dense-SAM/APBR-SAM 三 seed × 200 epochs 是本计划最大新增成本；当前没有可信本机计时，必须用 1-epoch 非正式 pilot 测量后再排期。

## 未解决问题

- CIFAR-100 train/test 的具体来源与哈希；
- Dense-SAM/APBR-SAM CIFAR-100 checkpoint 尚不存在；
- 正式 calibration 预算/增强与 validation 选择规则未获批准；
- ROSE pinned N:M 缺陷未获作者修复/确认；
- FP32 GPU 与 FP64 CPU 之间的 Cholesky 稳定性、damping grid 不能在看到 test 后调参；
- 实际稀疏 kernel 速度不在本轮实现范围，accuracy 比较不应宣称部署加速。
