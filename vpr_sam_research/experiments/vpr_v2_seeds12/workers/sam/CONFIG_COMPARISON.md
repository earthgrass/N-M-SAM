# SSAM / F-SAM 配置及实现差异审计

2026-10-24，Asia/Shanghai。本轮只执行 Dense-SAM；SSAM为主要配置来源。
SSAM锁定 `76974fd0255cf2f61201e19566cb64ceb429ecdd`；
F-SAM锁定 `2e2f9899ab473880430640fa5def4ad8f5aa2401`。
实际审阅文件以Git blob原字节保存于reference/，SHA256见reference/UPSTREAM_SHA256.json。
两库均未检出额外AGENTS.md。本轮不执行其训练入口或隐式导入其配置。

| 项目 | SSAM2022官方路径 | F-SAM实际代码 | 本轮采用 |
|---|---|---|---|
| 模型 | models/resnet.py：3×3/stride1 stem，无maxpool，BasicBlock[2,2,2,2]，64/128/256/512，avg_pool2d(4)，linear10；PyTorch默认初始化 | models/resnet.py同CIFAR尺度，但模块名称是conv2_x..conv5_x、fc，inplace ReLU、AdaptiveAvgPool；不能直接按名称互载 | SSAM原类文本；AST相同；11173962可训练参数；从零初始化 |
| 数据 | CIFAR10_cutout，50k/10k官方划分，无内部Validation | utils.get_datasets_cutout同50k/10k | 全50k/10k，不读任何45k/5k split文件 |
| Batch / tail | batch128，RandomSampler，训练drop_last=True，测试False；390训练批/epoch | 默认batch128；CIFAR10 loader未设drop_last，故False，391批/epoch | 训练390批/49920样本，随机尾80张不参与该轮；测试79批/全部10000 |
| Workers / pin | parser默认8 / True | parser默认4 / True | 2 / True，运行兼容设置明确记录 |
| LR / momentum / WD | LR.05、momentum.9；SAM实验WD.001，parser基础默认5e-4不可混淆 | parser LR.1、WD1e-4；run.sh显式LR.05、WD.001、momentum默认.9 | .05/.9/.001，nesterov=False |
| rho | CIFAR10论文/README .1；parser默认.05 | parser默认.1；run.sh是CIFAR100/.2 | .1，非adaptive标准SAM，不使用FriendlySAM |
| SAM公式 | 全局norm，rho*g/(norm+1e-7)，原实现add/sub恢复 | SAM类标准两步；FriendlySAM另有历史梯度噪声机制 | 全局norm、同一个batch Tensor、数值epsilon1e-7；精确copy恢复，g2留到SGD完成 |
| LR scheduler | 自定义CosineLRscheduler；训练后调用step(epoch)，epoch从0开始 | parser默认step；cosine分支为逐batch LambdaLR；train每batch无条件scheduler.step | SSAM原类AST提取；保留调用顺序和零基索引；前两轮LR=.05 |
| 增强顺序 | Crop→Flip→Tensor→Normalize→Cutout | Flip→Crop→Tensor→Normalize→Cutout | SSAM顺序 |
| Normalize | mean[125.3,123,113.9]/255；std[63,62.1,66.7]/255 | mean(.4914,.4822,.4465)，std(.2023,.1994,.2010) | SSAM除法结果精确序列化，不沿用F-SAM数值 |
| Cutout | gate由torch.rand决定；中心由NumPy randint决定；size16/p.5，归一化后乘0/1 mask，边界截断 | 同size/p，但中心由torch.randint决定且原地置零 | SSAM实现；固定20个RNG种子逐元素与官方函数对照 |
| BN第二次前向 | model.train，两次均更新running mean/var/count | train.py disable_running_stats设momentum=0；均值方差不更新，counter仍递增 | 双次更新；不调用disable/restore/track_running_stats=False，不额外重估BN |
| Seed / flags | parser默认1234；setup_seed设Python/NumPy/CPU/CUDA种子，未在此固定所有确定性flag | parser默认1；set_seed设deterministic=True/benchmark=False，但main随后benchmark=True | 用户最新覆盖seed0；deterministic=True、benchmark=False、AMP/TF32/compile关闭；不是旧硬件逐bit复现 |
| checkpoint选择 | 每epoch官方Test，严格更高accuracy时checkpoint.pth覆盖 | 计算best_prec1/is_best，但当前save_checkpoint仅torch.save当前model.th，未按is_best另存 | 本轮显式保存best_test.pt与last.pt；并列最早；不能称best_val |
| 状态 / 发布 | 原入口保存model、wrapper optimizer、scheduler等；SGD状态完整性需另核验 | model.th主要state_dict/best_prec1，resume完整性不足 | schema2完整SGD/scheduler/RNG/config/source/data，临时及发布后双重读取，SHA256 |

来源：[SSAM固定仓库](https://github.com/Mi-Peng/Sparse-Sharpness-Aware-Minimization/tree/76974fd0255cf2f61201e19566cb64ceb429ecdd)、
[F-SAM固定仓库](https://github.com/nblt/F-SAM/tree/2e2f9899ab473880430640fa5def4ad8f5aa2401)。

## 调度的实际执行口径

初始化SGD LR=.05；epoch索引0训练使用.05，随后step(0)仍为.05；
索引1训练仍使用.05，随后step(1)才衰减。
第200轮（索引199）实际使用cos(198π/200)对应LR；最后调用step(199)。
这不是“修正off-by-one”，而是按用户要求保留官方实际执行路径。
日志同时保存learning_rate_used及learning_rate_after_epoch；200点序列与官方类独立比较。

## 兼容偏差与边界

- 用户将任务书seed1234改为0；见reference/SEED_OVERRIDE.md。未引入其他seed。
- 当前torch2.8/CUDA12.8/RTX5090，FP32确定性执行，worker2；不使用作者旧CUDA二进制。
- 参数恢复从减去epsilon改为保存值copy，避免FP32抵消误差；不改变扰动公式、g2或SGD定义。
- 独立checkpoint保存完整底层SGD与RNG并严格有限性检查；不复用官方易缺失的wrapper状态。
- 数据只读使用已存在官方文件，不下载、不上传、不重划分；无45k/5k验证数据路径。
- 初次CPU验收检出Std十进制舍入与官方float64除法结果不完全相同，已按官方除法精确序列化，
  没有更换统计量；保留原失败日志。正式训练前需重新通过完整验收。
- 原45k/5k、单次BN、Best Validation任务已被新授权替代并终止，权重和日志保留，绝不warm start。

Test逐轮参与模型选择是本轮明确授权的文献对齐口径。
Best Test不是独立无偏泛化估计，也不是后续VPR/Guide在该Test上反复开发的独立测试证据。
96.83%仅是外部参考，不能触发调参、自动重训或失败记录删除。
