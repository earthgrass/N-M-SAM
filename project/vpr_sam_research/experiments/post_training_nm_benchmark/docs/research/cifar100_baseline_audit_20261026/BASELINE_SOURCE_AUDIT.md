# Wanda、SparseGPT、ROSE 官方来源与源码证据链

## 结论矩阵

| 方法 | 官方源码核验 | 原生视觉/CNN 路径 | 原生 ResNet18 Conv2d N:M | 本轮判断 |
|---|---|---|---|---|
| Wanda | PASS | 有 ImageNet ConvNeXt/ViT 入口，但只剪 block 内 `nn.Linear` | `NOT_SUPPORTED_IN_PINNED_SOURCE` | 独立 CNN 适配可保留评分语义 |
| SparseGPT | PASS | 无视觉入口；核心类表面接受 Conv2d | `NOT_SUPPORTED_IN_PINNED_SOURCE`：Conv 输入未 unfold | 独立局部线性适配可保留 OBS 更新 |
| ROSE | PASS | LLaMA/Mistral LLM 入口；核心类表面接受 Conv2d | `NOT_SUPPORTED_IN_PINNED_SOURCE`，且 N:M 重排分支实测失败 | 只能以“论文语义受控修复版”推进 |

## Wanda

- 身份：Mingjie Sun、Zhuang Liu、Anna Bair、J. Zico Kolter，**A Simple and Effective Pruning Approach for Large Language Models**，ICLR 2024。正式 OpenReview PDF 明确论文、作者、venue 与核心结论：[OpenReview](https://openreview.net/pdf?id=PxoFut3dWW)。
- 官方仓库：[locuslab/wanda](https://github.com/locuslab/wanda)，锁定 commit [`8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6`](https://github.com/locuslab/wanda/tree/8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6)。2026-10-26 的远端 HEAD 与本地只读 clone 一致。
- LLM 入口与核心：`main.py`；[`lib/prune.py::prune_wanda`](https://github.com/locuslab/wanda/blob/8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6/lib/prune.py#L127-L211)；[`lib/layerwrapper.py::WrappedGPT`](https://github.com/locuslab/wanda/blob/8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6/lib/layerwrapper.py#L5-L33)。代码计算 `abs(W)*sqrt(scaler_row)`，其中 `scaler_row` 累积输入列平方范数；每个 output row 内删除最小项，保留权重不更新。
- N:M 位置：`lib/prune.py` 第 169–174 行。`prune_n` 表示**删除数**，按权重矩阵每行连续 input columns 分组；`torch.topk(... largest=False)` 没有冻结 stable tie 语义，尾组也进入同一切片。
- 官方视觉代码确实存在：[`image_classifiers/README.md`](https://github.com/locuslab/wanda/blob/8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6/image_classifiers/README.md) 支持 ConvNeXt-B、DeiT-B、ViT-L/ImageNet。但 [`prune_utils.py::find_layers`](https://github.com/locuslab/wanda/blob/8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6/image_classifiers/prune_utils.py#L4-L13) 只找 `nn.Linear`，mask 只有 layer/row 非结构化比例模式，没有 ResNet、Conv2d 或 N:M 路径。
- 因此“Wanda 官方只有语言模型代码”是错误说法；准确说法是“官方有视觉 Transformer/ConvNeXt 的 Linear-only 非结构化实现，但没有 CIFAR-ResNet18 Conv2d N:M 实现”。

## SparseGPT

- 身份：Elias Frantar、Dan Alistarh，**SparseGPT: Massive Language Models Can be Accurately Pruned in One-Shot**，ICML 2023，PMLR 202:10323–10337：[PMLR](https://proceedings.mlr.press/v202/frantar23a.html)。论文与页面明确覆盖 OPT/BLOOM、one-shot、2:4/4:8。
- 官方仓库：[IST-DASLab/sparsegpt](https://github.com/IST-DASLab/sparsegpt)，锁定 commit [`147d2159dc4f3e9f73e47b32c04d7b3708f44436`](https://github.com/IST-DASLab/sparsegpt/tree/147d2159dc4f3e9f73e47b32c04d7b3708f44436)。2026-10-26 远端 HEAD 与本地只读 clone 一致。
- 入口：`opt.py`、`bloom.py`、`llama.py`；核心 [`sparsegpt.py::SparseGPT`](https://github.com/IST-DASLab/sparsegpt/blob/147d2159dc4f3e9f73e47b32c04d7b3708f44436/sparsegpt.py#L17-L147)。
- `add_batch` 累积 `H ∝ XXᵀ`；`fasterprune` 处理 dead columns，加 `percdamp*mean(diag(H))`，经 Cholesky→inverse→upper Cholesky 得补偿因子；随后按列确定 q、计算 `(w-q)/d` 并更新当前块后续列和后续块。不是静态 Hessian 分数排序。
- 原生 N:M 在第 106–108 行：`prunen` 是删除数，按每行连续列进行动态 mask 选择；第 122、128 行执行同坐标补偿。blocksize 若不与 M 对齐可能改变分组起点，官方调用的 128 对 2:4/4:8 对齐。
- `modelutils.py` 和构造函数把 `nn.Conv2d` 列为可接受层，并 flatten OIHW；但 `add_batch` 只为 Linear/Transformers Conv1D 重排输入，Conv2d 输入保持 NCHW，无法构造所需 `[Cin*kH*kW, samples*locations]`。仓库也没有 ResNet/ImageNet/CIFAR 入口。因此有效 Conv2d 支持为 `NOT_SUPPORTED_IN_PINNED_SOURCE`。

## ROSE

- 本任务中的准确身份已确认：Mingluo Su、Huan Wang，**ROSE: Reordered SparseGPT for More Accurate One-Shot Large Language Models Pruning**，CPAL 2026 oral；[arXiv 2603.05878](https://arxiv.org/abs/2603.05878)，[官方仓库](https://github.com/mingluo-su/ROSE)。不是检索分割等同名 ROSE。
- 锁定 commit [`88d922b3f4582b53b33e16e1396f82df563055f2`](https://github.com/mingluo-su/ROSE/tree/88d922b3f4582b53b33e16e1396f82df563055f2)，2026-10-26 远端 HEAD。Apache-2.0 clone 保存在 ignored `artifacts/upstream/ROSE`，未复制进适配源码。
- 入口：`main.py`→`lib/prune.py::prune_model`→[`lib/prune_zoo/rose.py::ROSE`](https://github.com/mingluo-su/ROSE/blob/88d922b3f4582b53b33e16e1396f82df563055f2/lib/prune_zoo/rose.py)。模型加载器是 LLaMA/Mistral 风格 LLM 流程。
- 论文算法：用 Wanda 分数预剪；计算 column loss 与 block loss；block 内列按损失降序、block 按损失降序；只有论文式 `R_rel=(max-min)/mean > 0.5` 的 columnar layer 才重排；同时置换 W 和 H，运行 SparseGPT，再逆置换。
- 锁定源码与论文有三项重要差异/缺陷：
  1. Conv2d 只 flatten 权重，`add_batch` 没有 unfold，统计形状不成立；
  2. 源码第 209 行用 `(max-min)/max`，不是论文 Eq. (7) 的均值分母；
  3. N:M 分支没有定义 `reorder_column`，但触发重排时第 212 行使用它。本轮同输入实测得到 `UnboundLocalError`，完整 traceback 在 `TEST_EVIDENCE/smoke_results_v2.json`。
- 论文第 5.5 节说明 2:4 时把每四个权重作为单位重排，不改变最终标准 2:4 图样；但当前 pinned source 无法可靠执行该路径。因此本轮实现明确命名为 controlled paper-faithful repair，不称官方复现。

## 来源锁与字节证据

官方核心文件 SHA256、运行环境、同输入对照和失败 traceback 均保存在 `TEST_EVIDENCE/smoke_results_v2.json`。适配器不在运行时导入历史 APBR/OBC 工程，也没有修改三份官方 clone。
