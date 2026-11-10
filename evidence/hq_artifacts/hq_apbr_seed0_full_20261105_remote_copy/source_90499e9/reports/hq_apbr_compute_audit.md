# HQ-APBR V1 计算可行性审计（Phase 3）

**状态：BLOCKED_FOR_CURRENT_GPU_MEASUREMENT。** 本次只读审计在本机进行；Python 3.12.10、PyTorch 2.13.0+cpu，`torch.version.cuda=None`、`torch.cuda.is_available()=False`、device count 0，`nvidia-smi` 不在 PATH。未连接或启动历史服务器，也未运行训练或 Test。以下层形状和静态内存是当前模型代码计算结果；历史 GPU 数据有明确的独立来源，不能充当 HQ-APBR V1 本次实测。

只读来源工程根目录：`C:\Users\earthgrass73\Desktop\NM稀疏\APBR-post-training-benchmark\vpr_sam_research\experiments\post_training_nm_benchmark`。下文代码与历史证据路径均相对于该根目录；本 HQ-APBR V1 报告位于独立目录，不修改来源工程。

## 1. 实际模型范围和 Gram 大小

由 `baseline_adapters.cifar100_model.build_cifar100_model()` 和 `benchmark.nm.eligible_layers()` 在本机枚举：19 个 eligible Conv2d，11,157,504 个 eligible 权重坐标；其中包含 3 个 1×1 shortcut Conv2d。首层 Conv2d 与末层 Linear 被排除。所有下列层 `groups=1`。`K=C_in k_H k_W`；每层一份共享于该层输出行的 `K×K` 输入 Gram。内存仅是**单个稠密矩阵本体**，未包括模型、激活、Cholesky/逆因子副本和 CUDA 工作区。

| Eligible layer | 权重 OIHW | K | 输出行 | Gram FP32 MiB | Gram FP64 MiB |
|---|---:|---:|---:|---:|---:|
| layer1.0.conv1 | 64×64×3×3 | 576 | 64 | 1.27 | 2.53 |
| layer1.0.conv2 | 64×64×3×3 | 576 | 64 | 1.27 | 2.53 |
| layer1.1.conv1 | 64×64×3×3 | 576 | 64 | 1.27 | 2.53 |
| layer1.1.conv2 | 64×64×3×3 | 576 | 64 | 1.27 | 2.53 |
| layer2.0.conv1 | 128×64×3×3 | 576 | 128 | 1.27 | 2.53 |
| layer2.0.conv2 | 128×128×3×3 | 1152 | 128 | 5.06 | 10.12 |
| layer2.0.shortcut.0 | 128×64×1×1 | 64 | 128 | 0.02 | 0.03 |
| layer2.1.conv1 | 128×128×3×3 | 1152 | 128 | 5.06 | 10.12 |
| layer2.1.conv2 | 128×128×3×3 | 1152 | 128 | 5.06 | 10.12 |
| layer3.0.conv1 | 256×128×3×3 | 1152 | 256 | 5.06 | 10.12 |
| layer3.0.conv2 | 256×256×3×3 | 2304 | 256 | 20.25 | 40.50 |
| layer3.0.shortcut.0 | 256×128×1×1 | 128 | 256 | 0.06 | 0.12 |
| layer3.1.conv1 | 256×256×3×3 | 2304 | 256 | 20.25 | 40.50 |
| layer3.1.conv2 | 256×256×3×3 | 2304 | 256 | 20.25 | 40.50 |
| layer4.0.conv1 | 512×256×3×3 | 2304 | 512 | 20.25 | 40.50 |
| layer4.0.conv2 | 512×512×3×3 | 4608 | 512 | 81.00 | 162.00 |
| layer4.0.shortcut.0 | 512×256×1×1 | 256 | 512 | 0.25 | 0.50 |
| layer4.1.conv1 | 512×512×3×3 | 4608 | 512 | 81.00 | 162.00 |
| layer4.1.conv2 | 512×512×3×3 | 4608 | 512 | 81.00 | 162.00 |

总和为 91,987,968 个 Gram 元素；若同时保留全部 19 层，单份 FP32 Gram 约 **350.91 MiB**，FP64 约 **701.81 MiB**。若逐层计算并释放，最大单份矩阵为 81/162 MiB。K=4608 层有 3 个；三份同尺寸 FP32 矩阵本体即 243 MiB。一般稠密 Cholesky 和精确 inverse-diagonal 的运算量是 `O(K³)`，此处 `ΣK³=349,546,479,616`（作为复杂度权重，不是实测 FLOPs）。只提取逆对角并不消除分解成本；分块求解单位向量可减少峰值临时内存，但求解所有对角仍需覆盖全部 K 个右端。

## 2. 与既有 SparseGPT 实现的关系

当前 `SparseGPTAdapter.add_batch()` 用 `F.unfold` 后的每层 patch 矩阵累计 `2XXᵀ`，`hessians()` 再除以累计 patch 列数；即实际返回 **`H_SG=2XXᵀ/n`**。相对阻尼为 `0.01 × mean(diag(H_SG))`，代码路径为 Cholesky → `torch.cholesky_inverse` → 上三角 Cholesky 补偿因子。HQ 规范的 `H_HQ=XXᵀ/n` 比它小 2 倍。若复用同一相对阻尼，需要阻尼也同步除以 2；对应 Q 整体乘 2。在没有特殊零对角修复、所有层均按同一系数缩放时，对 `0.5 log diag(Q)` 的**全局中心化差值**不变，但未中心化的目标值和原始 OBS 代价尺度会改变。正式代码必须显式固定一种口径，并审计零对角“dead feature”修复逻辑；该实现把 dead feature 的对角置 1 后再阻尼，此操作若未经相应缩放，会破坏上述严格等比例关系。不能只改 H 而不改阻尼。

当前 `grouped_patch_matrices()` 支持真实 grouped Conv2d，将每组 patch/Gram 分开；本模型 eligible 19 层均为 `groups=1`。`F.unfold` 的次序是 channel → kernel row → kernel column，与 OIHW 每个 output row 展平一致。`iter_grouped_patch_chunks()` 限制后续矩阵乘法片段，但 `F.unfold` 本身先完整物化；在 CIFAR-32 上历史 smoke 可行，大图扩展需另做空间分块。

HQ V1 只需 `diag((H+λI)^-1)`，不必运行 Group-OBS 六候选和逐列补偿。精确实现宜用 Cholesky 后的三角求解或 `cholesky_inverse` 提取对角，避免不稳定的通用 `torch.linalg.inv`；两者仍需实测 K=4608 的内存、耗时和数值。对角必须映射到每个输出行对应的输入坐标，几何目标有 11,157,504 个坐标，不能将 19 层均值简单平均。

## 3. 历史 GPU 证据：可参考，不能代替 HQ V1 实测

1. **RTX 4080 SUPER 单层 SparseGPT 门禁**：`layer4.0.conv2`、K=4608、32 张真实 CIFAR-100 Train 图、CUDA FP32、相对阻尼 0.01。历史记录显示 Hessian 构建 0.012786 s、分解 0.075125 s、OBS 补偿 1.370275 s；peak allocated 519,058,944 bytes（495.01 MiB），reserved 614,465,536 bytes（586 MiB），无 OOM、Cholesky 失败或非有限值。原 V1 的**进程内**显存释放检查失败；后来 V2 在**进程退出后**的资源门禁通过。这是原 SparseGPT 的单层执行，不是 HQ 刷新阶段的耗时或状态。证据：`docs/research/cifar100_gpu_gate_20261026/REMOTE_GPU_GATE_4080S_20261026/GPU_GATE_RESULTS_REMOTE.json` 及 `GPU_RESOURCE_V2/GPU_RESOURCE_GATE_V2_RESULTS.json`。
2. **RTX 5090 Joint Phase 4.1 smoke**：真实 19 层 Hessian/factor refresh 0.23103 s，peak allocated/reserved 3,471,265,792 / 4,634,705,920 bytes，全部 Cholesky 成功。它从 128 张固定 Train 图收集激活，每层抽样最多 8192 patches；不同层样本量不同，3 个 K=4608 层为 2048 patches。该 smoke 构造了完整逆矩阵和六候选因子，**超出** HQ 仅求逆对角的工作，但仍没有 HQ 的逐 Epoch 几何目标、持久状态、200 次刷新以及实际 HQ 训练。因此 0.23103 s 只能作为不同实现、不同资源作用域下的历史参考，不能乘 200 直接宣称 HQ 总成本。该记录中最大 `condition_max=14,027.05` 指六候选 **2×2 Q 子块**的条件数，不是整层 H 的条件数。证据：`artifacts/apbr_gpt_joint_seed0_rebased/phase4_1_handoff_20261101/remote_evidence_copy/smoke_results/hessian_refresh.json`、`hessian_layers.csv` 和同目录 `PHASE4_1_GPU_SMOKE_REPORT.md`。

## 4. HQ V1 真实 GPU 审计尚缺的测量

使用明确的一张可用 GPU、冻结 Train 校准 ID 和真实 Dense checkpoint，逐层记录：输入激活采集、unfold、patch 抽样、Gram 构建、dead-coordinate 处理、相对阻尼、Cholesky、inverse-diagonal、目标广播/中心化、状态写入的同步 CUDA 时间；输入与 Q 对角范围、最小/最大阻尼、Cholesky info、非有限值、峰值 allocated/reserved、运行前后可用显存。应比较相同训练环境中一次正常 SAM Epoch 与一次完整 HQ Epoch（含校准刷新），以及多个不同 Epoch 模型下的条件性；只测随机初始化单次刷新不够。每次刷新后释放本层大矩阵，不建立 19 层并存的训练期 Hessian 缓存。未预注册并验证之前，不得在 Cholesky 失败时自动改阻尼、减少样本或换近似。

计算报告必须分别给出单 Epoch 额外时间、200 次刷新总额外时间、每层瓶颈和峰值 GPU 显存，再据此判断是否值得运行小规模 Pilot。若成本不可接受，可以提出降低刷新频率、减少校准样本或可审计的 inverse-diagonal 近似；任一改变都属于新候选协议，须先冻结并单独测试，不能静默替换 HQ V1。

**当前结论：** 静态模型尺寸已审计，历史 GPU 数值可行性提供参考；HQ-APBR 每 Epoch 19 层精确 inverse-diagonal 的实际耗时、显存和数值稳定性仍**未验证**。本机无 CUDA，本轮 Phase 3 保持 BLOCKED，不能宣称通过 GPU 性能门禁或建议启动 200 Epoch。
