# τ 来源审计

日期：2026-11-06。只读检查论文、源码、Git、本地原始对话、配置、冻结清单及评测 JSON。未执行训练、剪枝、数值测试或 CIFAR-100 Test，未载入 checkpoint。小型源码和 JSON 哈希实际重算；大 checkpoint/Mask 哈希交叉核对历史记录，未重新读取全部二进制。

## 1. 核心回答

1. **逐步近优规则在 2026-11-03 的设计中已经存在，不是 11 月 6 日写论文时新增。** 原始对话包含 assistant 提议、用户批准、assistant 写入 0.01、披露 1% 后的全模型评测授权。
2. **具体 0.01 由当时 assistant 写入 caller。** 为什么恰好取 1% 的定量依据为 **UNKNOWN**；未找到用户单独指定该数值的消息。
3. **最早引入 Commit：UNKNOWN。** 实现未跟踪，运行时 HEAD 不能当成来源提交。
4. **77.63%=(77.83%+77.43%)/2**，两项都是原版 APBR-VPR 父模型、相同 selector、τ=0.01，且两次删除都使用近优几何裁决。
5. 本轮无 τ 定义与旧算法不同：首删直接几何 argmin，次删纯条件 OBS argmin，再一次联合补偿。已检查产物不能为它提供准确率。
6. 论文 SAM 77.43% 的具体运行绑定为 **UNKNOWN**。真实 SAM Seed 2 与 APBR Seed 1 都出现过此值，不能仅凭数字认定抄错。

## 2. 路径与 Git

工程根：`C:/Users/earthgrass73/Desktop/NM稀疏/APBR-post-training-benchmark`。

下文 E：`C:/Users/earthgrass73/Desktop/NM稀疏/APBR-post-training-benchmark/vpr_sam_research/experiments/post_training_nm_benchmark`。

下文 P：`C:/Users/earthgrass73/Desktop/NM稀疏/paper`。

| 字段 | 读取得到的值 |
|---|---|
| Branch | `research/post-training-nm-benchmark` |
| HEAD | `55fe3e300dbbd09c1178a6316afebc9ecf367947` |
| HEAD author | `earthgrass <earthgrass@users.noreply.github.com>` |
| HEAD time | `2026-10-27T10:48:09+08:00` |
| HEAD subject | `Record endpoint reporting identity and tests` |
| Git directory | `C:/Users/earthgrass73/Desktop/NM稀疏/VPR-SAM/.git/worktrees/APBR-post-training-benchmark` |
| Subdirectory prefix | `vpr_sam_research/` |
| 近优 adapter | `??`，未跟踪 |

HEAD 只涉及协议身份与测试记录。以下只读搜索均无匹配：

- `git log --all --full-history -- :(top)vpr_sam_research/experiments/post_training_nm_benchmark/baseline_adapters/apbr_nearoptimal_conditional_obs/adapter.py`
- `git log --all -G 'near.optimal|nearoptimal|APBRNearOptimal' -- :(top)**`
- `git log --all -S 'TAU = 0.01' -- :(top)**`
- `git log --all -S 'apbr_nearoptimal_conditional_obs_prune' -- :(top)**`

不能把 HEAD 作者认定为算法作者。下面的实现责任来自对话工具记录，而非 Git。

## 3. 谁、哪次任务引入

原始主聊天：[rollout JSONL](C:/Users/earthgrass73/.codex/sessions/2026/10/26/rollout-2026-10-26T18-27-17-01a177c0-0eba-79c2-8f46-f63f19ec94a7.jsonl)。
以下为 2026-11-03 UTC，北京时间加 8 小时。

| 行号 | UTC | 可核实记录 |
|---:|---|---|
| 33620 | 13:49:01.752 | 用户提出 APBR 结合 ExactOBS。 |
| 33712 | 13:51:19.034 | assistant 提议物理 OBS 筛近优候选，再使用 APBR 几何。 |
| 33741 | 13:51:51.289 | 子代理 `/root/apbr_exact_math` 给出逐步 `c_i <= (1+τ)c_min` 方案。 |
| 33767 | 13:52:45.499 | assistant 向用户明确给出带 τ 四步方案。 |
| 33774 | 14:00:18.495 | 用户：“ok 执行你说的这个”。 |
| 33973 | 14:05:29.971 | assistant 的工具调用新增 Pilot runner，写入 `TAU = 0.01`。 |
| 34160 | 14:09:38.622 | assistant 披露“1% 近优窗口内”的 Pilot 结果。 |
| 34193 | 14:11:04.749 | 用户：“直接按照策略 剪枝补偿之后 拿测试集验证即可”。 |
| 34200 | 14:11:25.186 | assistant 声明用冻结 1% 规则完成 19 层和固定 Test。 |
| 34317 | 14:16:41.204 | 报告真实几何77.83、物理77.72、单位77.76、打乱77.82。 |
| 34812 | 15:02:17.060 | Seed 1：SAM 条件贪心77.77，APBR近优几何77.43。 |
| 35612 | 15:48:40.502 | 汇总 APBR 77.63±0.28；SAM 77.81±0.05。 |

L33767 的关键提议是：“只保留代价不超过最小值 (1+τ) 倍的候选……在这些候选中，用 APBR 的 |w_i|exp(-s_i) 选删位。”

最早可读 adapter 创建记录：
[pilot_math_tests 子会话](C:/Users/earthgrass73/.codex/sessions/2026/11/03/rollout-2026-11-03T22-00-48-01a1a1b6-68dd-7813-873e-db574f4876a6.jsonl)，L565，`2026-11-03T14:04:23.064Z`。该 adapter 默认 τ=0；实验 0.01 来自主 runner。

**设计依据：**历史提议明确希望保持物理二阶代价优先，让几何在物理近似等价选择间起作用，不修改补偿目标。泛 τ 方案得到批准，1% 披露后也有按策略继续评测的授权。**为什么具体选1%、是否有训练侧阈值筛选结论：UNKNOWN。** 不能把这解释为用户亲自指定1%，也不能说完全没有设计授权。

部分历史工具字段不可读取。“最早”限定为本次检索到的最早可读记录。

### 更早的不同算法

`E/docs/research/apbr_gpt_group_joint_obs_20261031/DESIGN_NOTE.md:130–147` 和 [旧 Group-OBS 任务](<C:/Users/earthgrass73/.codex/attachments/b4fe3571-8445-4b97-b786-9e348960f2d2/Pasted text.txt>) L309–354，已经在**六种删除对**上设置0.5%、1%、2%窗口，主设置1%。

该设计对应 `baseline_adapters/group_joint_obs/adapter.py::select_group_mask`，是对整对候选筛选；11月3日算法是每次条件单点选择筛选。旧六对授权不能证明两者是同一方法。

## 4. 当前实现的精确规则

[near-optimal adapter](C:/Users/earthgrass73/Desktop/NM稀疏/APBR-post-training-benchmark/vpr_sam_research/experiments/post_training_nm_benchmark/baseline_adapters/apbr_nearoptimal_conditional_obs/adapter.py)

| 位置 | 规则 |
|---|---|
| `_choose` L67–91 | 取物理成本最小值，限定 `cost <= (1+tau)*min_cost`，窗口内最小化 `2*log(abs(w))-2*s`。 |
| L82–83 | `tau==0` 或 `mode=="physical"` 时直接物理 argmin，包括精确平局。 |
| L192–195 | 第一删：计算当前物理单点成本，然后调用 `_choose`。 |
| L199–210 | 虚拟更新后计算第二次成本，再调用同一个 `_choose`，第二次也用几何。 |
| L225–248 | 从保存的当前物理权重进行一次联合补偿，并向后续活动列传播。 |

| 步骤 | 已运行近优版 | 本轮指定无 τ 版 |
|---|---|---|
| 首删 | OBS近优窗口内几何 argmin | 当前组直接几何 argmin |
| 次删 | 条件OBS近优窗口内几何 argmin | 条件OBS直接 argmin |
| 几何参与次数 | 两次 | 一次 |
| 阈值 | 0.01 | 无 |
| 最终补偿 | 一次物理 Joint OBS | 一次物理 Joint OBS |

**这是选 Mask 方法的改变。** 旧代码将 τ 设为0会得到两次纯物理选择，并不能得到首删几何、次删OBS。

最早保留 Pilot preflight：`E/artifacts/apbr_nearoptimal_conditional_seed0_train_pilot_20261103_v1/preflight.json`，L2083有 `tau:0.01`，L8–10绑定当前 adapter SHA。v1因权重视图别名作废，可用于参数来源追溯，不能作为有效性能结果。

## 5. 数字对应的版本

| 字段 | APBR Seed 0 | APBR Seed 1 |
|---|---|---|
| results，路径相对E | `artifacts/apbr_nearoptimal_conditional_seed0_full_cpu_20261103_v1/results.json` | `artifacts/sam_apbr_conditional_seed1_cpu_20261103_v1/results.json` |
| JSON selector | `raw_hard_test.true.test_accuracy` | `test.apbr_geo_conditional.test_accuracy` |
| 数值 | 0.7783 | 0.7743 |
| mode / τ | true / 0.01 | true / 0.01 |
| 父训练机制 | vpr | vpr |
| Parent SHA256 | `4332079becdb58fb50f0e68b8bd0dcc251db0184a6050facb524817520a51d82` | `6571e30ed783508f28bc378aa3b22a739bf1ddbbb790db3d98c08858e126e10b` |
| Output SHA256 | `96b52828a7da67b8056dc8e07f0b5bd553666c334e43cb84506f7fa5b8d9eebc` | `de237a1d579e628d1d6998156f0c616f71c7ff2bae6b9da670661694c3f8c28d` |
| Mask SHA256 | `0ff102be87086ea52e6657900b92a52768c4ce3d3250180cb1f6624cff472bb2` | `03c3420db3457609b30b45b184b542f5ae53d03d9164d17e71bed80b9da5ae19` |

两次 adapter SHA相同：
`24b7eb92eae8fbb1e0e9dbf88eafa06de9673898318ffd0770a80a0a1759847a`。

既有汇总 `E/docs/research/sam_apbr_l2_conditional_seed01_20261103/comparison.json`，`rows.apbr_vpr_geometric` 给出 `mean_accuracy_percent=77.63`、`sample_sd_accuracy_percent=0.282842712474623`。

### SAM 77.43%的歧义

`P/tables/ablation.tex:5` 未绑定Seed、父checkpoint或JSON；`P/review/conditional_greedy_revision_20261106/CLAIM_EVIDENCE.md:53`明确其为作者指定值、run provenance待核对。

真实 SAM Seed 2记录：

- `E/artifacts/sam_conditional_seed2_cpu_20261104_v1/results.json`，`test.physical.test_accuracy=0.7743`。
- `E/artifacts/sam_conditional_seed2_newfinal_cpu_20261104_v1/results.json:34`，同字段同值。
- 新Final parent：`070c8d04af80a0bf011647aa8bad65e65e3fd64bedde64780adc029b40b07ec4`。
- 新输出：`3053d320405ca88d449870d29d5e90aa0f7f99379d25c9bfe9321a844f6b2754`。
- Mask：`5b42cce6bf2ff0c45183afbd2122840d27b032c57034e30208f63be1c44ca820`。
- caller传 `tau=0.01, geometry_mode="physical"`，后者使τ不参与选点。

可以确认存在真实SAM产物，**不能确认论文该格绑定哪份记录**。不能断言误抄APBR Seed1，也不能把单Seed2与APBR Seed0/1均值视为配对均值。

## 6. Test前冻结

| Run | runner常量 | preflight | freeze与Test顺序 |
|---|---|---|---|
| Seed0 | `run_apbr_nearoptimal_conditional_seed0_full_cpu.py:40` | L1065 τ=.01 | L231–232检查源码，L249–254写freeze，L257读Test，L261评测 |
| Seed1 | `run_sam_apbr_conditional_seed1_cpu.py:39` | L1038 τ=.01 | L195–196检查源码，L197–200写freeze，L202读Test，L207评测 |

pre_test_freeze未单列τ，但source_sha256绑定含常量的runner；checkpoint payload也记录τ。重算的caller哈希与preflight/freeze/results一致：

- Seed0：`f5371f7a5fffacde6399017b41e26e60431e4c96a84352630845eb58c0aebab8`。
- Seed1：`49085de0b997aa5c4e14b69e55798024ea228ccc95013d8c14c841ecffc6abb6`。
- 共享校准ID：`3b7da27d79762235c87f18571bc73224d6e6f2b3ba663b7d9a4a3de312f23157`。
- 正式YAML：`e4be5df7cf39b1ebd2730999cd6fd42458a5bcc0779e621894b1c588616879d6`。τ不在该YAML中，而是独立runner常量。

2026-11-03文件UTC时间顺序：

| Run | preflight | freeze | results |
|---|---|---|---|
| Seed0 | 14:13:10 | 14:15:07 | 14:16:37 |
| Seed1 | 14:57:00 | 14:59:56 | 15:00:49 |

这些本地交叉证据支持两次完整评测使用固定0.01，并在产物冻结后访问Test。文件时间及程序自报告不是外部防篡改注册；“研究开始即预注册1%”和“从未受任何更早Test观察影响”仍为UNKNOWN。当前材料也不足以指控Test调参。

## 7. 论文是否一致、是否混用

基准稿 [main.tex](C:/Users/earthgrass73/Desktop/NM稀疏/paper/main.tex) L178–183、L213–216，及PDF第5–6页，明确每一步近优筛选并用几何。这与已运行true/.01主路径一致，与本轮无τ定义不同。

另外：

1. Algorithm1未写τ=0时跳过几何的分支，在物理精确平局下不完全对应代码。
2. 固定实际L2训练公式与77.63的VPR父模型不同。`src/vpr_sam/optim/geometry.py`原版为 `rho*A²g/||Ag||`；fixed-L2分支才再归一化到实际半径。
3. 消融SAM77.43未绑定运行和聚合范围，不能作为已验证的配对几何收益。

**没有发现77.63内部混合不同selector或τ。** 两个输入有相同selector SHA和参数。需要分清的是论文方法身份、训练父模型身份及对照身份。

## 8. 处置及未改动证据

工程源码、配置、历史checkpoint、Mask、JSON和原论文保持不变。旧结果保留于本报告；无τ论文采用本轮指定定义，其实验为PENDING。

原稿SHA256：`0e8031274946ddabb8d84f6eb2274e8b4863f0674148893d3e1010211feb989a`。

原PDF SHA256：`be8b402134eb13dda810babb4158ada6317003bf06d3d464abbe3976116a21b8`。
