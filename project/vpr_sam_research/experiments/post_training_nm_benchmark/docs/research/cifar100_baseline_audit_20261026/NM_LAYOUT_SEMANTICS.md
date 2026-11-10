# N:M 布局、坐标与尾部语义

## 当前项目统一约束

本轮统一比较候选沿用**当前 OBC 冻结配置**，不是把项目级“所有未来正式实验均已批准”作默认：

- 2:4 表示每个完整四元组保留 2、删除 2；
- Conv2d 权重采用 OIHW；对每个输出通道独立，将 `(input-channel-within-group, kh, kw)` 按 PyTorch contiguous 顺序展平；
- 每个 output row 的连续四列成组；不完整尾部保持 dense；
- 首个 Conv2d、最后 Linear、BN affine 和 bias 排除；
- 统一适配的 exact tie 使用 stable ascending index：较低索引先被删除；
- bool mask 表示 keep，不能从 `weight != 0` 反推，因为被保留权重也可能数值为零。

形式化地，Conv2d 坐标映射为

`j = ((i_local * kH) + kh) * kW + kw`，`Wmat[o,j] = W[o,i_local,kh,kw]`。

组卷积中 `i_local∈[0,Cin/groups)`；每个卷积 group 有独立 patch matrix/Hessian，不能把不同输入通道组错误共享为一个 H。

## 原生实现与统一实现对照

| 方法 | 原生 Linear N:M | 原生 Conv2d | 尾部/tie | 统一 2:4 变化 |
|---|---|---|---|---|
| Wanda LLM | output row 连续 input columns；`prune_n`=删除数 | 无有效官方 CNN N:M；视觉入口 Linear-only | partial slice 仍 top-k；短于 prune_n 可失败；top-k tie 未冻结 | 完整组同一可行集合；tail 改为 dense，tie 显式 stable；Conv 用真实 patch norm |
| SparseGPT | 同上，但 mask 对补偿中的当前 W 动态选择 | 构造函数 flatten OIHW，统计未 unfold，故不成立 | 同一 partial slice 风险；block 边界须与 M 对齐 | 保留逐列 OBS 更新；tail dense；每个 conv group 独立 H；blocksize 强制整除 M |
| ROSE | 论文 2:4 把四元组作为重排单位；源码 N:M 路径有未定义变量 | 无有效官方 Conv 路径 | 论文 Eq.(7) 与源码分母不同 | 按论文：组内列重排+整组重排，SparseGPT 后逆置换；修复版而非官方原版 |

## 各层属性

1. **普通 Linear**：`W[out,in]`，激活矩阵为 `[in, vectors]`；三种适配均支持。
2. **普通 Conv2d**：`Wmat[out,Cin*kH*kW]`，`Xpatch[Cin*kH*kW,N*L]`；支持 stride、padding、dilation，因为它们只改变 patch 抽取和 L，不改变权重坐标。
3. **grouped Conv2d**：按 group 分出 `Wg[Cout/G,Cin/G*kH*kW]` 与 `Xg[...]`，独立统计、独立剪枝和补偿；支持。
4. **depthwise Conv2d**：是 `groups=Cin` 的特例；数学映射支持。若 `kH*kW<4`，没有完整 2:4 组，整层保持 dense；这不是 50% 稀疏层。
5. **bias**：前向映射中在矩阵乘法后逐输出通道加；不参与 mask、Wanda 分数或 Hessian 权重补偿。
6. **不连续张量**：输入由 `F.unfold` 定义语义，权重只使用合法 reshape；测试覆盖非连续 NCHW 输入。

## ROSE 为什么不改变统一可行 mask 集合

受控适配只允许两类置换：(a) 一个原始四元组内部的列置换；(b) 完整四元组作为整体的置换。SparseGPT 在置换空间对每个连续四元组删两项，最后逆置换。因此每个原始四元组仍恰好保留两项。若允许任意跨组列置换，最终可行 mask 集合会改变，本实现明确禁止。

## 结论

- Wanda/SparseGPT 原生 LLM 2:4 与项目完整组的**集合定义**在宽度整除 4 时一致，但 tie、tail 和 Conv 坐标语义并不自动一致。
- ResNet18 的标准 3×3 Conv 每行宽度通常可被 4 整除取决于输入通道；首层 `3*3*3=27` 还有尾部且本来排除。1×1 shortcut 宽度等于输入通道，当前架构中通常可整除 4。
- `MASK_VALIDATION_PASS` 只说明结构合法；SparseGPT/ROSE 还必须验证补偿写回坐标，本轮已测试该项。
