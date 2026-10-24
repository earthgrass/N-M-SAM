# 决策 C：证据仍不足，先补针对性机制检查

Guided 在两套全部随机对照上 Raw Hard Accuracy/Loss 更好，构成固定模型上积极的 Mask 选择证据；它不是“完全没有信息”。Fine 的 Accuracy 超过随机最大仅1张样本，且 BN 后优势消失/反转，不能据此证明真实删除风险预测信息在标准化 BN 后仍成立。

因此不选择 A 来授权立即进入正式 Block-wise 训练，也没有证据支持 B 的“先改变冻结公式”。独立 Block-wise CPU 原型已实现并通过测试，作为未来设计储备；本次没有训练它。

建议下轮先预注册条件删除损伤预测检验：固定 gap/rank/权重尺度/层，评价局部删除影响与 s 的增量解释；明确 BN 指标为 Raw 与标准化后双端点；再决定是否申请新的稠密训练 seed 和 Block-wise Gamma_B 协议。任何新训练均等待用户明确授权，不自动执行。

已完成 Phase0来源核验、Phase1小型证据归档、Phase2基线/BN审计、Phase3共64组无训练对照、Phase4数学与CPU原型。未执行新200epoch、Gamma搜索、额外训练seed或Test评价。最终发布分支/commit与Draft PR见交付消息；所有历史分支和产物保持不变。
