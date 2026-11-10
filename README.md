# Joint APBR / SAM-Joint：训练期二阶指导

本分支保存训练期二阶指导的 Joint APBR / SAM-Joint，包含完整指导与每十 epoch 指导。静态后训练 `apbr_gpt` 和 `group_joint_obs` 不是该训练队列，另见 OBS 探索分支。

优先阅读 [`inventory/joint.md`](inventory/joint.md)；逐 cohort 评测位置和 SHA256 见 [`inventory/joint.json`](inventory/joint.json)。Seed 0 完整小型证据在 `evidence/workspace/joint_seed0_copy/`；Seed 1 与 epoch10 证据在 `evidence/experimental_artifacts/apbr_gpt_joint_seed1_sparse_v2/`。

保留 Phase 4 的梯度交接失败和 Phase 4.1 修复通过两段历史。顶层 RTX5090 YAML 是早期三分支/Recovery 计划；实际范围以运行 manifest、队列源码及 Final Test 为准。APBR-Joint 使用原版椭球 APBR，不是 APBR-L2。APBR Seed 1 epoch10 已被用户取消，不能自动恢复它。

总索引：[研究存档总索引](https://github.com/earthgrass/N-M-SAM/tree/archive/research-index-20261110)。

## 文件身份和恢复

- `archive_metadata/SOURCE_FILE_MAP.json`：归档路径、原工作区相对路径、原文件 SHA256，复制内容未修改。
- `archive_metadata/SOURCE_REPOSITORIES.json`：原仓库 HEAD、分支和工作区差异。未提交实现以文件哈希标识；旧 HEAD 不代表包含后续所有代码。
- `archive_metadata/EXTERNAL_ASSETS_SHA256.json`：留在本地的大型资产及本次重算哈希。
- `SHA256SUMS.txt`：当前归档全部文件的完整性清单（不含该清单本身和 `.git`）。
- `RESUME.md`：恢复前的核对顺序。

共享源码树是依赖闭包，包含多种研究实现以保留原 import 布局；**存在于源码树不代表本分支运行过该方法**。实际实验身份以本分支 evidence 和 inventory 为准。历史报告里的绝对路径及旧状态原样保留，路径映射见 SOURCE_FILE_MAP。

嵌套 `.gitattributes` 原样保存为 `.gitattributes.source`，避免 Git 换行转换影响 SHA256；文件映射记录了改名。
