# APBR-L2：固定全局 L2 半径的独立分支

本分支保存 APBR-L2 训练与条件贪心后处理。扰动为 `rho A²g / ||A²g||`，实际全局 L2 半径固定；原版 APBR 使用 `||Ag||`，二者必须分别报告。

主要证据：`evidence/experimental_artifacts/apbr_fixed_l2_seed0_20261102/`、`dense_sam_apbr_l2_seed1to4_20261103_local_copy/` 和各 `apbr_l2_nearoptimal_conditional_*` 目录。任务名 seed1to4 是历史计划名，不代表所有计划都完成；以每个 run 的状态、final checkpoint、评测 JSON 为准。

代码入口仍为共享的 `src/vpr_sam/optim/geometry.py` 与 formal training runner，但必须读取对应 fixed-L2 冻结配置。`inventory/apbr.json` 同时解释论文版与已实现版，不得把历史 tau 条件选择重命名成无 tau。

总索引：[研究存档总索引](https://github.com/earthgrass/N-M-SAM/tree/archive/research-index-20261110)。

## 文件身份和恢复

- `archive_metadata/SOURCE_FILE_MAP.json`：归档路径、原工作区相对路径、原文件 SHA256，复制内容未修改。
- `archive_metadata/SOURCE_REPOSITORIES.json`：原仓库 HEAD、分支和工作区差异。未提交实现以文件哈希标识；旧 HEAD 不代表包含后续所有代码。
- `archive_metadata/EXTERNAL_ASSETS_SHA256.json`：留在本地的大型资产及本次重算哈希。
- `SHA256SUMS.txt`：当前归档全部文件的完整性清单（不含该清单本身和 `.git`）。
- `RESUME.md`：恢复前的核对顺序。

共享源码树是依赖闭包，包含多种研究实现以保留原 import 布局；**存在于源码树不代表本分支运行过该方法**。实际实验身份以本分支 evidence 和 inventory 为准。历史报告里的绝对路径及旧状态原样保留，路径映射见 SOURCE_FILE_MAP。

嵌套 `.gitattributes` 原样保存为 `.gitattributes.source`，避免 Git 换行转换影响 SHA256；文件映射记录了改名。
