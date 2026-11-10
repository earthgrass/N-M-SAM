# 原版 APBR / SAM：已实现训练与条件贪心实验

本分支保存原版 APBR（代码中常称 VPR）与 SAM 的已实现研究及小型运行证据。原版扰动为 `rho A²g / ||Ag||`，不能标记为固定全局 L2 范数。

当前已运行的几何近优条件 OBS：第一次和第二次选择均在物理 OBS 最小代价的 `1+tau` 窗口内按几何分数选择，`tau=0.01`。纯物理条件贪心是另一条对照。最新论文中的“APBR 直接首删、纯条件 OBS 次删、无 tau”单独存放在论文分支，不应从旧结果反推其已实现。

阅读 `inventory/apbr.md` 和 `inventory/apbr.json`；实际运行终点位于 `evidence/experimental_artifacts/`。原始 formal 配置、训练入口、模型和几何实现位于 `project/vpr_sam_research/`。保留历史 Raw/Native 与 Recovery 的原记录，不能把不同终点改称同一种补偿。

总索引：[研究存档总索引](https://github.com/earthgrass/N-M-SAM/tree/archive/research-index-20261110)。

## 文件身份和恢复

- `archive_metadata/SOURCE_FILE_MAP.json`：归档路径、原工作区相对路径、原文件 SHA256，复制内容未修改。
- `archive_metadata/SOURCE_REPOSITORIES.json`：原仓库 HEAD、分支和工作区差异。未提交实现以文件哈希标识；旧 HEAD 不代表包含后续所有代码。
- `archive_metadata/EXTERNAL_ASSETS_SHA256.json`：留在本地的大型资产及本次重算哈希。
- `SHA256SUMS.txt`：当前归档全部文件的完整性清单（不含该清单本身和 `.git`）。
- `RESUME.md`：恢复前的核对顺序。

共享源码树是依赖闭包，包含多种研究实现以保留原 import 布局；**存在于源码树不代表本分支运行过该方法**。实际实验身份以本分支 evidence 和 inventory 为准。历史报告里的绝对路径及旧状态原样保留，路径映射见 SOURCE_FILE_MAP。

嵌套 `.gitattributes` 原样保存为 `.gitattributes.source`，避免 Git 换行转换影响 SHA256；文件映射记录了改名。
