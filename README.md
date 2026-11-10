# 最新 APBR 论文：无 τ 的 APBR-first 方法稿

本分支原样保存当前最新论文与来源审计，不修改正文、公式、表格或作者提供的数据。

最新本地论文 Git HEAD：`5ea3a042f44a8bdbf4e842b745536fa026b6d612`。

- LaTeX 主入口：[`manuscript/main.tex`](manuscript/main.tex)。
- 最新已编译 PDF：[`APBR-SAM_geometry_first_20261106_ablation_analysis.pdf`](manuscript/APBR-SAM_geometry_first_20261106_ablation_analysis.pdf)。其它命名 PDF 是历史导出。
- 算法来源审计：[`TAU_ORIGIN_AUDIT.md`](manuscript/TAU_ORIGIN_AUDIT.md)。
- 作者数据更新记录：[`AUTHOR_VALUE_UPDATE_20261106.md`](manuscript/review/AUTHOR_VALUE_UPDATE_20261106.md)。

论文当前规则：APBR 分数 `|w| exp(-s)` 直接决定首次删除；虚拟 OBS 条件化后以物理条件代价选择第二次删除；从当前原始物理状态执行一次联合 OBS；无 tau 窗口。当前清点没有发现已独立验证的匹配执行记录。

表中 `77.63±0.14` 与 `77.43±0.11` 按作者明确复核后的指示保留。旧实现的 tau=.01 结果、作者提供的论文值、最新无 tau 方法稿分别标识，不将它们合成为新的已验证实验。此说明属于存档索引，未加入论文正文。

总索引：[研究存档总索引](https://github.com/earthgrass/N-M-SAM/tree/archive/research-index-20261110)。

## 文件身份和恢复

- `archive_metadata/SOURCE_FILE_MAP.json`：归档路径、原工作区相对路径、原文件 SHA256，复制内容未修改。
- `archive_metadata/SOURCE_REPOSITORIES.json`：原仓库 HEAD、分支和工作区差异。未提交实现以文件哈希标识；旧 HEAD 不代表包含后续所有代码。
- `archive_metadata/EXTERNAL_ASSETS_SHA256.json`：留在本地的大型资产及本次重算哈希。
- `SHA256SUMS.txt`：当前归档全部文件的完整性清单（不含该清单本身和 `.git`）。
- `RESUME.md`：恢复前的核对顺序。

共享源码树是依赖闭包，包含多种研究实现以保留原 import 布局；**存在于源码树不代表本分支运行过该方法**。实际实验身份以本分支 evidence 和 inventory 为准。历史报告里的绝对路径及旧状态原样保留，路径映射见 SOURCE_FILE_MAP。
