# HQ-APBR V1 / HC-SAM：完整实验及负结果审计

本分支保存独立 HQ-APBR V1 / HC-SAM 工程、200 Epoch Seed 0、配套稿件及负结果审计。它不是原版 APBR，也不是 APBR-L2 或训练期 Joint。

当前权威入口：[`ROOT_CAUSE_AUDIT_REPORT.md`](project/HQ-APBR-V1/reports/seed0_negative_root_cause_audit_20261105/ROOT_CAUSE_AUDIT_REPORT.md)。工程原 README 的“尚未训练”是历史 CPU 阶段说明，原样留存，不能据此判定最终状态。

已记录 Dense 78.51%、Mask-only 48.06%、Fixed-mask OBS 74.72%、Native SparseGPT 76.59%。不同终点的 Mask 身份与校准证据见审计，不能直接把所有差值归因于同一 Mask 的补偿。

保留 GPU numerical PASS、performance FAIL（30.55% 对 30% 门限）以及用户明确 override；没有把性能 FAIL 改写成 PASS。实际运行证据在 `evidence/hq_artifacts/`，完整训练日志和几何演化摘要保留。

总索引：[研究存档总索引](https://github.com/earthgrass/N-M-SAM/tree/archive/research-index-20261110)。

## 文件身份和恢复

- `archive_metadata/SOURCE_FILE_MAP.json`：归档路径、原工作区相对路径、原文件 SHA256，复制内容未修改。
- `archive_metadata/SOURCE_REPOSITORIES.json`：原仓库 HEAD、分支和工作区差异。未提交实现以文件哈希标识；旧 HEAD 不代表包含后续所有代码。
- `archive_metadata/EXTERNAL_ASSETS_SHA256.json`：留在本地的大型资产及本次重算哈希。
- `SHA256SUMS.txt`：当前归档全部文件的完整性清单（不含该清单本身和 `.git`）。
- `RESUME.md`：恢复前的核对顺序。

共享源码树是依赖闭包，包含多种研究实现以保留原 import 布局；**存在于源码树不代表本分支运行过该方法**。实际实验身份以本分支 evidence 和 inventory 为准。历史报告里的绝对路径及旧状态原样保留，路径映射见 SOURCE_FILE_MAP。

嵌套 `.gitattributes` 原样保存为 `.gitattributes.source`，避免 Git 换行转换影响 SHA256；文件映射记录了改名。
