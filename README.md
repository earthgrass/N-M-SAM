# 块级 APBR v2：保留取消与回传缺口

本分支保存 2026-10-28 块级 APBR v2 的源码、配置与审计索引；不能用 2026-10-24 的旧 prototype 代替。

**本地证据不完整。** 清点仅发现 `seed0_v2_transfer_parts/seed0_v2_bundle.tar.part.000`，21,102,592 bytes；只读 tar 检查为 `unexpected end of data`。该分片仅在外部资产清单中记录，没有把它当完整可解包备份。

已知历史 Seeds 1/2 由用户取消。当前存档不声称完整 Seed 0 checkpoint/评测已回传，也不声称可立即恢复。后续应先恢复缺失包并独立验证完整性，而非重跑或从其它研究线借用结果。具体来源见 `inventory/explorations.json`。

总索引：[研究存档总索引](https://github.com/earthgrass/N-M-SAM/tree/archive/research-index-20261110)。

## 文件身份和恢复

- `archive_metadata/SOURCE_FILE_MAP.json`：归档路径、原工作区相对路径、原文件 SHA256，复制内容未修改。
- `archive_metadata/SOURCE_REPOSITORIES.json`：原仓库 HEAD、分支和工作区差异。未提交实现以文件哈希标识；旧 HEAD 不代表包含后续所有代码。
- `archive_metadata/EXTERNAL_ASSETS_SHA256.json`：留在本地的大型资产及本次重算哈希。
- `SHA256SUMS.txt`：当前归档全部文件的完整性清单（不含该清单本身和 `.git`）。
- `RESUME.md`：恢复前的核对顺序。

共享源码树是依赖闭包，包含多种研究实现以保留原 import 布局；**存在于源码树不代表本分支运行过该方法**。实际实验身份以本分支 evidence 和 inventory 为准。历史报告里的绝对路径及旧状态原样保留，路径映射见 SOURCE_FILE_MAP。

嵌套 `.gitattributes` 原样保存为 `.gitattributes.source`，避免 Git 换行转换影响 SHA256；文件映射记录了改名。
