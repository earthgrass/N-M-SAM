# 研究存档总索引

本次为科研暂停前的静态存档。没有运行训练、剪枝、Test，也没有连接或启动服务器。原始工作区和现有 GitHub 分支保持原状。

各研究线作为独立快照分支上传。先根据方法身份选分支，再按运行配置、父 checkpoint 和评测哈希定位结果，不按文件名或单个准确率混用。

| 研究线 | 分支 |
|---|---|
| [原版 APBR / SAM：已实现训练与条件贪心实验](https://github.com/earthgrass/N-M-SAM/tree/archive/apbr-implemented-20261110) | `archive/apbr-implemented-20261110` |
| [APBR-L2：固定全局 L2 半径的独立分支](https://github.com/earthgrass/N-M-SAM/tree/archive/apbr-l2-20261110) | `archive/apbr-l2-20261110` |
| [最新 APBR 论文：无 τ 的 APBR-first 方法稿](https://github.com/earthgrass/N-M-SAM/tree/archive/apbr-paper-20261110) | `archive/apbr-paper-20261110` |
| [Joint APBR / SAM-Joint：训练期二阶指导](https://github.com/earthgrass/N-M-SAM/tree/archive/joint-20261110) | `archive/joint-20261110` |
| [OBS、Group、Geo、Radius、Shared-Q 后训练探索](https://github.com/earthgrass/N-M-SAM/tree/archive/obs-explorations-20261110) | `archive/obs-explorations-20261110` |
| [块级 APBR v2：保留取消与回传缺口](https://github.com/earthgrass/N-M-SAM/tree/archive/blockwise-20261110) | `archive/blockwise-20261110` |
| [HQ-APBR V1 / HC-SAM：完整实验及负结果审计](https://github.com/earthgrass/N-M-SAM/tree/archive/hc-sam-20261110) | `archive/hc-sam-20261110` |

## 已知的配套边界

- 最新论文无 tau 的 APBR-first 提案与已运行 tau=.01 几何近优条件 OBS 分开存档；论文原文和作者确认值不改动。
- 原版 APBR 与 APBR-L2 的扰动归一化不同；Joint APBR 使用原版椭球机制。
- Joint 的实际队列与早期计划不同；APBR Seed1 epoch10 的取消状态保留。
- OBS 探索中的 Radius、Geo-rank、六候选 Geo、Shared-Q 各有独立定义和结果。
- Blockwise v2 本地仅找到不完整回传分片；早期 Group Joint 八组的成功全网证据未在已检查目录找到。
- HC-SAM 已完成，保留负结果、性能 FAIL 和明确 override。

## 大型资产

源码、冻结配置、小型结果 JSON/CSV、审计文档及自有稿件已纳入 Git。模型权重、数据集、压缩包和环境没有作为 Git blobs 上传。

`archive_metadata/EXTERNAL_ASSETS_SHA256.json` 是本次实际重新读取并计算 SHA256 的本地资产清单，不是远端权重备份。共 901 个文件，24.16 GiB。**请继续保留本地权重；仅有 GitHub 存档不能重建这些二进制资产。** 数据集需按原官方身份审计重新准备。

## 文件身份和恢复

- `archive_metadata/SOURCE_FILE_MAP.json`：归档路径、原工作区相对路径、原文件 SHA256，复制内容未修改。
- `archive_metadata/SOURCE_REPOSITORIES.json`：原仓库 HEAD、分支和工作区差异。未提交实现以文件哈希标识；旧 HEAD 不代表包含后续所有代码。
- `archive_metadata/EXTERNAL_ASSETS_SHA256.json`：留在本地的大型资产及本次重算哈希。
- `SHA256SUMS.txt`：当前归档全部文件的完整性清单（不含该清单本身和 `.git`）。
- `RESUME.md`：恢复前的核对顺序。

共享源码树是依赖闭包，包含多种研究实现以保留原 import 布局；**存在于源码树不代表本分支运行过该方法**。实际实验身份以本分支 evidence 和 inventory 为准。历史报告里的绝对路径及旧状态原样保留，路径映射见 SOURCE_FILE_MAP。
