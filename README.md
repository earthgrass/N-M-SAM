# OBS、Group、Geo、Radius、Shared-Q 后训练探索

本分支保留后训练研究的实现与独立证据。先读 [`inventory/explorations.json`](inventory/explorations.json)。

| 名称 | 不可混淆的定义 |
|---|---|
| Original SparseGPT | 原生动态逐权重选择与逐列 OBS |
| Radius-Guided | 原生评分乘 `exp(-s)` |
| Geo-rank | 原生评分乘 `exp(-2s)`，不是六候选枚举 |
| Dynamic Group-OBS | 六种删除对的联合代价，组级补偿 |
| Geo-Select | 归一化权重选六对，物理权重补偿 |
| Geo-Full | 归一化坐标选择与补偿，再映回物理坐标 |
| Fixed-mask OBS | 从原 Dense 权重出发，固定外部 Mask |
| Shared-Q / Exact-Q | 独立几何正则补偿研究，不是 Joint 训练 |

`apbr_unified_gpt_full_experiment/shared_q_gate_summary.json` 才是后续 Shared-Q 19 层、33 个训练侧诊断终点的门禁记录；早期 `summary.json` 是 Exact Pilot 阶段。Shared-Q 为负门禁，未访问 Final Test。

`group_joint_obs_seed0_full_20261031` 本地只发现早期 Replay 失败相关小型证据；没有据此确认完整八组成功。失败记录保留，不由后续 Joint 训练结果替代。

总索引：[研究存档总索引](https://github.com/earthgrass/N-M-SAM/tree/archive/research-index-20261110)。

## 文件身份和恢复

- `archive_metadata/SOURCE_FILE_MAP.json`：归档路径、原工作区相对路径、原文件 SHA256，复制内容未修改。
- `archive_metadata/SOURCE_REPOSITORIES.json`：原仓库 HEAD、分支和工作区差异。未提交实现以文件哈希标识；旧 HEAD 不代表包含后续所有代码。
- `archive_metadata/EXTERNAL_ASSETS_SHA256.json`：留在本地的大型资产及本次重算哈希。
- `SHA256SUMS.txt`：当前归档全部文件的完整性清单（不含该清单本身和 `.git`）。
- `RESUME.md`：恢复前的核对顺序。

共享源码树是依赖闭包，包含多种研究实现以保留原 import 布局；**存在于源码树不代表本分支运行过该方法**。实际实验身份以本分支 evidence 和 inventory 为准。历史报告里的绝对路径及旧状态原样保留，路径映射见 SOURCE_FILE_MAP。

嵌套 `.gitattributes` 原样保存为 `.gitattributes.source`，避免 Git 换行转换影响 SHA256；文件映射记录了改名。
