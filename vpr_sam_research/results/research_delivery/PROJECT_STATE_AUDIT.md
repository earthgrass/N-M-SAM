# Project state audit

## 已核验来源

- VPR-SAM: `40e0c1bd01b94207ea236ea4384d8410362628ca`；工作树 clean=True；与远程分支 HEAD 核对一致。
- VPR-L2-seed0: `f7427339de6d6589f53b0191068448cf642901b2`；工作树 clean=True；与远程分支 HEAD 核对一致。
- VPR-L2-recovery: `f0e3a4acac8650370afb9865ff82a4765d2af676`；工作树 clean=True；与远程分支 HEAD 核对一致。
- VPR-L2-geometry-mask: `492c5976032741c17600acfcbc823bbf7569789b`；工作树 clean=True；与远程分支 HEAD 核对一致。
- VPR-L2-guided-recovery: `1345d24d557e88974eb6cc5cadabf4cae754d742`；工作树 clean=True；与远程分支 HEAD 核对一致。

V1 服务器冻结 d4c58d6，V2 f742733，Guided recovery 1345d24；服务器工作树核对干净。V1 本地 audit-fixes 的 40e0c1b 是随后文档提交，不替代训练 producer。
历史本地完整性报告包含 V1 六条主训练和恢复的 lineage/hash；VPR-L2/三 Mask/Guided recovery 均有 schema2 审计与 checkpoint/geometry/mask 摘要。Dense-SAM seed0 逐 epoch 原日志来自 main200_delivery 原归档，而非其他旧 Dense-SAM 项目。

## 存储与发布边界

- GitHub 既有分支已提交源码与配置；新研究结果不能在推送核验之前声称已发布。新归档工作分支和 Draft PR 见最终提交记录。
- 本地仅有小型日志、JSON/CSV、图表、报告、归档包；不会把本地同步等同 GitHub 发布。
- 大 checkpoint、原 CIFAR 数据、完整 FP64 geometry、完整 Mask 张量留在私有服务器，未上传 GitHub。Checkpoint 获取需用户提供授权存储副本，并按 manifest SHA256 验证；仓库不包含凭据或公开下载地址。
- 通过哈希确认：V1 seed0 日志归档/导出清单/服务器相同；V2 与 Guided epoch 日志、best/final checkpoint 已核验；本轮机制评价严格重放旧指标并对历史文件做前后哈希校验。
- 已有历史产物不重命名、不覆盖、不追溯修改；未发现用户未提交修改。
- 未独立验证服务器之外的大 checkpoint 第二份异地备份；不将本地小证据包视作大权重备份。
- 未授权的新 200 epoch、额外训练 seed、Gamma_B 搜索均未执行。

本轮独立 CPU 重新读取 V1 全部 18 个 dense/hard/recovery final checkpoint：schema2/来源/模型有限值/保存轮次通过，SHA256 与原正式清单一致且前后不变。详见 CURRENT_STATE_CPU_AUDIT.json。
