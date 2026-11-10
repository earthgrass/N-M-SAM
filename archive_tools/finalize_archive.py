from pathlib import Path
import os,json,hashlib,shutil
OUT=Path(__file__).resolve().parent
if os.name=='nt':OUT=Path('\\\\?\\'+str(OUT))
PACK=OUT/'packages_v2'
URL='https://github.com/earthgrass/N-M-SAM/tree/archive/'
names={
 'research-index':'研究存档总索引',
 'apbr-implemented':'原版 APBR / SAM：已实现训练与条件贪心实验',
 'apbr-l2':'APBR-L2：固定全局 L2 半径的独立分支',
 'apbr-paper':'最新 APBR 论文：无 τ 的 APBR-first 方法稿',
 'joint':'Joint APBR / SAM-Joint：训练期二阶指导',
 'obs-explorations':'OBS、Group、Geo、Radius、Shared-Q 后训练探索',
 'blockwise':'块级 APBR v2：保留取消与回传缺口',
 'hc-sam':'HQ-APBR V1 / HC-SAM：完整实验及负结果审计',
}
summary={
'apbr-implemented':'''本分支保存原版 APBR（代码中常称 VPR）与 SAM 的已实现研究及小型运行证据。原版扰动为 `rho A²g / ||Ag||`，不能标记为固定全局 L2 范数。

当前已运行的几何近优条件 OBS：第一次和第二次选择均在物理 OBS 最小代价的 `1+tau` 窗口内按几何分数选择，`tau=0.01`。纯物理条件贪心是另一条对照。最新论文中的“APBR 直接首删、纯条件 OBS 次删、无 tau”单独存放在论文分支，不应从旧结果反推其已实现。

阅读 `inventory/apbr.md` 和 `inventory/apbr.json`；实际运行终点位于 `evidence/experimental_artifacts/`。原始 formal 配置、训练入口、模型和几何实现位于 `project/vpr_sam_research/`。保留历史 Raw/Native 与 Recovery 的原记录，不能把不同终点改称同一种补偿。''',
'apbr-l2':'''本分支保存 APBR-L2 训练与条件贪心后处理。扰动为 `rho A²g / ||A²g||`，实际全局 L2 半径固定；原版 APBR 使用 `||Ag||`，二者必须分别报告。

主要证据：`evidence/experimental_artifacts/apbr_fixed_l2_seed0_20261102/`、`dense_sam_apbr_l2_seed1to4_20261103_local_copy/` 和各 `apbr_l2_nearoptimal_conditional_*` 目录。任务名 seed1to4 是历史计划名，不代表所有计划都完成；以每个 run 的状态、final checkpoint、评测 JSON 为准。

代码入口仍为共享的 `src/vpr_sam/optim/geometry.py` 与 formal training runner，但必须读取对应 fixed-L2 冻结配置。`inventory/apbr.json` 同时解释论文版与已实现版，不得把历史 tau 条件选择重命名成无 tau。''',
'apbr-paper':'''本分支原样保存当前最新论文与来源审计，不修改正文、公式、表格或作者提供的数据。

最新本地论文 Git HEAD：`5ea3a042f44a8bdbf4e842b745536fa026b6d612`。

- LaTeX 主入口：[`manuscript/main.tex`](manuscript/main.tex)。
- 最新已编译 PDF：[`APBR-SAM_geometry_first_20261106_ablation_analysis.pdf`](manuscript/APBR-SAM_geometry_first_20261106_ablation_analysis.pdf)。其它命名 PDF 是历史导出。
- 算法来源审计：[`TAU_ORIGIN_AUDIT.md`](manuscript/TAU_ORIGIN_AUDIT.md)。
- 作者数据更新记录：[`AUTHOR_VALUE_UPDATE_20261106.md`](manuscript/review/AUTHOR_VALUE_UPDATE_20261106.md)。

论文当前规则：APBR 分数 `|w| exp(-s)` 直接决定首次删除；虚拟 OBS 条件化后以物理条件代价选择第二次删除；从当前原始物理状态执行一次联合 OBS；无 tau 窗口。当前清点没有发现已独立验证的匹配执行记录。

表中 `77.63±0.14` 与 `77.43±0.11` 按作者明确复核后的指示保留。旧实现的 tau=.01 结果、作者提供的论文值、最新无 tau 方法稿分别标识，不将它们合成为新的已验证实验。此说明属于存档索引，未加入论文正文。''',
'joint':'''本分支保存训练期二阶指导的 Joint APBR / SAM-Joint，包含完整指导与每十 epoch 指导。静态后训练 `apbr_gpt` 和 `group_joint_obs` 不是该训练队列，另见 OBS 探索分支。

优先阅读 [`inventory/joint.md`](inventory/joint.md)；逐 cohort 评测位置和 SHA256 见 [`inventory/joint.json`](inventory/joint.json)。Seed 0 完整小型证据在 `evidence/workspace/joint_seed0_copy/`；Seed 1 与 epoch10 证据在 `evidence/experimental_artifacts/apbr_gpt_joint_seed1_sparse_v2/`。

保留 Phase 4 的梯度交接失败和 Phase 4.1 修复通过两段历史。顶层 RTX5090 YAML 是早期三分支/Recovery 计划；实际范围以运行 manifest、队列源码及 Final Test 为准。APBR-Joint 使用原版椭球 APBR，不是 APBR-L2。APBR Seed 1 epoch10 已被用户取消，不能自动恢复它。''',
'obs-explorations':'''本分支保留后训练研究的实现与独立证据。先读 [`inventory/explorations.json`](inventory/explorations.json)。

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

`group_joint_obs_seed0_full_20261031` 本地只发现早期 Replay 失败相关小型证据；没有据此确认完整八组成功。失败记录保留，不由后续 Joint 训练结果替代。''',
'blockwise':'''本分支保存 2026-10-28 块级 APBR v2 的源码、配置与审计索引；不能用 2026-10-24 的旧 prototype 代替。

**本地证据不完整。** 清点仅发现 `seed0_v2_transfer_parts/seed0_v2_bundle.tar.part.000`，21,102,592 bytes；只读 tar 检查为 `unexpected end of data`。该分片仅在外部资产清单中记录，没有把它当完整可解包备份。

已知历史 Seeds 1/2 由用户取消。当前存档不声称完整 Seed 0 checkpoint/评测已回传，也不声称可立即恢复。后续应先恢复缺失包并独立验证完整性，而非重跑或从其它研究线借用结果。具体来源见 `inventory/explorations.json`。''',
'hc-sam':'''本分支保存独立 HQ-APBR V1 / HC-SAM 工程、200 Epoch Seed 0、配套稿件及负结果审计。它不是原版 APBR，也不是 APBR-L2 或训练期 Joint。

当前权威入口：[`ROOT_CAUSE_AUDIT_REPORT.md`](project/HQ-APBR-V1/reports/seed0_negative_root_cause_audit_20261105/ROOT_CAUSE_AUDIT_REPORT.md)。工程原 README 的“尚未训练”是历史 CPU 阶段说明，原样留存，不能据此判定最终状态。

已记录 Dense 78.51%、Mask-only 48.06%、Fixed-mask OBS 74.72%、Native SparseGPT 76.59%。不同终点的 Mask 身份与校准证据见审计，不能直接把所有差值归因于同一 Mask 的补偿。

保留 GPU numerical PASS、performance FAIL（30.55% 对 30% 门限）以及用户明确 override；没有把性能 FAIL 改写成 PASS。实际运行证据在 `evidence/hq_artifacts/`，完整训练日志和几何演化摘要保留。''',
}

def dump(p,o):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

assets=json.loads((OUT/'EXTERNAL_ASSETS_SHA256.json').read_text(encoding='utf-8'))
lookup={x['workspace_path']:x for x in assets['assets']}
links='\n'.join(f'| [{names[n]}]({URL}{n}-20261110) | `archive/{n}-20261110` |' for n in names if n!='research-index')
for n,title in names.items():
    p=PACK/n
    if n=='research-index':
        body='''本次为科研暂停前的静态存档。没有运行训练、剪枝、Test，也没有连接或启动服务器。原始工作区和现有 GitHub 分支保持原状。

各研究线作为独立快照分支上传。先根据方法身份选分支，再按运行配置、父 checkpoint 和评测哈希定位结果，不按文件名或单个准确率混用。

| 研究线 | 分支 |
|---|---|
'''+links+'''

## 已知的配套边界

- 最新论文无 tau 的 APBR-first 提案与已运行 tau=.01 几何近优条件 OBS 分开存档；论文原文和作者确认值不改动。
- 原版 APBR 与 APBR-L2 的扰动归一化不同；Joint APBR 使用原版椭球机制。
- Joint 的实际队列与早期计划不同；APBR Seed1 epoch10 的取消状态保留。
- OBS 探索中的 Radius、Geo-rank、六候选 Geo、Shared-Q 各有独立定义和结果。
- Blockwise v2 本地仅找到不完整回传分片；早期 Group Joint 八组的成功全网证据未在已检查目录找到。
- HC-SAM 已完成，保留负结果、性能 FAIL 和明确 override。

## 大型资产

源码、冻结配置、小型结果 JSON/CSV、审计文档及自有稿件已纳入 Git。模型权重、数据集、压缩包和环境没有作为 Git blobs 上传。

`archive_metadata/EXTERNAL_ASSETS_SHA256.json` 是本次实际重新读取并计算 SHA256 的本地资产清单，不是远端权重备份。共 '''+str(assets['count'])+' 个文件，'+f"{assets['bytes']/1024**3:.2f}"+''' GiB。**请继续保留本地权重；仅有 GitHub 存档不能重建这些二进制资产。** 数据集需按原官方身份审计重新准备。
'''
        dump(p/'archive_metadata/EXTERNAL_ASSETS_SHA256.json',assets)
        for f in ['prepare_archive.py','hash_external_assets.py','finalize_archive.py']:
            q=p/'archive_tools'/f;q.parent.mkdir(exist_ok=True);shutil.copyfile(OUT/f,q)
    else:
        body=summary[n]
        missing=json.loads((p/'archive_metadata/OMITTED_LOCAL_ASSETS.json').read_text(encoding='utf-8'))
        selected=[lookup[x['workspace_path']] for x in missing if x['workspace_path'] in lookup]
        dump(p/'archive_metadata/EXTERNAL_ASSETS_SHA256.json',{'scope':'External local bytes, not uploaded to GitHub','assets':selected})
        body+='\n\n总索引：['+names['research-index']+']('+URL+'research-index-20261110)。\n'
    body+='''
## 文件身份和恢复

- `archive_metadata/SOURCE_FILE_MAP.json`：归档路径、原工作区相对路径、原文件 SHA256，复制内容未修改。
- `archive_metadata/SOURCE_REPOSITORIES.json`：原仓库 HEAD、分支和工作区差异。未提交实现以文件哈希标识；旧 HEAD 不代表包含后续所有代码。
- `archive_metadata/EXTERNAL_ASSETS_SHA256.json`：留在本地的大型资产及本次重算哈希。
- `SHA256SUMS.txt`：当前归档全部文件的完整性清单（不含该清单本身和 `.git`）。
- `RESUME.md`：恢复前的核对顺序。

共享源码树是依赖闭包，包含多种研究实现以保留原 import 布局；**存在于源码树不代表本分支运行过该方法**。实际实验身份以本分支 evidence 和 inventory 为准。历史报告里的绝对路径及旧状态原样保留，路径映射见 SOURCE_FILE_MAP。
'''
    (p/'README.md').write_text('# '+title+'\n\n'+body,encoding='utf-8')
    (p/'RESUME.md').write_text('''# 后续恢复入口

1. 从本分支 README 与 inventory 确认方法版本、约束、真实完成/失败/取消状态。归档不会自动启动任何任务。
2. 校验 `SHA256SUMS.txt`。按 `SOURCE_FILE_MAP.json` 查找原路径；不能将不同父模型、Mask、补偿器或论文版本互换。
3. 从本地保存位置取回所需 checkpoint/Mask/Geometry，对照 `EXTERNAL_ASSETS_SHA256.json` 和原 run manifest 的父子哈希；数据集按原官方 CIFAR-100 审计准备。没有文件时先标记缺失，不重训替代。
4. 源码保留原包布局。实验工程在 `project/vpr_sam_research/experiments/post_training_nm_benchmark`，依赖的 VPR 包在 `project/vpr_sam_research/src`；HQ 独立工程在 `project/HQ-APBR-V1`。以本分支实际存在目录为准。
5. 从实际运行的 freeze/config/environment/source identity 文件复原环境与命令。顶层计划配置不一定是最终运行配置；不要直接执行历史队列启动器。历史绝对路径需要在新副本中显式映射，原配置保持不变。
6. 先确认资源与原门禁、再做所需针对性验证。已完成的 Test 不重复执行；失败不自动重试；取消的队列不恢复。本次用户仅授权存档，新的训练或评测需新的明确指令。

此归档进行了文件复制/哈希/身份检查，没有重新运行科研测试。历史测试日志仅代表其记录的代码与环境。
''',encoding='utf-8')
    (p/'.gitignore').write_text('__pycache__/\n*.py[cod]\n.venv/\n.pytest_cache/\n',encoding='utf-8')
    (p/'.gitattributes').write_text('* -text\n',encoding='utf-8')
    dump(p/'archive_metadata/ARCHIVE_STATUS.json',{'date':'2026-11-10','state':'ARCHIVED_FOR_FUTURE_MANUAL_RESUMPTION','new_training':False,'new_pruning':False,'new_test_evaluation':False,'remote_server_connected':False,'experiment_completion_claims':'See run-specific evidence; archival completion is not experiment completion'})
    fs=sorted(f for f in p.rglob('*') if f.is_file() and '.git' not in f.relative_to(p).parts and f!=p/'SHA256SUMS.txt')
    (p/'SHA256SUMS.txt').write_text(''.join(sha(f)+'  '+f.relative_to(p).as_posix()+'\n' for f in fs),encoding='utf-8')
print('Finalized',len(names),'archive branches')
