"""Build public, small research evidence from verified private delivery roots.

Usage: python mechanism_study/archive_results.py --workspace /private/workspace
No model loading, network access, training, or edits to historical inputs.
"""
import argparse,csv,hashlib,json,re,shutil,subprocess,tarfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def table(p,rows):
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def md(p,text): p.write_text(text+'\n',encoding='utf-8')

def main(workspace):
    root=Path(__file__).resolve().parents[1]; out=root/'results/research_delivery';out.mkdir(exist_ok=True,parents=True)
    historical=workspace/'VPR-SAM/vpr_sam_research/artifacts/main200_delivery'
    v2=workspace/'VPR-L2-seed0/vpr_sam_research'
    guided=workspace/'VPR-L2-guided-recovery/vpr_sam_research/artifacts/vpr_l2_guided_recovery_seed0'
    prior=load(historical/'evidence/artifacts/main200_final_delivery/final_audit.json')
    source_receipts={}
    with tarfile.open(historical/'metadata_and_figures.tar.gz') as t:
        def get(name):
            b=t.extractfile(name).read();source_receipts['v1_archive/'+name]=hashlib.sha256(b).hexdigest();return json.loads(b)
        samdir='runs/main200_seeds012_v1/sam_seed0/pruning/n2_m4_exclude_first_last_magnitude_dense_final/recovery/'
        sam=get(samdir+'epochs.json');samcfg=get(samdir+'config.json');sammanifest=get(samdir+'data_manifest.json');samsummary=get(samdir+'summary.json')
        exported=load(historical/'evidence/artifacts/main200_final_delivery/export_manifest.json')
        assert source_receipts['v1_archive/'+samdir+'epochs.json']==exported[samdir+'epochs.json']['sha256']
    recovery=v2/'runs/vpr_l2_seed0_v2/fixed_l2_seed0/pruning/n2_m4_exclude_first_last_magnitude_dense_final/recovery'
    mag=load(recovery/'epochs.json');grec=load(guided/'imported_guided_raw/recovery/epochs.json')
    rows=[r for r in prior['records'] if r['seed']==0 and r['method']=='sam'];assert len(rows)==1;samrecord=rows[0]
    oldmask=load(v2/'artifacts/geometry_guided_mask_seed0/final_audit.json')
    maskmetrics={r['method']:r for r in oldmask['methods']}
    mechanism=out/'mechanism';bn_sam=load(mechanism/'DENSE_SAM_BN.json')
    metrics=[];epochs=[]
    for name,rr,record,dense,bn in [('dense_sam_magnitude',sam,None,samrecord['dense'],bn_sam['bn']),('vpr_l2_magnitude',mag,maskmetrics['magnitude'],oldmask['dense'],maskmetrics['magnitude']['bn_calibrated']),('vpr_l2_geometry_guided',grec,maskmetrics['geometry_guided'],oldmask['dense'],maskmetrics['geometry_guided']['bn_calibrated'])]:
        assert [r['epoch'] for r in rr]==list(range(1,21));best=max(rr,key=lambda r:r['validation']['accuracy']);last=rr[-1]['validation']
        raw=samrecord['hard'] if record is None else record['raw']
        metrics.append(dict(method=name,seed=0,dense_accuracy=dense['accuracy'],dense_loss=dense['loss'],raw_accuracy=raw['accuracy'],raw_loss=raw['loss'],bn_accuracy=bn['accuracy'],bn_loss=bn['loss'],best_accuracy=best['validation']['accuracy'],best_epoch=best['epoch'],best_epoch_loss=best['validation']['loss'],final_accuracy=last['accuracy'],final_loss=last['loss'],dense_to_hard_drop=dense['accuracy']-raw['accuracy'],dense_to_recovery_final_drop=dense['accuracy']-last['accuracy'],dense_to_recovery_best_drop=dense['accuracy']-best['validation']['accuracy']))
        for r in rr:epochs.append(dict(method=name,epoch=r['epoch'],train_loss=r['train_loss'],validation_accuracy=r['validation']['accuracy'],validation_loss=r['validation']['loss']))
        d=out/name;d.mkdir(exist_ok=True)
        dump(d/'epochs.json',rr)
    table(out/'BASELINE_COMPARISON.csv',metrics);table(out/'RECOVERY_EPOCHS.csv',epochs)
    for name,d in [('vpr_l2_magnitude',recovery),('vpr_l2_geometry_guided',guided/'imported_guided_raw/recovery')]:
        cfg=load(d/'config.json');manifest=load(d/'data_manifest.json');summary=load(d/'summary.json')
        assert manifest==sammanifest and summary['order_hash']==samsummary['order_hash']
        for key in ['seed','model','data','optimizer','training','recovery','evaluation','device','threads','deterministic']:assert cfg[key]==samcfg[key],key
        source_receipts[name+'/epochs.json']=sha(d/'epochs.json')
        for f in ['config.json','summary.json','data_manifest.json']:source_receipts[name+'/'+f]=sha(d/f)
        publiccfg=dict(cfg);publiccfg['output']='PRIVATE_RUN_OUTPUT'
        dump(out/name/'recovery_config.json',publiccfg)
    publiccfg=dict(samcfg);publiccfg['output']='PRIVATE_RUN_OUTPUT';dump(out/'dense_sam_magnitude/recovery_config.json',publiccfg)
    v1rows=[]
    for r in prior['records']:
        v1rows.append(dict(seed=r['seed'],method=r['method'],dense_accuracy=r['dense']['accuracy'],dense_loss=r['dense']['loss'],raw_accuracy=r['hard']['accuracy'],raw_loss=r['hard']['loss'],recovery_final_accuracy=r['recovered']['accuracy'],recovery_final_loss=r['recovered']['loss'],commit=r['commit'],dense_sha256=r['checkpoint_hashes']['dense'],hard_sha256=r['checkpoint_hashes']['hard'],recovery_sha256=r['checkpoint_hashes']['recovered']))
    table(root/'results/v1_main200/per_seed_metrics.csv',v1rows)
    shutil.copy2(historical/'seed_metrics.csv',root/'results/v1_main200/original_seed_metrics.csv')
    dump(root/'results/v1_main200/provenance.json',dict(commit=prior['commit'],checkpoints=[{k:v for k,v in r.items() if k.endswith('sha256') or k in ['seed','method']} for r in v1rows],original_final_audit_sha256=sha(historical/'evidence/artifacts/main200_final_delivery/final_audit.json')))
    flatmask=[]
    for r in oldmask['methods']:flatmask.append(dict(method=r['method'],raw_accuracy=r['raw']['accuracy'],raw_loss=r['raw']['loss'],bn_accuracy=r['bn_calibrated']['accuracy'],bn_loss=r['bn_calibrated']['loss'],mask_sha256=r['mask_sha256'],**r['global_deletion']))
    table(root/'results/vpr_l2_seed0/three_mask_comparison.csv',flatmask);table(root/'results/vpr_l2_seed0/recovery_comparison.csv',metrics)
    dump(root/'results/vpr_l2_seed0/provenance.json',dict(dense_commit=oldmask['producer_commit'],mask_commit=oldmask['diagnostic_commit'],guided_recovery_commit='1345d24d557e88974eb6cc5cadabf4cae754d742',dense_sha256=oldmask['final_checkpoint_sha256'],geometry_sha256=oldmask['geometry_sha256'],guided_recovery_sha256=load(guided/'INTEGRITY_AUDIT.json')['final_checkpoint_sha256']))
    # Source custody; original filenames stay logical, never disclose host/account paths.
    heads={}
    for name in ['VPR-SAM','VPR-L2-seed0','VPR-L2-recovery','VPR-L2-geometry-mask','VPR-L2-guided-recovery']:
        repo=workspace/name;heads[name]=dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),clean=not bool(subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()))
    assert all(x['clean'] for x in heads.values())
    dump(out/'EXPERIMENT_PROVENANCE_MANIFEST.json',dict(historical_workspaces=heads,inputs_sha256=source_receipts,v1_commit=prior['commit'],v2_commit=oldmask['producer_commit'],dataset_fingerprint=samsummary['dataset_fingerprint'],recovery_order_hash=samsummary['order_hash'],initialization_hash=samsummary['initialization_hash'],checkpoints_on_private_server_only=True,no_test_metrics=True,public_path_policy='logical run IDs only; no private host/account/UUID',v1_checkpoint_hashes=[r['checkpoint_hashes'] for r in prior['records']]))
    # Numerical report for BOTH frozen random ensembles, with explicit mask-sensitivity scope.
    ensemble_stats={}
    for version,folder in [('coarse','mechanism'),('fine','mechanism_fine')]:
        d=out/folder
        if not d.exists():continue
        rr=list(csv.DictReader((d/'MATCHED_RANDOM_RESULTS.csv').open()));assert len(rr)==34
        summaries=[]
        for key in ['raw_accuracy','raw_loss','raw_loss_damage','bn_accuracy','bn_loss','bn_loss_damage','D2','DA']:
            a=np.array([float(r[key]) for r in rr[2:]]);g=float(rr[1][key]);q=np.quantile(a,[.025,.25,.5,.75,.975])
            summaries.append(dict(metric=key,mean=float(a.mean()),std_sample=float(a.std(ddof=1)),min=float(a.min()),q025=q[0],q25=q[1],median=q[2],q75=q[3],q975=q[4],max=float(a.max()),guided=g,random_ge_guided=int((a>=g).sum()),random_le_guided=int((a<=g).sum())))
        table(d/'DISTRIBUTION_SUMMARY.csv',summaries);ensemble_stats[version]=summaries
    table(out/'MATCHED_RANDOM_RESULTS.csv',[dict(ensemble=v,**r) for v,f in [('coarse','mechanism'),('fine','mechanism_fine')] if (out/f).exists() for r in csv.DictReader((out/f/'MATCHED_RANDOM_RESULTS.csv').open())])
    shutil.copy2(mechanism/'MASK_FLIP_ANALYSIS.csv',out/'MASK_FLIP_ANALYSIS.csv')
    flips=list(csv.DictReader((mechanism/'MASK_FLIP_ANALYSIS.csv').open()));nflip=sum(int(r['flipped_coordinates']) for r in flips);nblocks=sum(int(r['changed_blocks']) for r in flips);total=sum(int(r['total_blocks'])*4 for r in flips)
    pred={}
    for m in ['magnitude','geometry_guided']:pred[m]={int(r['id']):r for r in csv.DictReader((mechanism/(m+'_raw_predictions.csv')).open())}
    assert pred['magnitude'].keys()==pred['geometry_guided'].keys()
    corrected=sum(pred['magnitude'][i]['correct']=='False' and pred['geometry_guided'][i]['correct']=='True' for i in pred['magnitude'])
    harmed=sum(pred['magnitude'][i]['correct']=='True' and pred['geometry_guided'][i]['correct']=='False' for i in pred['magnitude'])
    dump(out/'sample_correctness_flips.json',dict(corrected=corrected,harmed=harmed,net=corrected-harmed,validation_count=5000,scope='Raw Hard; paired sample descriptions, no causal significance claim'))
    lines=['# Matched-random Mask 机制报告','',f'同一 VPR-L2 seed 0 final，{nflip} 个坐标改变（{100*nflip/total:.9f}%），涉及 {nblocks} 个完整四元组。所有变化均为 2 坐标交换；没有 4 坐标交换。',
           '每层详细分布、rank-pattern、边界 gap、s 极差及半径比见 CSV/JSON。BN affine 不参与 Mask。', '',
           '## 两套预冻结随机对照','',
           'Coarse：32 个 PCG64 seeds 10000–10031，严格匹配 layer×20 分位 gap 箱×rank-pattern。首次审计发现近零 gap 箱较宽，因此完整保留该组，并在 fine 评价前独立冻结固定对数 gap 箱（相邻正边界比 10^0.1）；Fine 同样 32 个 seeds。没有按验证结果选择 seeds、丢弃对照或替换 coarse 结果。',
           '两组均匹配每层改动块数、2/4 坐标交换数和实际删除/添加的 Magnitude rank-pattern。随机候选池包含 Guided 块，精细箱可能使重合率较高；必须联合 overlap 与 gap 分布解释。不是完全独立于 Guided 的不相交随机集合。',
           '', '| Ensemble | Metric | Random mean | SD | Min | Median | Max | Guided | Random ≥ Guided | Random ≤ Guided |','|---|---|---|---|---|---|---|---|---|']
    for v,ss in ensemble_stats.items():
        for s in ss:lines.append('| '+v+' | '+s['metric']+' | '+' | '.join(f"{s[k]:.10g}" for k in ['mean','std_sample','min','median','max','guided','random_ge_guided','random_le_guided'])+' |')
    lines+=['','完整 2.5/25/50/75/97.5 分位数、最优/最差及全部 64 个对照见各组 DISTRIBUTION_SUMMARY.csv 与 MATCHED_RANDOM_RESULTS.csv。Accuracy 为比例，乘 100 得百分数；随机组是固定模型上的 Mask 敏感性分布，不是训练多 seed 分布。',
            f'Raw Hard 样本翻转：Guided 修正 {corrected} 张，新增错误 {harmed} 张，净改善 {corrected-harmed} 张；这是配对样本描述，不解释为跨训练 seed 统计显著。',
            '几何删除代价 DA 按评分定义被 Guided 最小化，不能用它的降低自行证明真实 Loss 改善；D2 与真实 loss damage 分开记录。',
            '## 机制判断','',
            '1. learned geometry 确实改变块内排序，但只影响非常少的四元组。变化多在 near-tie 边界且分散于多个层。',
            '2. Raw Hard 收益是否超出随机范围需同时看 coarse/fine；不能仅引用粗箱结果。',
            '3. BN 重校准后 Guided 相对 Magnitude 的 Accuracy 优势消失并反转，Loss 也略差。这说明 Raw 收益很大部分与保留的 BN buffers 和激活分布兼容性有关；校准干预支持该机制，但不能断言唯一原因。',
            '4. 当前证据不证明 s 是独立于 |w| 的真实有限删除风险预测器：rank-pattern、gap、权重尺度、层及 BN 效应仍可能混杂。',
            '5. 允许工程原型不等于授权 Block-wise 正式训练。后续需预注册更针对性的条件信息检验与多 seed；本次不训练。']
    md(out/'MATCHED_RANDOM_REPORT.md','\n'.join(lines))
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for (name,rr),color in zip([('Dense-SAM',sam),('VPR-L2 Magnitude',mag),('VPR-L2 Guided',grec)],['#2463A0','#777777','#D27620']):
        axes[0].plot([r['epoch'] for r in rr],[r['validation']['accuracy']*100 for r in rr],label=name,color=color,marker='o',ms=2)
        axes[1].plot([r['epoch'] for r in rr],[r['validation']['loss'] for r in rr],color=color)
        axes[2].plot([r['epoch'] for r in rr],[r['train_loss'] for r in rr],color=color)
    for ax,title in zip(axes,['Validation accuracy (%)','Validation loss','Training loss']):ax.set(xlabel='Recovery epoch',title=title);ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.suptitle('Seed 0 recovery | post-hoc curves; Final remains primary');fig.tight_layout();fig.savefig(out/'recovery_curves.png',dpi=180);plt.close(fig)
    shutil.copy2(out/'recovery_curves.png',root/'results/vpr_l2_seed0/recovery_curves.png')
    md(out/'RESEARCH_STATUS.md', '\n'.join(['# Research status | 2026-10-24','',
        'V1 SAM/VPR seeds 0/1/2 的六条 200 epoch 轨迹及六条 20 epoch 恢复完成。VPR-L2 seed 0、三 Mask 无训练评价和两条 Magnitude/Guided 恢复完成。',
        '同预算 seed 0 Best Validation Accuracy：Dense-SAM+Magnitude 95.90%（20）；VPR-L2+Magnitude 96.00%（8）；VPR-L2+Guided 96.08%（15）。',
        'Final 分别为 95.90%、95.84%、95.88%；不能用探索性 Best 改写 Final 主结果。Guided 的验证 Loss 不优于 SAM。尚无多 seed VPR-L2/Guided 或独立 Test 泛化证据。',
        '本轮完成 32 coarse + 32 fine matched-random Mask 无训练评价。Block-wise 仅独立原型、CPU 测试与数学设计，没有正式训练结果。']))
    md(root/'results/v1_main200/summary.md','# V1 three-seed archive\n\nSix 200-epoch dense runs and six 20-epoch fixed-mask recoveries completed. See per_seed_metrics.csv and provenance.json. Final remains primary; no test selection.')
    md(root/'results/vpr_l2_seed0/summary.md','# VPR-L2 seed 0 exploration\n\nOne dense trajectory, three-mask evaluation, Magnitude and Guided recovery completed. Best accuracy 96.08% is post-hoc validation selection, not independent generalization evidence. See research_delivery for mechanisms.')
    md(out/'BN_MECHANISM_AUDIT.md', '\n'.join(['# BN mechanism audit','',
        'Frozen step.py：center forward 为 model.train，更新 running stats；第二个对抗 forward 在 freeze_bn 中将 track_running_stats=False，使用当前 batch 统计但不更新 running buffers。退出恢复标志；原文件未改。',
        'Raw Hard 仅权重乘固定 Mask，保留所有 BN buffers。BN-only 是独立模型副本，reset_running_stats、momentum=None、仅 BN train、其他层 eval、no_grad、全部参数 requires_grad=False；校准前后参数 tensor_hash 相同。',
        '三方法使用相同 4096 原始 train IDs、无增强、32×128、验证 5000/batch256。Raw Recovery 的父 checkpoint 为未经校准 pruned.pt；SGD 新优化器、固定 Mask，20 epoch/7040 steps；数据 manifest 和 order_hash 相同。',
        'VPR-L2 中 BN affine 参与全空间几何；Mask 仅 Conv/Linear weight。geometry_allocation.csv 给出各参数 log-volume 与 RMS，用于审查跨空间预算迁移。预算偏移不能仅由参数数量推断为真实性能原因。',
        'Guided−Magnitude Raw Accuracy +0.62pp，BN-only 后 −0.10pp；Raw loss 改善 0.01469609，BN 后 loss 反而增加 0.00026513。BN 校准显著减弱 Raw 收益；不能据此断言 BN 是唯一机制。',
        'Dense-SAM BN-only 本轮补充评价：Accuracy '+str(bn_sam['bn']['accuracy'])+'，Loss '+str(bn_sam['bn']['loss'])+'；它是新非训练诊断，未追溯替换 V1 正式 Raw/Final。',
        '未发现需要更改冻结历史数据的 BN 规则缺陷。若未来比较其他 BN 算法/样本量，必须预登记并保留本口径；本轮未搜索 BN 样本量。']))
    md(out/'PROJECT_STATE_AUDIT.md', '\n'.join(['# Project state audit','',
        '## 已核验来源','',*[f"- {n}: `{r['head']}`；工作树 clean={r['clean']}；与远程分支 HEAD 核对一致。" for n,r in heads.items()],
        '', 'V1 服务器冻结 d4c58d6，V2 f742733，Guided recovery 1345d24；服务器工作树核对干净。V1 本地 audit-fixes 的 40e0c1b 是随后文档提交，不替代训练 producer。',
        '历史本地完整性报告包含 V1 六条主训练和恢复的 lineage/hash；VPR-L2/三 Mask/Guided recovery 均有 schema2 审计与 checkpoint/geometry/mask 摘要。Dense-SAM seed0 逐 epoch 原日志来自 main200_delivery 原归档，而非其他旧 Dense-SAM 项目。',
        '', '## 存储与发布边界','',
        '- GitHub 既有分支已提交源码与配置；新研究结果不能在推送核验之前声称已发布。新归档工作分支和 Draft PR 见最终提交记录。',
        '- 本地仅有小型日志、JSON/CSV、图表、报告、归档包；不会把本地同步等同 GitHub 发布。',
        '- 大 checkpoint、原 CIFAR 数据、完整 FP64 geometry、完整 Mask 张量留在私有服务器，未上传 GitHub。Checkpoint 获取需用户提供授权存储副本，并按 manifest SHA256 验证；仓库不包含凭据或公开下载地址。',
        '- 通过哈希确认：V1 seed0 日志归档/导出清单/服务器相同；V2 与 Guided epoch 日志、best/final checkpoint 已核验；本轮机制评价严格重放旧指标并对历史文件做前后哈希校验。',
        '- 已有历史产物不重命名、不覆盖、不追溯修改；未发现用户未提交修改。',
        '- 未独立验证服务器之外的大 checkpoint 第二份异地备份；不将本地小证据包视作大权重备份。',
        '- 未授权的新 200 epoch、额外训练 seed、Gamma_B 搜索均未执行。']))
    if (out/'CURRENT_STATE_CPU_AUDIT.json').exists():
        a=load(out/'CURRENT_STATE_CPU_AUDIT.json');assert a['status']=='PASS' and len(a['checkpoints'])==18
        with (out/'PROJECT_STATE_AUDIT.md').open('a',encoding='utf-8') as f:
            f.write('\n本轮独立 CPU 重新读取 V1 全部 18 个 dense/hard/recovery final checkpoint：schema2/来源/模型有限值/保存轮次通过，SHA256 与原正式清单一致且前后不变。详见 CURRENT_STATE_CPU_AUDIT.json。\n')
    with (out/'MATCHED_RANDOM_REPORT.md').open('a',encoding='utf-8') as f:
        f.write('\n## 独立重建与匹配质量\n\n')
        for label,folder in [('coarse','mechanism'),('fine','mechanism_fine')]:
            aa=load(out/folder/'CONTROL_INDEPENDENT_AUDIT.json');assert aa['status']=='PASS' and len(aa['controls'])==32
            freeze=load(out/folder/'RANDOM_MASK_FREEZE.json');over=[x['overlapping_guided_blocks'] for x in freeze['masks']]
            f.write(f"{label}：32 个随机 Mask 全部在 CPU 上按已冻结 construction 独立重建，tensor SHA256、合法2:4、layer×bin×rank-pattern 计数均精确一致。Guided gap 分布 {json.dumps(aa['guided_gap'])}；随机 pooled gap 分布 {json.dumps(aa['random_gap'])}。与原 Guided 的改动块重合数量 min/mean/max 为 {min(over)}/{np.mean(over):.3f}/{max(over)}（每组2466块）。\n\n")
        f.write('Fine 随机 Raw Accuracy 范围 84.82%–85.64%，Guided 85.66% 仅比随机最大值高1张验证样本；Raw Loss 0.43639640 低于随机最小0.43988022。两组结果一致支持固定模型上有限的 Raw Mask 选择证据，但不能泛化为训练方法优势。Fine BN 后随机均值93.249375%，Guided93.18%；31/32对照不低于Guided，其中1组并列。Guided BN Loss 比25/32对照更高。\n')
        f.write(f'\n变化分布：2466 块中2453块为第2名→第3名替换，10块为第1名→第3名，3块为第2名→第4名。最大单层360块（layer3.0.conv2），不是只集中一个层。变化坐标4932/11157504={100*nflip/total:.9f}%。\n')
    with (out/'BN_MECHANISM_AUDIT.md').open('a',encoding='utf-8') as f:
        f.write('\n几何分配：Mask-eligible 11157504 坐标的 log-volume +3342.6406452187、s RMS 0.00756526；非剪枝16458坐标总量 −3342.6406452187、s RMS 0.26969966，互相抵消保持全局体积。这证明有显著空间间几何迁移，但不证明该迁移造成剪枝损伤或 BN 效应。\n')
    md(out/'NEXT_EXPERIMENT_DECISION.md','''# 决策 C：证据仍不足，先补针对性机制检查

Guided 在两套全部随机对照上 Raw Hard Accuracy/Loss 更好，构成固定模型上积极的 Mask 选择证据；它不是“完全没有信息”。Fine 的 Accuracy 超过随机最大仅1张样本，且 BN 后优势消失/反转，不能据此证明真实删除风险预测信息在标准化 BN 后仍成立。

因此不选择 A 来授权立即进入正式 Block-wise 训练，也没有证据支持 B 的“先改变冻结公式”。独立 Block-wise CPU 原型已实现并通过测试，作为未来设计储备；本次没有训练它。

建议下轮先预注册条件删除损伤预测检验：固定 gap/rank/权重尺度/层，评价局部删除影响与 s 的增量解释；明确 BN 指标为 Raw 与标准化后双端点；再决定是否申请新的稠密训练 seed 和 Block-wise Gamma_B 协议。任何新训练均等待用户明确授权，不自动执行。

已完成 Phase0来源核验、Phase1小型证据归档、Phase2基线/BN审计、Phase3共64组无训练对照、Phase4数学与CPU原型。未执行新200epoch、Gamma搜索、额外训练seed或Test评价。最终发布分支/commit与Draft PR见交付消息；所有历史分支和产物保持不变。
''')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);args=p.parse_args();main(args.workspace)
