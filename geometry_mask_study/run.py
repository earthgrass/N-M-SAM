"""One frozen-checkpoint, three-mask evaluation. NO optimizer or training call."""
import argparse,csv,gc,hashlib,itertools,json,math,subprocess,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch import nn
from masks import three_masks,mask_difference,deletion_metrics
from vpr_sam.training.engine import read_checkpoint
from vpr_sam.pruning import make_masks,apply_masks,validate_masks
from vpr_sam.models import build_model
from vpr_sam.optim.geometry import Geometry
from vpr_sam.data import build_data,Images,loader
from vpr_sam.evaluation import evaluate
from vpr_sam.repro import atomic_json,file_hash,source_hashes,tensor_hash,seed_all,seed_for,environment
from vpr_sam.l2_protocol import OUTPUT,GPU_UUID

HERE=Path(__file__).resolve().parent
PRODUCER='f7427339de6d6589f53b0191068448cf642901b2'


def preserved_files(root):
    files=list((root/OUTPUT).rglob('*'))+list((root/'artifacts/vpr_l2').rglob('*'))+list(root.glob('VPR_L2_*.md'))
    return {str(p.relative_to(root)):file_hash(p) for p in files if p.is_file()}


def dump_csv(path,rows):
    if not rows:return
    with path.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def bn_only(model,train,validation,device):
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    initial=tensor_hash(dict(model.named_parameters()))
    modules=[m for m in model.modules() if isinstance(m,nn.modules.batchnorm._BatchNorm)]
    for m in modules:m.reset_running_stats();m.momentum=None;m.train()
    with torch.no_grad():
        for x,_,_ in loader(train,128,0,workers=0):model(x.to(device))
    if any(int(m.num_batches_tracked)!=32 for m in modules):raise RuntimeError('Incorrect BN calibration batches')
    if tensor_hash(dict(model.named_parameters()))!=initial:raise RuntimeError('BN calibration changed weights')
    model.eval()
    return evaluate(model,loader(validation,256,0,workers=0),device),dict(trainable_weights_unchanged=True,
        parameter_sha256=initial,bn_modules=len(modules),batches_per_bn=32,samples=4096,augmentation=False,
        algorithm='reset running stats, momentum=None, BN-only train, others eval',optimizer_steps=0)


def main(root,out):
    if out.exists():raise RuntimeError('Refuse overwrite/automatic rerun of diagnostic directory')
    out.mkdir(parents=True)
    atomic_json(out/'status.json',dict(status='RUNNING',scope='three masks, no training',time=time.time()))
    for name,sha in json.loads((HERE/'SOURCE_SHA256.json').read_text()).items():
        if file_hash(HERE/name)!=sha:raise RuntimeError('Diagnostic source changed')
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()!=PRODUCER:
        raise RuntimeError('Producer HEAD changed')
    if subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip():raise RuntimeError('Producer dirty')
    gates=json.loads((root/'artifacts/vpr_l2/control/gates.json').read_text())
    if source_hashes(root)!=gates['source_hashes']:raise RuntimeError('Producer source changed')
    for p in (root/'artifacts/vpr_l2/control/status.json',root/'artifacts/vpr_l2/recovery_extension/status.json'):
        if json.loads(p.read_text())['status']!='COMPLETE':raise RuntimeError('Earlier authorized queue has not completed')
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise RuntimeError('GPU occupied; no concurrent evaluation permitted')
    before=preserved_files(root);atomic_json(out/'original_files_before.json',before)
    checkpoint=root/OUTPUT/'final.pt';state,_=read_checkpoint(checkpoint,root)
    if state['epoch']!=200 or state['stepper']['steps']!=70400 or state['next_batch']!=0:raise RuntimeError('Not final200/70400')
    c=state['config'];protocol=c['pruning']
    if (protocol['n'],protocol['m'],protocol['scope'],protocol['tail'])!=(2,4,'exclude_first_last','dense'):
        raise RuntimeError('Unexpected approved pruning protocol')
    torch.set_num_threads(c['threads']);seed_all(seed_for(0,'evaluation'),True)
    if environment(c)!=gates['environment']:raise RuntimeError('Approved runtime mismatch')
    if environment(c)['execution']['gpu_uuid'].removeprefix('GPU-')!=GPU_UUID:raise RuntimeError('Wrong GPU')
    model=build_model(c['model']);model.load_state_dict(state['model'])
    original_hash=tensor_hash(model.state_dict())
    geometry=Geometry(model.named_parameters(),c['geometry'],'fixed_l2',0)
    geometry.load_state_dict(state['stepper']['geometry'])
    logs={name:s for (name,_),s in zip(geometry.mapping,geometry.logs)}
    if geometry.d!=11173962 or len(logs)!=len(list(model.parameters())):raise RuntimeError('Global geometry dimension mismatch')
    geometry_hash=tensor_hash(logs)
    reference_masks,selection=make_masks(model,protocol)
    original_hard,_=read_checkpoint(root/OUTPUT/'pruning/n2_m4_exclude_first_last_magnitude_dense_final/pruned.pt',root)
    if original_hard['parent']['sha256']!=file_hash(checkpoint):raise RuntimeError('Historical magnitude parent mismatch')
    params=dict(model.named_parameters());modules=dict(model.named_modules());mask_sets={k:{} for k in ('magnitude','radius_only','geometry_guided')}
    mapping=[];offset=0
    for (name,shape),s in zip(geometry.mapping,geometry.logs):
        mapping.append(dict(name=name,shape=list(shape),global_offset=offset,numel=s.numel(),mask_eligible=name in reference_masks));offset+=s.numel()
    for name in reference_masks:
        module=modules[name.rsplit('.',1)[0]]
        if not isinstance(module,(nn.Conv2d,nn.Linear)) or isinstance(module,nn.modules.batchnorm._BatchNorm):
            raise RuntimeError('BN affine/non-weight entered Mask')
        masks=three_masks(params[name],logs[name],2,4,'dense')
        if not torch.equal(masks['magnitude'],reference_masks[name]) or not torch.equal(masks['magnitude'],original_hard['masks'][name]):
            raise RuntimeError('Magnitude tie/layout mismatch')
        for method,mask in masks.items():mask_sets[method][name]=mask
    atomic_json(out/'geometry_mapping.json',dict(d=geometry.d,parameters=mapping,geometry_sha256=geometry_hash,
        coordinate_layout='native contiguous parameter coordinates; per-output-channel flattened in/kh/kw for mask blocks'))
    torch.save(mask_sets,out/'masks.pt')
    layer_deletions=[];pair_rows=[];global_cost={};global_diffs=[]
    for method,masks in mask_sets.items():
        totals=dict(deleted_weights=0,displacement_squared_l2=0.,geometric_deletion_cost=0.)
        for name,mask in masks.items():
            metrics=deletion_metrics(params[name],logs[name],mask)
            layer_deletions.append(dict(method=method,parameter=name,**metrics))
            for k in totals:totals[k]+=metrics[k]
        totals.update(displacement_l2=math.sqrt(totals['displacement_squared_l2']),geometric_deletion_norm=math.sqrt(totals['geometric_deletion_cost']))
        global_cost[method]=totals
    for left,right in itertools.combinations(mask_sets,2):
        totals={k:0 for k in ('flipped_coordinates','total_target_weights','complete_block_weights','full_blocks','different_blocks')}
        for name in reference_masks:
            metrics=mask_difference(mask_sets[left][name],mask_sets[right][name])
            pair_rows.append(dict(left=left,right=right,parameter=name,**metrics))
            for k in totals:totals[k]+=metrics[k]
        totals.update(left=left,right=right,mask_flip_rate=totals['flipped_coordinates']/totals['complete_block_weights'],
            mask_flip_rate_including_dense_tail=totals['flipped_coordinates']/totals['total_target_weights'],
            different_block_fraction=totals['different_blocks']/totals['full_blocks'])
        global_diffs.append(totals)
    dump_csv(out/'layer_deletion_metrics.csv',layer_deletions);dump_csv(out/'layer_mask_differences.csv',pair_rows)
    atomic_json(out/'global_mask_differences.json',global_diffs);atomic_json(out/'global_deletion_metrics.json',global_cost)
    datasets,manifest=build_data(c,root)
    if manifest!=state['manifest']:raise RuntimeError('Dataset split mismatch')
    ids=json.loads((root/'docs/vpr_l2_reference/fixed_sample_ids.json').read_text())['0']['bn_train']
    if ids!=manifest['train_indices'][:4096]:raise RuntimeError('Frozen BN calibration samples mismatch')
    val=datasets['validation'];train=Images(val.images,val.labels,ids,0,augment=False,cutout=0)
    device=torch.device(c['device'])
    dense_model=build_model(c['model']).to(device);dense_model.load_state_dict(state['model'])
    dense=evaluate(dense_model,loader(val,256,0,workers=0),device)
    baseline=json.loads((root/'artifacts/vpr_l2/evaluation.json').read_text())
    if dense!=baseline['dense']['metrics']:raise RuntimeError('Dense final validation mismatch')
    del dense_model;gc.collect();torch.cuda.empty_cache()
    rows=[]
    for method,masks in mask_sets.items():
        sparse=build_model(c['model']).to(device);sparse.load_state_dict(state['model'])
        apply_masks(sparse,masks);validate_masks(sparse,masks,protocol)
        for name,value in sparse.state_dict().items():
            if name.endswith(('running_mean','running_var','num_batches_tracked')) and not torch.equal(value.cpu(),state['model'][name]):
                raise RuntimeError('Raw model BN changed')
        raw=evaluate(sparse,loader(val,256,0,workers=0),device)
        if method=='magnitude' and raw!=baseline['raw_hard']['hard_validation']:raise RuntimeError('Magnitude Raw Hard replay mismatch')
        calibrated,bn_evidence=bn_only(sparse,train,val,device)
        validate_masks(sparse,masks,protocol)
        if method=='magnitude' and calibrated!=baseline['bn_calibrated_hard']['metrics']:raise RuntimeError('Magnitude BN replay mismatch')
        row=dict(method=method,raw=raw,bn_calibrated=calibrated,
            raw_loss_damage=raw['loss']-dense['loss'],raw_accuracy_drop=dense['accuracy']-raw['accuracy'],
            bn_loss_damage=calibrated['loss']-dense['loss'],bn_accuracy_drop=dense['accuracy']-calibrated['accuracy'],
            mask_sha256=tensor_hash(masks),bn_evidence=bn_evidence,global_deletion=global_cost[method])
        rows.append(row);atomic_json(out/f'{method}_evaluation.json',row)
        del sparse;gc.collect();torch.cuda.empty_cache()
    if tensor_hash(model.state_dict())!=original_hash or tensor_hash(logs)!=geometry_hash:
        raise RuntimeError('Original weights/log-radius changed in memory')
    after=preserved_files(root)
    if after!=before or source_hashes(root)!=gates['source_hashes']:raise RuntimeError('Original checkpoint/Magnitude artifacts changed')
    atomic_json(out/'original_files_after.json',after)
    audit=dict(status='PASS',producer_commit=PRODUCER,
        diagnostic_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip(),
        final_checkpoint_sha256=file_hash(checkpoint),geometry_sha256=geometry_hash,masks_file_sha256=file_hash(out/'masks.pt'),
        dense=dense,methods=rows,global_mask_differences=global_diffs,
        log_radius_min=min(float(s.min()) for s in geometry.logs),
        log_radius_max=max(float(s.max()) for s in geometry.logs),
        global_log_volume=geometry.check_volume(geometry.logs),
        original_files_unchanged=len(before),geometry_unchanged=True,bn_affine_masked=False,
        test_used=False,optimizer_steps=0,recovery_runs_added=0,source_hashes=gates['source_hashes'],
        bn_train_ids_sha256=hashlib.sha256(np.asarray(ids,dtype=np.int64).tobytes()).hexdigest(),
        protocol=protocol,tie_rule='keep highest score, stable lower coordinate index first; B uses score=-s')
    atomic_json(out/'final_audit.json',audit)
    report(out,audit)
    atomic_json(out/'status.json',dict(status='COMPLETE',optimizer_steps=0,further_gpu_tasks=0,time=time.time()))
    atomic_json(out/'DELIVERY_SHA256.json',{str(p.relative_to(out)):file_hash(p) for p in out.rglob('*') if p.is_file() and p.name!='DELIVERY_SHA256.json'})
    print('THREE MASK EVALUATION COMPLETE; no training performed',flush=True)


def report(out,audit):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    text='# VPR-L2 seed0：三 Mask 无训练对照\n\n'
    text+='同一 final 权重和 FP64 log-radius；固定2:4、exclude_first_last、dense tail。未修改 s，未使用 Test，未执行任何优化器 step 或新增恢复。\n\n'
    text+='A Magnitude 删除最小 |w|；B Radius-only 删除最大 s（保留评分 -s 最大者）；C Geometry-guided 删除最小 |w|exp(-s)。'
    text+='所有保留评分均按降序 stable argsort，评分相同保留较低坐标，与原 Magnitude 一致。BN affine 不进入 Mask。\n\n'
    text+='几何删除代价定义为 sum_deleted (w_i exp(-s_i))²，另报告其平方根；删除位移为 ||w_deleted||₂。'
    text+='Mask Flip Rate 为完整2:4块内坐标的 Hamming 差异比例，另提供含 dense tail 的分母；不同块比例统计至少一个位置不同的完整块。全局比例由计数汇总，不对层比例直接平均。\n\n'
    text+='| Mask | Raw Accuracy | Raw Loss | Raw Acc Drop | Raw Loss Damage | BN Accuracy | BN Loss | BN Acc Drop | BN Loss Damage |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
    dense=audit['dense']
    for row in audit['methods']:
        text+='| '+row['method']+' | '+' | '.join(f'{v:.9f}' for v in (row['raw']['accuracy'],row['raw']['loss'],row['raw_accuracy_drop'],row['raw_loss_damage'],row['bn_calibrated']['accuracy'],row['bn_calibrated']['loss'],row['bn_accuracy_drop'],row['bn_loss_damage']))+' |\n'
    text+='\nDense final validation: '+json.dumps(dense)+'\n\n'
    text+='| Pair | Mask Flip Rate | Different Blocks |\n|---|---:|---:|\n'
    for p in audit['global_mask_differences']:text+=f"| {p['left']} / {p['right']} | {p['mask_flip_rate']:.9f} | {p['different_block_fraction']:.9f} |\n"
    text+='\n| Mask | Deletion L2 | Geometry deletion cost | Geometry deletion norm |\n|---|---:|---:|---:|\n'
    for row in audit['methods']:
        d=row['global_deletion'];text+=f"| {row['method']} | {d['displacement_l2']:.9f} | {d['geometric_deletion_cost']:.9f} | {d['geometric_deletion_norm']:.9f} |\n"
    text+='\n每层结果见 layer_mask_differences.csv 和 layer_deletion_metrics.csv。BN 只在独立副本上用同一4096训练样本、32×128无增强批次、reset running stats、momentum=None、仅BN train重估；不改变可训练权重。原始 Magnitude 的 Raw/BN 指标逐值重放一致。\n\n'
    text+='C 的评分按定义最小化此对角几何删除代价，不能据此宣称真实 Loss 或准确率更优。应分别比较实际 Raw/BN 损伤；仅一个 checkpoint、一个 seed，不作显著性或可恢复性结论。B/C 的固定 Mask 恢复未获本轮授权，未执行。\n\n'
    fig,ax=plt.subplots(figsize=(7,4));x=np.arange(3);rows=audit['methods']
    ax.bar(x-.18,[r['raw']['accuracy'] for r in rows],width=.36,label='Raw Hard')
    ax.bar(x+.18,[r['bn_calibrated']['accuracy'] for r in rows],width=.36,label='BN-only recalibrated')
    ax.axhline(dense['accuracy'],color='black',ls='--',label='Dense final')
    ax.set(xticks=x,xticklabels=[r['method'] for r in rows],ylabel='Validation accuracy');ax.legend();fig.tight_layout()
    for suffix in ('png','pdf'):fig.savefig(out/f'three_mask_accuracy.{suffix}',dpi=180)
    plt.close(fig)
    text+='![Three masks](three_mask_accuracy.png)\n\n'
    text+='Producer commit: `'+audit['producer_commit']+'`\n\nDiagnostic commit: `'+audit['diagnostic_commit']+'`\n\nFinal checkpoint SHA256: `'+audit['final_checkpoint_sha256']+'`\n\n'
    text+=f"完整性审计 PASS：{audit['original_files_unchanged']} 个原产物逐文件 SHA256 前后不变；权重/几何内存哈希不变；全部 Mask 严格满足完整块2:4与 dense tail。\n"
    (out/'THREE_MASK_NO_TRAINING_REPORT.md').write_text(text,encoding='utf8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    try:main(args.project.resolve(),args.output.resolve())
    except BaseException as exc:
        args.output.mkdir(parents=True,exist_ok=True)
        atomic_json(args.output/'failure.json',dict(status='FAILED',error=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),automatic_retry=False))
        raise
