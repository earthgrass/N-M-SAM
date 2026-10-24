"""Exactly one imported Geometry-guided fixed-mask SGD recovery."""
import copy,csv,gc,json,math,os,shutil,subprocess,time,traceback
from pathlib import Path
import torch
from vpr_sam.config import recovery_config
from vpr_sam.l2_protocol import differences,full_order_hash,GPU_UUID
from vpr_sam.guided_recovery_adapter import AuditedRecoveryStepper,arithmetic_aulc
from vpr_sam.repro import (atomic_json,atomic_save,file_hash,tensor_hash,source_hashes,
    seed_all,seed_for,environment)
from vpr_sam.provenance import seal_artifact
from vpr_sam.models import build_model
from vpr_sam.data import build_data,loader
from vpr_sam.pruning import apply_masks,validate_masks
from vpr_sam.evaluation import evaluate
from vpr_sam.training import engine

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/vpr_l2_guided_recovery_seed0'
CONFIG=ROOT/'configs/approved_guided_recovery/seed0.json'


def original_files(root):
    paths=list((root/'runs/vpr_l2_seed0_v2').rglob('*'))
    paths+=list((root/'artifacts/vpr_l2').rglob('*'))
    paths+=list((root/'artifacts/geometry_guided_mask_seed0').rglob('*'))
    paths+=list(root.glob('VPR_L2_*.md'))
    return {str(p.relative_to(root)):file_hash(p) for p in paths if p.is_file()}


def writecsv(path,rows):
    with path.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    if (OUT/'status.json').exists():raise RuntimeError('Refuse duplicate/restart of authorized recovery')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise RuntimeError('New snapshot must be clean')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    sources=source_hashes(ROOT)
    receipt=json.loads((OUT/'input_bundle.receipt.json').read_text())
    inputs=OUT/'input_bundle.pt'
    if file_hash(inputs)!=receipt['input_sha256']:raise RuntimeError('Input transport identity mismatch')
    bundle=torch.load(inputs,map_location='cpu',weights_only=False)
    if bundle['schema']!='verified-guided-recovery-inputs-v1':raise RuntimeError('Unverified imported inputs')
    masks=bundle['masks']
    if tensor_hash(masks)!=bundle['guided_mask_sha256']:raise RuntimeError('Frozen guided mask identity mismatch')
    original_root=Path(bundle['original_project'])
    before=original_files(original_root)
    if before!=bundle['original_files_before']:raise RuntimeError('Original artifacts changed since verified export')
    if source_hashes(original_root)!=bundle['original_sources']:raise RuntimeError('Original producer source changed')
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise RuntimeError('GPU occupied; no concurrent GPU task permitted')
    if shutil.disk_usage(ROOT).free<5*1024**3:raise RuntimeError('Insufficient disk reserve')
    c=json.loads(CONFIG.read_text());rc=recovery_config(c)
    diff=differences(bundle['magnitude_config'],rc)
    if [r['field'] for r in diff]!=['output','pruning.score']:raise RuntimeError('Extra recovery variables changed: '+str(diff))
    if rc['pruning']['score']!='geometry_guided' or rc['optimizer']['method']!='sgd' or rc['training']['epochs']!=20:
        raise RuntimeError('Unapproved recovery protocol')
    expected_order=bundle['magnitude_order_hash']
    if full_order_hash(bundle['manifest'],20)!=expected_order:raise RuntimeError('Recovery data order mismatch')
    torch.set_num_threads(rc['threads']);seed_all(seed_for(0,'initialization'),True)
    env=environment(c)
    if env!=bundle['original_environment'] or env['execution']['gpu_uuid'].removeprefix('GPU-')!=GPU_UUID:
        raise RuntimeError('Approved RTX5090 runtime mismatch')
    import vpr_sam
    if not Path(vpr_sam.__file__).resolve().is_relative_to(ROOT/'src'):raise RuntimeError('Wrong project module path')
    datasets,manifest=build_data(c,ROOT)
    if manifest!=bundle['manifest']:raise RuntimeError('Original data/split mismatch')
    model=build_model(c['model']);model.load_state_dict(bundle['model']);apply_masks(model,masks)
    validate_masks(model,masks,c['pruning'])
    for name,value in model.state_dict().items():
        if name.endswith(('running_mean','running_var','num_batches_tracked')) and not torch.equal(value,bundle['model'][name]):
            raise RuntimeError('Reconstructed Raw Hard used recalibrated BN')
    if tensor_hash(dict(model.named_parameters()))!=bundle['guided_raw_parameter_sha256']:
        raise RuntimeError('Reconstructed weights differ from audited guided Raw Hard')
    device=torch.device(c['device']);model.to(device)
    raw=evaluate(model,loader(datasets['validation'],c['evaluation']['batch_size'],0,workers=c['data']['workers']),device)
    if raw!=bundle['guided_raw']:raise RuntimeError('Reconstructed Raw Hard validation replay mismatch')
    origin={k:bundle[k] for k in ('producer_commit','dense_checkpoint_sha256','guided_masks_file_sha256',
        'guided_mask_sha256','diagnostic_audit_sha256','diagnostic_commit','geometry_sha256','magnitude_recovery_sha256')}
    origin.update(adapter_commit=commit,verified_import_sha256=file_hash(inputs),config_diff=diff)
    parent=OUT/'imported_guided_raw/pruned.pt'
    payload=dict(schema=2,stage='pruned',config=c,model={k:v.cpu().clone() for k,v in model.state_dict().items()},
        masks=masks,manifest=manifest,initialization_hash=bundle['initialization_hash'],
        report=dict(protocol=c['pruning'],hard_validation=raw,bn_calibration='none',mask_selection='imported frozen tensor; no score recomputation'),
        source_hashes=sources,environment=env,parent=dict(path='external original final.pt',sha256=bundle['dense_checkpoint_sha256']),
        import_provenance=origin)
    atomic_save(parent,seal_artifact(payload))
    initial_model_hash=tensor_hash(model.state_dict())
    # Engineering-only one-real-batch check on a disposable model. Formal run
    # below reconstructs the original Raw Hard and creates another fresh SGD.
    probe_model=copy.deepcopy(model);probe=AuditedRecoveryStepper(probe_model,rc,masks)
    x,y,_=next(iter(loader(datasets['train'],rc['training']['batch_size'],0,0,workers=0,shuffle=True)))
    probe_info=probe.step(x.to(device),y.to(device))
    if tensor_hash(model.state_dict())!=initial_model_hash:raise RuntimeError('Precheck changed formal starting model')
    atomic_json(OUT/'real_batch_precheck.json',dict(status='PASS',formal_warm_start=False,discarded=True,info=probe_info))
    del probe,probe_model,model,payload;gc.collect();torch.cuda.empty_cache()
    atomic_json(OUT/'protocol_audit.json',dict(status='PASS',config=c,recovery_config=rc,config_sha256=file_hash(CONFIG),
        source_hashes=sources,environment=env,import_provenance=origin,
        raw_epoch0=raw,mask_sha256=tensor_hash(masks),expected_order_hash=expected_order,
        geometry_updated=False,mask_recomputed=False,optimizer='fresh ordinary SGD',BN='original SGD training rule'))
    (OUT/'RECOVERY_PROTOCOL_AUDIT.md').write_text('# Guided recovery protocol audit\n\nPASS.\n\n'
        'Original checkpoints were verified/exported in their ORIGINAL f742733 runtime. This project has a NEW source snapshot and honest derived schema2 parent. '
        'The only base-module changes are accepting geometry_guided metadata and prohibiting make_masks from silently generating Magnitude. '
        'geometry.py, step.py and engine.py are unchanged. Frozen mask tensors are imported, never recomputed.\n\n'
        'Only recovery protocol differences from actual Magnitude recovery: output identity and pruning.score metadata. '
        'SGD/LR/momentum/weight-decay/nesterov/cosine/data/augmentation/seed/BN rules/budget are identical. '
        'Ordinary SGD updates all trainable parameters including BN affine; masked weights, gradients and momentum remain zero. '
        'No FP64 geometry update or perturbation occurs.\n\n'
        'One real training batch was checked on a disposable model; this is not a formal warm start. '
        'Epoch wall times run between last-checkpoint publication boundaries; first includes startup. '
        'Step time sums are separately recorded.\n\n'
        'AULC for THIS comparison is the newly specified arithmetic mean of actual Accuracy epochs0..20, not the earlier trapezoidal area. '
        'Both historical Magnitude and new Guided are recomputed with the same mean21 definition.\n\n'
        +json.dumps(origin,indent=2)+'\n',encoding='utf8')
    atomic_json(OUT/'status.json',dict(status='RUNNING',epochs=20,only_method='guided fixed-mask SGD',time=time.time()))
    engine.Stepper=AuditedRecoveryStepper
    saved=engine.atomic_save;last_boundary=time.perf_counter();epoch_logs=[]
    initial_bn=[int(v) for k,v in bundle['model'].items() if k.endswith('num_batches_tracked')]
    epoch_logs.append(dict(epoch=0,train_loss=None,validation_accuracy=raw['accuracy'],validation_loss=raw['loss'],
        learning_rate=rc['optimizer']['lr'],mask_violations=0,regrown_parameters=0,
        masked_gradient_violations=0,masked_momentum_violations=0,bn_batches_min=min(initial_bn),bn_batches_max=max(initial_bn),
        bn_buffers_sha256=tensor_hash({k:v for k,v in bundle['model'].items() if 'running_' in k or k.endswith('num_batches_tracked')}),
        epoch_wall_seconds=0.,epoch_step_seconds=0.))
    writecsv(OUT/'RECOVERY_TRAINING_LOG.csv',epoch_logs)
    def audited_save(path,state):
        nonlocal last_boundary
        state['guided_recovery_import_provenance']=origin
        saved(path,state)
        if path.name!='last.pt':return
        now=time.perf_counter();row=state['epoch_rows'][-1];e=row['epoch']
        steps=[r for r in state['step_rows'] if r['epoch']==e-1]
        bn=[int(v) for k,v in state['model'].items() if k.endswith('num_batches_tracked')]
        bnstate={k:v for k,v in state['model'].items() if 'running_' in k or k.endswith('num_batches_tracked')}
        epoch_logs.append(dict(epoch=e,train_loss=row['train_loss'],validation_accuracy=row['validation']['accuracy'],validation_loss=row['validation']['loss'],
            learning_rate=row['lr'],mask_violations=sum(r['mask_violations'] for r in steps),
            regrown_parameters=sum(r['regrown_parameters'] for r in steps),
            masked_gradient_violations=sum(r['masked_gradient_violations'] for r in steps),
            masked_momentum_violations=sum(r['masked_momentum_violations'] for r in steps),
            bn_batches_min=min(bn),bn_batches_max=max(bn),bn_buffers_sha256=tensor_hash(bnstate),
            epoch_wall_seconds=now-last_boundary,epoch_step_seconds=sum(r['step_seconds'] for r in steps)))
        last_boundary=now;writecsv(OUT/'RECOVERY_TRAINING_LOG.csv',epoch_logs)
        if e in (1,5,10,15,20):shutil.copyfile(path,path.parent/f'epoch_{e:03d}.pt')
        if shutil.disk_usage(ROOT).free<4*1024**3:raise RuntimeError('Disk reserve below4GiB')
        if source_hashes(ROOT)!=sources:raise RuntimeError('Frozen training source changed')
    engine.atomic_save=audited_save
    result=engine.train(c,ROOT,formal_allowed=True,recover_from=parent)
    final_path=parent.parent/'recovery/final.pt'
    final,_=engine.read_checkpoint(final_path,ROOT)
    if final['stage']!='recovery' or final['epoch']!=20 or final['stepper']['steps']!=7040 or final['next_batch']!=0:
        raise RuntimeError('Recovery final budget mismatch')
    if final['order_hash']!=expected_order or final['manifest']!=manifest:raise RuntimeError('Recovery paired identity mismatch')
    if final['scheduler']['T_max']!=20 or final['scheduler']['last_epoch']!=20:
        raise RuntimeError('Recovery scheduler budget mismatch')
    if final['stepper']['geometry']['logs'] or final['stepper']['geometry']['updates']:
        raise RuntimeError('Recovery unexpectedly updated geometry')
    if tensor_hash(final['masks'])!=bundle['guided_mask_sha256'] or final['parent']['sha256']!=file_hash(parent):
        raise RuntimeError('Frozen mask/parent chain mismatch')
    checked=build_model(c['model']);checked.load_state_dict(final['model']);validate_masks(checked,final['masks'],c['pruning'])
    if not all(bool(torch.isfinite(v).all()) for v in final['model'].values()):raise RuntimeError('Nonfinite final weights')
    evaluated=engine.evaluate_checkpoint(final_path,ROOT,'validation',False)
    if evaluated['metrics']!=final['epoch_rows'][-1]['validation']:raise RuntimeError('Final checkpoint evaluation mismatch')
    if original_files(original_root)!=before or source_hashes(original_root)!=bundle['original_sources']:
        raise RuntimeError('Original artifacts changed')
    if source_hashes(ROOT)!=sources:raise RuntimeError('New source snapshot changed')
    comparisons=[]
    for method,epochs,raw_metric,bn_metric in [('magnitude',bundle['magnitude_epochs'],bundle['magnitude_raw'],bundle['magnitude_bn']),
            ('geometry_guided',final['epoch_rows'],raw,bundle['guided_bn'])]:
        row=dict(method=method,raw_accuracy=raw_metric['accuracy'],raw_loss=raw_metric['loss'],
            bn_accuracy=bn_metric['accuracy'],bn_loss=bn_metric['loss'],
            final_accuracy=epochs[-1]['validation']['accuracy'],final_loss=epochs[-1]['validation']['loss'],
            accuracy_aulc_mean21=arithmetic_aulc(raw_metric['accuracy'],epochs))
        for e in (1,5,10,15,20):
            row[f'epoch{e}_accuracy']=epochs[e-1]['validation']['accuracy'];row[f'epoch{e}_loss']=epochs[e-1]['validation']['loss']
        comparisons.append(row)
    writecsv(OUT/'RECOVERY_COMPARISON.csv',comparisons)
    milestones={str(e):file_hash(parent.parent/f'recovery/epoch_{e:03d}.pt') for e in (1,5,10,15,20)}
    audit=dict(status='PASS',commit=commit,config_sha256=file_hash(CONFIG),source_hashes=sources,
        import_provenance=origin,mask_sha256=bundle['guided_mask_sha256'],parent_sha256=file_hash(parent),
        final_checkpoint_sha256=file_hash(final_path),milestone_checkpoint_sha256=milestones,
        epochs=20,steps=7040,order_hash=expected_order,initialization_hash=final['initialization_hash'],
        dataset_fingerprint=manifest['fingerprint'],original_files_unchanged=len(before),
        mask_violations=0,gradient_violations=0,momentum_violations=0,regrowth=0,
        BN_rule_unchanged=True,optimizer_fresh=True,geometry_updates=0,test_used=False,
        elapsed_seconds=result['elapsed_seconds'],peak_cuda_bytes=result['peak_cuda_bytes'],comparison=comparisons)
    atomic_json(OUT/'INTEGRITY_AUDIT.json',audit)
    make_report(bundle,final['epoch_rows'],comparisons,audit)
    atomic_json(OUT/'status.json',dict(status='COMPLETE',epochs=20,steps=7040,further_gpu_tasks=0,time=time.time()))
    atomic_json(OUT/'DELIVERY_SHA256.json',{str(p.relative_to(OUT)):file_hash(p) for p in OUT.rglob('*') if p.is_file() and p.name!='DELIVERY_SHA256.json'})
    print('GUIDED RECOVERY COMPLETE; no further GPU task',flush=True)


def make_report(bundle,epochs,rows,audit):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for label,values,raw in [('Magnitude',bundle['magnitude_epochs'],bundle['magnitude_raw']),('Geometry-guided',epochs,bundle['guided_raw'])]:
        for ax,key in zip(axes,['accuracy','loss']):
            ax.plot(range(21),[raw[key]]+[r['validation'][key] for r in values],label=label)
            ax.set(xlabel='Recovery epoch (0 = Raw Hard)',ylabel='Validation '+key);ax.grid(alpha=.2);ax.legend()
    fig.tight_layout();fig.savefig(OUT/'RECOVERY_CURVES.png',dpi=180);fig.savefig(OUT/'RECOVERY_CURVES.pdf');plt.close(fig)
    a,b=rows;da=b['final_accuracy']-a['final_accuracy'];dl=b['final_loss']-a['final_loss'];du=b['accuracy_aulc_mean21']-a['accuracy_aulc_mean21']
    text='# Geometry-guided 2:4 Mask：固定掩码恢复结果\n\n'
    text+='同一 VPR-L2 seed0 final、同一验证划分和恢复预算，唯一研究变量为冻结 Mask。未重训 Dense、未重选 Mask、未用 Test。\n\n'
    text+='| 指标 | Magnitude | Geometry-guided |\n|---|---:|---:|\n'
    for key in ['raw_accuracy','raw_loss','bn_accuracy','bn_loss','epoch1_accuracy','epoch5_accuracy','epoch10_accuracy','epoch15_accuracy','epoch20_accuracy','final_loss','accuracy_aulc_mean21']:
        text+=f'| {key} | {a[key]:.9f} | {b[key]:.9f} |\n'
    text+='\nAULC 严格采用本任务新定义：(Acc0+...+Acc20)/21；Epoch0 来自未经 BN 重校准的实际 Raw Hard。历史 Magnitude 同口径重算，未沿用先前梯形积分值。\n\n'
    text+=f'最终差值 Guided−Magnitude：Accuracy {da*100:+.4f} pp；Loss {dl:+.9f}；Mean21 AULC {du*100:+.4f} pp。\n\n'
    text+='## 七项科研问题\n\n'
    text+=f"1. 20epoch 绝对准确率：{'提高' if da>0 else ('降低' if da<0 else '相同')}，差值 {da*100:+.4f} pp。\n\n"
    text+=f"2. 最终 Validation Loss：{'降低' if dl<0 else ('升高' if dl>0 else '相同')}，差值 {dl:+.9f}。\n\n"
    text+=f"3. 恢复 AULC：{'提高' if du>0 else ('降低' if du<0 else '相同')}；早期速度还应结合 Epoch1/5/10/15 的逐点曲线，不能仅据最终值判断。\n\n"
    raw_delta=(bundle['guided_raw']['accuracy']-bundle['magnitude_raw']['accuracy'])*100
    bn_delta=(bundle['guided_bn']['accuracy']-bundle['magnitude_bn']['accuracy'])*100
    text+=f"4. Raw Hard 的 {raw_delta:+.2f}pp 在恢复后变为 {da*100:+.4f}pp，{'仍有正向差异，但幅度需按实际结果解释' if da>0 else '原正向优势未保留'}。\n\n"
    text+=f"5. BN-only Hard 原差值 {bn_delta:+.2f}pp 与恢复后 {'方向一致' if da<0 else ('恢复后相同准确率，并非同一结论' if da==0 else '方向不同')}；BN-only 不更新权重，不能替代恢复训练。\n\n"
    if da>0 and dl<0 and du>0:
        recommendation='三个指标方向一致，值得将其作为预注册跨 seed 复核的候选；是否投入扩展需同时权衡此处的实际差值大小，尚不能认定具有稳定优势。'
    else:
        recommendation='指标未同时支持一致改善，当前不足以据此认定值得直接扩成正式多 seed 主实验；可与用户讨论是否开展机制复核，不能自动追加。'
    text+='6. '+recommendation+'单 seed 不提供统计显著性，不据代理代价宣称成功。\n\n'
    early=b['epoch1_accuracy']-a['epoch1_accuracy']
    text+=f'7. Epoch1 差值 {early*100:+.4f}pp，Epoch20 差值 {da*100:+.4f}pp。若末期更接近，较长恢复预算导致趋同是一种解释；本实验没有改变预算做因果验证，不能断言。\n\n'
    text+='![Recovery curves](RECOVERY_CURVES.png)\n\n'
    text+='完整配置、来源链、冻结 Mask/各 milestone/final SHA256、BN 与防重生审计见 INTEGRITY_AUDIT.json、protocol_audit.json；每 epoch 原始指标见 RECOVERY_TRAINING_LOG.csv。\n\n'
    text+='恢复 producer commit: `'+audit['commit']+'`\n\nFinal checkpoint SHA256: `'+audit['final_checkpoint_sha256']+'`\n\n'
    text+='已完成唯一20epoch/7040steps轨迹并停止后续GPU工作。原有产物逐文件哈希未改变。\n'
    (OUT/'GUIDED_RECOVERY_FINAL_REPORT.md').write_text(text,encoding='utf8')


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        OUT.mkdir(parents=True,exist_ok=True)
        atomic_json(OUT/'failure.json',dict(status='FAILED',error=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),automatic_retry=False))
        raise
