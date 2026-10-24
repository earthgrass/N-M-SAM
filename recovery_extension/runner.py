"""One newly authorized recovery, outside the immutable dense source snapshot."""
import argparse, csv, hashlib, json, math, os, shutil, signal
try:
    import fcntl
except ImportError:
    fcntl=None  # reporting/math tests also run on Windows; controller is Linux-only
import subprocess, sys, time, traceback
from pathlib import Path
import torch
from vpr_sam.config import recovery_config
from vpr_sam.l2_protocol import OUTPUT, differences, full_order_hash
from vpr_sam.repro import atomic_json, file_hash, source_hashes, tensor_hash
from vpr_sam.training import engine
from vpr_sam.optim import Stepper

EXT=Path(__file__).resolve().parent
EXPECTED_DENSE_COMMIT='f7427339de6d6589f53b0191068448cf642901b2'


def curve_metrics(epochs, hard, dense):
    if [e['epoch'] for e in epochs]!=list(range(1,21)):
        raise ValueError('Recovery must contain exactly epochs 1..20')
    acc=[hard['accuracy']]+[e['validation']['accuracy'] for e in epochs]
    losses=[hard['loss']]+[e['validation']['loss'] for e in epochs]
    if not all(math.isfinite(x) for x in acc+losses):raise ValueError('Nonfinite recovery curve')
    final=epochs[-1]['validation']
    area=sum((a+b)/2 for a,b in zip(acc[:-1],acc[1:]))
    return dict(final=final,accuracy_aulc_0_20=area,accuracy_aulc_normalized=area/20,
        accuracy_aulc_1_20=sum((a+b)/2 for a,b in zip(acc[1:-1],acc[2:]))/19,
        loss_aulc_normalized=sum((a+b)/2 for a,b in zip(losses[:-1],losses[1:]))/20,
        hard_to_recovery_accuracy_gain=final['accuracy']-hard['accuracy'],
        hard_to_recovery_loss_reduction=hard['loss']-final['loss'],
        dense_to_recovery_accuracy_gap=dense['accuracy']-final['accuracy'],
        recovery_minus_dense_loss=final['loss']-dense['loss'])


def frozen(root):
    gates=json.loads((root/'artifacts/vpr_l2/control/gates.json').read_text())
    if source_hashes(root)!=gates['source_hashes']:raise RuntimeError('Dense source snapshot changed')
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()!=EXPECTED_DENSE_COMMIT:
        raise RuntimeError('Dense frozen HEAD changed')
    if subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip():
        raise RuntimeError('Dense worktree is dirty')
    for name,sha in json.loads((EXT/'reference_sha256.json').read_text()).items():
        if file_hash(EXT/name)!=sha:raise RuntimeError('Extension reference changed')
    return gates


def worker(root):
    gates=frozen(root);out=root/'artifacts/vpr_l2/recovery_extension'
    evaluation=json.loads((root/'artifacts/vpr_l2/evaluation.json').read_text())
    hard=root/OUTPUT/'pruning/n2_m4_exclude_first_last_magnitude_dense_final/pruned.pt'
    parent,_=engine.read_checkpoint(hard,root)
    if parent['stage']!='pruned' or file_hash(hard)!=evaluation['checkpoint_sha256']['hard']:
        raise RuntimeError('Not the original raw-hard artifact')
    dense,_=engine.read_checkpoint(root/OUTPUT/'final.pt',root)
    for name,value in dense['model'].items():
        if name.endswith(('running_mean','running_var','num_batches_tracked')) and not torch.equal(value,parent['model'][name]):
            raise RuntimeError('Raw Hard BN differs from dense original')
    old=json.loads((root/'docs/vpr_l2_reference/v1_vpr_seed0.json').read_text())
    rc=recovery_config(parent['config']);baseline=recovery_config(old)
    diff=differences(baseline,rc)
    if [r['field'] for r in diff]!=['output']:raise RuntimeError('Recovery protocol differs from V1: '+str(diff))
    if rc['optimizer']['method']!='sgd' or rc['training']['epochs']!=20:raise RuntimeError('Unapproved recovery method/budget')
    reference_gates=json.loads((root/'docs/vpr_l2_reference/v1_gates.json').read_text())
    expected_order=reference_gates['identities']['0']['recovery_order_hash']
    if full_order_hash(parent['manifest'],20)!=expected_order:raise RuntimeError('Recovery sample order mismatch')
    if (hard.parent/'recovery').exists():raise RuntimeError('Recovery already exists; no automatic restart or duplicate run')
    raw_model_hash=tensor_hash(parent['model']);raw_mask_hash=tensor_hash(parent['masks'])
    class FreshSGD(Stepper):
        def __init__(self,model,config,masks=None):
            super().__init__(model,config,masks)
            if self.method!='sgd' or self.steps!=0 or self.base.state:
                raise RuntimeError('Recovery optimizer must be fresh SGD')
            if tensor_hash(model.state_dict())!=raw_model_hash:
                raise RuntimeError('Recovery initialization differs from Raw Hard model/BN')
            atomic_json(out/'optimizer_initialization.json',dict(fresh_sgd=True,steps=0,momentum_entries=0,
                initial_model_sha256=raw_model_hash,raw_masks_sha256=raw_mask_hash,optimizer=config['optimizer']))
    engine.Stepper=FreshSGD
    pre=dict(status='PASS',producer_commit=EXPECTED_DENSE_COMMIT,
        extension_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=EXT,text=True).strip(),
        recovery_protocol=rc,protocol_diff_from_v1=diff,raw_hard_sha256=file_hash(hard),
        raw_model_sha256=raw_model_hash,raw_masks_sha256=raw_mask_hash,expected_order_hash=expected_order,
        bn_recalibrated_parent=False,checkpoint_selection='final',test_used=False)
    atomic_json(out/'precheck.json',pre)
    del dense
    result=engine.train(parent['config'],root,formal_allowed=True,recover_from=hard)
    final_path=hard.parent/'recovery/final.pt'
    final,_=engine.read_checkpoint(final_path,root)
    if final['stage']!='recovery' or final['epoch']!=20 or final['stepper']['steps']!=7040 or final['next_batch']!=0:
        raise RuntimeError('Recovery budget/stage mismatch')
    if final['order_hash']!=expected_order or final['manifest']!=parent['manifest']:
        raise RuntimeError('Recovery paired data identity mismatch')
    if final['initialization_hash']!=parent['initialization_hash'] or tensor_hash(final['masks'])!=raw_mask_hash:
        raise RuntimeError('Recovery parent or fixed Mask mismatch')
    if final['parent']['sha256']!=file_hash(hard) or final['parent']['sha256']!=pre['raw_hard_sha256']:
        raise RuntimeError('Recovery parent provenance mismatch')
    from vpr_sam.pruning import validate_masks
    from vpr_sam.models import build_model
    model=build_model(rc['model']);model.load_state_dict(final['model']);validate_masks(model,final['masks'],rc['pruning'])
    if not all(bool(torch.isfinite(v).all()) for v in final['model'].values()):raise RuntimeError('Nonfinite recovered model')
    evaluated=engine.evaluate_checkpoint(final_path,root,'validation',False)
    if evaluated['metrics']!=final['epoch_rows'][-1]['validation']:
        raise RuntimeError('Final checkpoint/curve validation mismatch')
    rows=[];curves={}
    historical=json.loads((root/'docs/vpr_l2_reference/v1_final_audit.json').read_text())
    for record in historical['records']:
        if record['seed']!=0:continue
        method=record['method'];epochs=json.loads((EXT/f'reference/{method}_seed0_recovery_epochs.json').read_text())
        if epochs[-1]['validation']!=record['recovered']:raise RuntimeError('Historical endpoint audit mismatch')
        curves[method]=epochs
        rows.append(dict(method=method,seed=0,dense=record['dense'],raw_hard=record['hard'],
            recovery=curve_metrics(epochs,record['hard'],record['dense']),checkpoint_sha256=record['checkpoint_hashes']['recovered']))
    curves['fixed_l2']=final['epoch_rows']
    rows.append(dict(method='fixed_l2',seed=0,dense=evaluation['dense']['metrics'],raw_hard=evaluation['raw_hard']['hard_validation'],
        recovery=curve_metrics(final['epoch_rows'],evaluation['raw_hard']['hard_validation'],evaluation['dense']['metrics']),
        checkpoint_sha256=file_hash(final_path)))
    audit=dict(status='PASS',producer_commit=EXPECTED_DENSE_COMMIT,extension_commit=pre['extension_commit'],
        recovery_epochs=20,recovery_steps=7040,recovery_order_hash=final['order_hash'],raw_hard_parent_sha256=file_hash(hard),
        final_recovery_sha256=file_hash(final_path),mask_sha256=raw_mask_hash,model_sha256=tensor_hash(final['model']),
        initialization_hash=final['initialization_hash'],dataset_fingerprint=final['manifest']['fingerprint'],
        recovery_protocol=rc,checkpoint_selection='final',fresh_sgd=True,raw_bn_parent=True,
        fixed_mask=True,test_used=False,additional_seeds=False,elapsed_seconds=result['elapsed_seconds'],
        peak_cuda_bytes=result['peak_cuda_bytes'],comparison=rows)
    atomic_json(out/'final_audit.json',audit);atomic_json(out/'curves.json',curves)
    make_report(root,out,audit,curves)
    frozen(root)


def make_report(root,out,audit,curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    table=[]
    for row in audit['comparison']:
        m=row['recovery'];table.append(dict(method=row['method'],dense_accuracy=row['dense']['accuracy'],
            raw_hard_accuracy=row['raw_hard']['accuracy'],final_accuracy=m['final']['accuracy'],final_loss=m['final']['loss'],
            accuracy_aulc=m['accuracy_aulc_normalized'],hard_to_recovery_gain=m['hard_to_recovery_accuracy_gain'],
            dense_to_recovery_gap=m['dense_to_recovery_accuracy_gap']))
    with (out/'comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    with (out/'recovery_epochs.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['method','epoch','validation_accuracy','validation_loss']);w.writeheader()
        for method,epochs in curves.items():
            for e in epochs:w.writerow(dict(method=method,epoch=e['epoch'],validation_accuracy=e['validation']['accuracy'],validation_loss=e['validation']['loss']))
    for metric in ('accuracy','loss'):
        fig,ax=plt.subplots(figsize=(7,4))
        for row in audit['comparison']:
            method=row['method'];values=[row['raw_hard'][metric]]+[e['validation'][metric] for e in curves[method]]
            ax.plot(range(21),values,label=method)
        ax.set(xlabel='Recovery epoch (0 = Raw Hard)',ylabel='Validation '+metric);ax.legend();ax.grid(alpha=.2);fig.tight_layout()
        for suffix in ('png','pdf'):fig.savefig(out/f'recovery_{metric}.{suffix}',dpi=180)
        plt.close(fig)
    text='# VPR-L2 seed0: authorized fixed-mask recovery\n\n'
    text+='Primary outcomes are pruning robustness and recoverability; dense accuracy alone does not establish method success.\n\n'
    text+='Exactly one newly authorized 20-epoch / 7040-step SGD recovery completed. Parent: original Raw Hard, never BN-recalibrated. '
    text+='Fresh optimizer, fixed original magnitude mask, unchanged V1 seed0 recovery protocol and sample order, final checkpoint selection. No Test or extra seed.\n\n'
    text+='AULC: trapezoidal integration at integer epochs 0..20, with epoch0 = Raw Hard validation accuracy. '
    text+='Primary normalized AULC divides area by20 and uses accuracy fractions (multiply by100 for percentage points). '
    text+='Raw area, normalized 1..20 variant and loss AULC are also retained; all methods use the same definitions.\n\n'
    text+='| method | dense acc | raw hard acc | recovery acc | recovery loss | accuracy AULC | hard→recovery gain | dense→recovery gap |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
    for r in table:text+='| '+r['method']+' | '+' | '.join(f'{r[k]:.9f}' for k in list(r)[1:])+' |\n'
    text+='\nSingle-seed exploratory comparison; no significance or guaranteed superiority claim. BN-only results remain distinct from Raw Hard and recovery.\n\n'
    text+='![Recovery accuracy](artifacts/vpr_l2/recovery_extension/recovery_accuracy.png)\n\n![Recovery loss](artifacts/vpr_l2/recovery_extension/recovery_loss.png)\n\n'
    text+='Full audited configuration and identities:\n\n```json\n'+json.dumps(audit,indent=2)+'\n```\n'
    (root/'VPR_L2_RECOVERY_REPORT.md').write_text(text,encoding='utf8')
    archive=out/'before_recovery_reports';archive.mkdir()
    for name in ('TRAINING_REPORT','GEOMETRY_ANALYSIS','PRUNING_EVALUATION','SEED0_COMPARISON','FINAL_AUDIT'):
        p=root/f'VPR_L2_{name}.md';shutil.copyfile(p,archive/p.name)
        p.write_text(p.read_text()+'\n\n---\n\n## Authorized evaluation extension\n\n'
            'The preceding section records the original dense/non-training scope. The subsequently authorized unique20-epoch fixed-mask SGD recovery is now complete. '
            'The recovery-inclusive audit and comparison supersede the earlier no-recovery scope statement.\n\n'+text,encoding='utf8')
    protocol=root/'VPR_L2_PROTOCOL_FREEZE.md'
    shutil.copyfile(protocol,archive/protocol.name)
    protocol.write_text(protocol.read_text()+'\n\n## Subsequent explicit user authorization\n\n'+(EXT/'AUTHORIZATION.md').read_text(),encoding='utf8')
    previous=root/'artifacts/vpr_l2/final_audit.json';shutil.copyfile(previous,out/'dense_scope_final_audit.json')
    extended=json.loads(previous.read_text());extended.update(recovery_executed=True,recovery_audit=audit,
        primary_research_outcomes=['N:M pruning robustness','fixed-mask recoverability'],dense_accuracy_sufficient_for_success=False)
    atomic_json(previous,extended)
    hashes={str(p.relative_to(root)):file_hash(p) for p in out.rglob('*') if p.is_file() and p.name!='DELIVERY_SHA256.json'}
    hashes.update({str(p.relative_to(root)):file_hash(p) for p in root.glob('VPR_L2_*.md')})
    atomic_json(out/'DELIVERY_SHA256.json',hashes)


def main(root):
    out=root/'artifacts/vpr_l2/recovery_extension';out.mkdir(parents=True,exist_ok=True)
    lock=(out/'extension.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status=out/'status.json'
    if status.exists():raise RuntimeError('Extension already registered; no duplicate/restart')
    frozen(root)
    atomic_json(status,dict(status='WAITING',reason='Waiting for original dense and non-training controller to complete',pid=os.getpid()))
    while True:
        base=json.loads((root/'artifacts/vpr_l2/control/status.json').read_text())
        if (root/'artifacts/vpr_l2/control/failure.json').exists() or base['status']=='FAILED':
            raise RuntimeError('Original campaign failed; recovery is not started')
        if base['status']=='COMPLETE':break
        time.sleep(30)
    frozen(root)
    gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if gpu:raise RuntimeError('GPU occupied after original pipeline; no concurrent recovery launch')
    child=None
    try:
        with (out/'recovery.log').open('x') as log:
            child=subprocess.Popen([sys.executable,'-u',str(EXT/'runner.py'),'--project',str(root),'--worker'],cwd=root,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            atomic_json(status,dict(status='RUNNING',pid=child.pid,controller_pid=os.getpid(),only_recovery_epochs=20))
            while child.poll() is None:
                if shutil.disk_usage(root).free<4*1024**3:raise RuntimeError('Disk reserve below4GiB')
                frozen(root)
                gpu=subprocess.run(['nvidia-smi','--query-gpu=uuid,memory.used','--format=csv,noheader'],capture_output=True,text=True,timeout=15)
                if gpu.returncode:raise RuntimeError('GPU/hardware monitoring failed')
                atomic_json(out/'telemetry.json',dict(time=time.time(),gpu=gpu.stdout.strip(),free_bytes=shutil.disk_usage(root).free))
                time.sleep(30)
            if child.returncode:raise RuntimeError(f'Recovery worker exit{child.returncode}; preserve recovery.log')
        atomic_json(status,dict(status='COMPLETE',recovery_epochs=20,recovery_steps=7040,further_gpu_tasks=0,time=time.time()))
        hashes={str(p.relative_to(root)):file_hash(p) for p in (root/'artifacts/vpr_l2').rglob('*')
            if p.is_file() and p.name!='DELIVERY_SHA256.json'}
        hashes.update({str(p.relative_to(root)):file_hash(p) for p in root.glob('VPR_L2_*.md')})
        atomic_json(out/'DELIVERY_SHA256.json',hashes)
        atomic_json(root/'artifacts/vpr_l2/DELIVERY_SHA256.json',hashes)
    except BaseException:
        if child and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--worker',action='store_true');args=p.parse_args()
    try:
        if args.worker:worker(args.project.resolve())
        else:main(args.project.resolve())
    except BaseException as exc:
        out=args.project/'artifacts/vpr_l2/recovery_extension';out.mkdir(parents=True,exist_ok=True)
        atomic_json(out/'failure.json',dict(status='FAILED',error=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),automatic_restart=False))
        raise
