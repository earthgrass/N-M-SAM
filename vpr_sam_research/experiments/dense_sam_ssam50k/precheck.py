"""Mandatory CPU/CUDA gates and disposable full-model real-data checks."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
import yaml

from audit import capture_history
from core import bn_snapshot,evaluate,sam_step
from data import make_loader,real_data,verify_worker_rng
from model import build_model
from state import (atomic_json,disk_check,environment,gpu_check,object_sha,seed_all,
                   sha_file,source_freeze,tensor_sha)
from train import optimizer_for

ROOT=Path(__file__).resolve().parent


def main():
    c=yaml.safe_load((ROOT/'config.yaml').read_text())
    gpu=gpu_check(c); free=disk_check(ROOT,c); source=source_freeze(ROOT)
    upstream=json.loads((ROOT/'reference/UPSTREAM_SHA256.json').read_text())
    for repo,record in upstream.items():
        for name,digest in record['files'].items():
            if sha_file(ROOT/'reference'/repo/name)!=digest:
                raise ValueError('Upstream source SHA256 failed: '+repo+'/'+name)
    artifacts=ROOT/'artifacts'; artifacts.mkdir(exist_ok=True)
    receipts={}
    for name,device in [('cpu','cpu'),('cuda',c['device'])]:
        result=subprocess.run([sys.executable,'-m','pytest','-q','tests','--basetemp=artifacts/pytest-'+name],
                              cwd=ROOT,env=dict(os.environ,ANCHOR_TEST_DEVICE=device),text=True,
                              stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        log=artifacts/('tests_'+name+'.log'); log.write_text(result.stdout,encoding='utf-8')
        if result.returncode: raise RuntimeError(f'{name} tests failed; see {log}')
        receipts[name]={'exit_code':0,'device':device,'sha256':sha_file(log)}
    historical_count=capture_history(artifacts/'historical_before.json')
    seed_all(c['seed'],c['threads'])
    train,test,manifest=real_data(c)
    assert len(make_loader(train,c,training=True))==390
    assert len(make_loader(test,c))==79
    model=build_model().to(c['device']); opt,scheduler=optimizer_for(model,c)
    initial=tensor_sha(model.state_dict()); before=bn_snapshot(model); rows=[]
    workers=verify_worker_rng()
    atomic_json(artifacts/'worker_rng_audit.json',workers)
    expected=[]
    temp_opt=torch.optim.SGD([torch.nn.Parameter(torch.zeros(()))],lr=c['learning_rate'])
    from scheduler import CosineLRscheduler
    schedule=CosineLRscheduler(temp_opt,T_max=200,eta_min=0.,warmup_epoch=0,warmup_init_lr=0.)
    for epoch in range(200):
        used=temp_opt.param_groups[0]['lr']; schedule.step(epoch)
        expected.append({'epoch':epoch+1,'used':used,'after_epoch':temp_opt.param_groups[0]['lr']})
    if expected!=json.loads((ROOT/'reference/LR_EXPECTED.json').read_text()):
        raise ValueError('200-epoch LR freeze mismatch')
    started=time.monotonic()
    for batch,(x,y,ids) in enumerate(make_loader(train,c,training=True)):
        if batch>=2: break
        result=sam_step(model,opt,x.to(c['device']),y.to(c['device']),c['rho'],c['numerical_epsilon'],audit_bn=True)
        rows.append({**result,'sample_ids':ids.tolist()})
    after=bn_snapshot(model)
    assert all(v['num_batches_tracked'].item()==before[n]['num_batches_tracked'].item()+4 for n,v in after.items())
    for r in rows:
        for v in r['bn_audit'].values():
            assert v['first_mean_change']>0 and v['second_mean_change']>0
            assert v['first_var_change']>0 and v['second_var_change']>0
    # Evaluation-mode smoke uses training images; it never pre-selects on Test.
    eval_data=[next(iter(make_loader(train,c,training=False,workers=0)))]
    evaluation=evaluate(model,eval_data,c['device'])
    atomic_json(artifacts/'precheck.json',{'status':'PASS','commit':source['commit'],
                'config_sha256':object_sha(c),'data_manifest_sha256':object_sha(manifest),
                'initialization_hash':initial,'gpu':gpu,'environment':environment(),
                'free_disk_bytes':free,'historical_files_captured':historical_count,'tests':receipts,
                'train_dataset_size':len(train),'test_dataset_size':len(test),'validation_size':0,
                'train_steps_per_epoch':390,'train_samples_per_epoch':49920,'formal_total_steps':78000,
                'real_data_train_batches':rows,'training_subset_eval_smoke':evaluation,
                'bn_counter_increment_two_steps':4,'elapsed_seconds':time.monotonic()-started,
                'worker_rng_audit':workers,'lr_sequence_sha256':sha_file(ROOT/'reference/LR_EXPECTED.json'),
                'test_evaluated_in_precheck':False,'smoke_weights_discarded':True})
    report=(f"# Final prelaunch audit: PASS\n\nCommit: {source['commit']}\n\n"
            f"Seed0; config SHA256 {object_sha(c)}; initialization {initial}.\n\n"
            f"Full50000/10000 official MD5 checks; no validation split;390steps/49920samples per epoch;78000formal steps.\n\n"
            f"BN both forwards update all running buffers and counters, matching SSAM; F-SAM momentum0 is not used.\n\n"
            f"CPU/CUDA receipts: {json.dumps(receipts)}\n\n"
            f"Worker Python/NumPy/Torch initialization: PASS;200-point exact official LR sequence: PASS.\n\n"
            f"GPU: {json.dumps(gpu)}; free disk bytes {free}; output {ROOT/c['output']}.\n\n"
            f"Checkpoint/RNG resume: tests PASS. Historical files captured: {historical_count}. Smoke weights discarded.\n\n"
            f"Best Test selection, earliest tie; Test participates in model selection and is not an independent unbiased estimate.\n")
    from state import atomic_text
    atomic_text(artifacts/'PRELAUNCH_AUDIT.md',report)
    print('PRECHECK PASS; full50k/10k, dual BN,390steps/epoch; disposable smoke weights discarded',flush=True)


if __name__=='__main__': main()
