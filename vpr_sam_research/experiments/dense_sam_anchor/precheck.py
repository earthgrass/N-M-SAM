"""Disposable CUDA/real-data smoke. Never supplies initial weights to formal training."""
import json
import os
import subprocess
import sys
from pathlib import Path
import time

import torch
import yaml

from core import bn_snapshot,evaluate,sam_step
from data import make_loader,real_data
from model import build_model
from state import atomic_json,disk_check,environment,gpu_check,object_sha,seed_all,sha_file,source_freeze,tensor_sha
from train import optimizer_for

ROOT=Path(__file__).resolve().parent


def main():
    c=yaml.safe_load((ROOT/'config.yaml').read_text())
    gpu=gpu_check(c); free=disk_check(ROOT,c); source=source_freeze(ROOT)
    (ROOT/'artifacts').mkdir(exist_ok=True)
    test_receipts={}
    for name,device in [('cpu','cpu'),('cuda',c['device'])]:
        env=dict(os.environ,ANCHOR_TEST_DEVICE=device)
        result=subprocess.run([sys.executable,'-m','pytest','-q','tests','--basetemp=artifacts/pytest-'+name],
                              cwd=ROOT,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        log=ROOT/'artifacts'/('tests_'+name+'.log')
        log.write_text(result.stdout,encoding='utf-8')
        if result.returncode:
            raise RuntimeError(f'{name} acceptance failed; see {log}')
        test_receipts[name]={'exit_code':result.returncode,'log_sha256':sha_file(log),'device':device}
    seed_all(c['seed'],c['threads'])
    train,validation,split=real_data(ROOT,c)
    model=build_model().to(c['device']); opt,scheduler=optimizer_for(model,c)
    initial=tensor_sha(model.state_dict())
    before=bn_snapshot(model)
    start=time.monotonic(); rows=[]
    for batch,(x,y,ids) in enumerate(make_loader(train,c,0,shuffle=True)):
        if batch>=2: break
        result=sam_step(model,opt,x.to(c['device']),y.to(c['device']),c['rho'],c['numerical_epsilon'])
        rows.append({**result,'sample_ids':ids.tolist()})
    after=bn_snapshot(model)
    assert all(v['num_batches_tracked'].item()==before[name]['num_batches_tracked'].item()+2 for name,v in after.items())
    small=[next(iter(make_loader(validation,c,workers=0)))]
    metrics=evaluate(model,small,c['device'])
    atomic_json(ROOT/'artifacts/precheck.json',{'status':'PASS','formal_training_started':False,
                'dataset':'official-cifar10-md5-verified','split_sha256':sha_file(ROOT/c['split_file']),
                'split_fingerprint':split['fingerprint'],'initialization_hash':initial,
                'source_commit':source['commit'],'environment':environment(),'gpu':gpu,
                'config_sha256':object_sha(c),'tests':test_receipts,
                'free_disk_bytes':free,'real_train_batches':rows,'validation_smoke':metrics,
                'bn_counters_increment':2,'elapsed_seconds':time.monotonic()-start,
                'test_evaluated':False,'smoke_weights_discarded':True})
    print('REAL DATA PRECHECK PASS; disposable weights discarded',flush=True)


if __name__=='__main__':
    main()
