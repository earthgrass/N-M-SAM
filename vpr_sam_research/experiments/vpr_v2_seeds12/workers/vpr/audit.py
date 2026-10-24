"""Independent CPU-only source, history, checkpoint and metric audits."""
import csv
import json
import math
from pathlib import Path

import torch

from state import atomic_json,atomic_text,object_sha,sha_file,source_freeze,validate_checkpoint

HISTORICAL_ROOTS=[
    '/root/autodl-tmp/vpr_v2_unified50k_20261024/project',
    '/root/autodl-tmp/vpr_acceptance_20261023/acceptance_fixed/vpr_sam_research',
    '/root/autodl-tmp/vpr_l2_seed0_20261024/workspace/vpr_sam_research',
    '/root/autodl-tmp/vpr_l2_guided_recovery_seed0_20261024/vpr_sam_research',
    '/root/autodl-tmp/dense_sam_anchor_20261024/project',
    '/root/autodl-tmp/dense_sam_ssam50k_20261024/project',
    '/root/autodl-tmp/dense_sam_ssam50k_20261024_lr_native_v1/project',
]


def capture_history(path):
    records={}
    for name in HISTORICAL_ROOTS:
        root=Path(name)
        if not root.exists():
            raise FileNotFoundError('Historical workspace missing: '+name)
        for p in root.rglob('*'):
            if p.is_file() and not p.is_symlink() and not any(k in p.relative_to(root).parts for k in
                ('.venv','venv','data','.git','__pycache__','.pytest_cache')):
                records[str(p)]=sha_file(p)
    atomic_json(path,records)
    return len(records)


def verify_history(path):
    records=json.loads(Path(path).read_text())
    differences=[p for p,digest in records.items() if not Path(p).is_file() or sha_file(p)!=digest]
    if differences:
        raise ValueError('Historical files changed: '+str(differences[:12]))
    return len(records)


def sums(out):
    files=[p for p in Path(out).iterdir() if p.is_file() and p.name not in ('SHA256SUMS',)
           and p.suffix in ('.pt','.json','.csv','.jsonl','.md','.log')]
    atomic_text(Path(out)/'SHA256SUMS',''.join(f'{sha_file(p)}  {p.name}\n' for p in sorted(files)))


def final_audit(root,out):
    root,out=Path(root),Path(out)
    frozen=source_freeze(root)
    rows=[json.loads(s) for s in (out/'epoch_metrics.jsonl').read_text().splitlines()]
    csv_rows=list(csv.DictReader((out/'epoch_metrics.csv').open(newline='')))
    if [r['epoch'] for r in rows]!=list(range(1,201)) or len(csv_rows)!=200:
        raise ValueError('Epoch coverage')
    for i,row in enumerate(rows):
        if row['steps']!=(i+1)*390 or row['train_samples']!=49920 or row['test_count']!=10000:
            raise ValueError('Samples/steps/evaluation coverage')
        if float(csv_rows[i]['test_accuracy'])!=row['test_accuracy'] or float(csv_rows[i]['test_loss'])!=row['test_loss']:
            raise ValueError('CSV/JSONL disagreement')
        lr_used=.05 if i==0 else .05*(1+math.cos(math.pi*(i-1)/200))/2
        lr_after=.05*(1+math.cos(math.pi*i/200))/2
        if row['learning_rate_used']!=lr_used or row['learning_rate_after_epoch']!=lr_after:
            raise ValueError('Official LR trajectory mismatch')
        if len(row['omitted_sample_ids'])!=80 or len(set(row['omitted_sample_ids']))!=80:
            raise ValueError('Omitted tail sample log')
    best=max(rows,key=lambda r:(r['test_accuracy'],-r['epoch']))
    selected={}; common_identity=None
    for name,expected in [('best_test.pt',best),('last.pt',rows[-1])]:
        p=out/name
        value=torch.load(p,map_location='cpu',weights_only=False)
        validate_checkpoint(value)
        if common_identity is None: common_identity=value['identity']
        elif value['identity']!=common_identity: raise ValueError('Best/Last different identity')
        if value['identity']['commit']!=frozen['commit'] or value['identity']['source_sha256']!=frozen['source_sha256']:
            raise ValueError('Source provenance')
        if value['epoch']!=expected['epoch'] or value['history'][-1]!=expected:
            raise ValueError('Selection/checkpoint mismatch')
        for key,tensor in value['model'].items():
            if key.endswith('num_batches_tracked') and tensor.item()!=expected['steps']*2:
                raise ValueError('BN counters not two per step: '+key)
        selected[name]={'epoch':value['epoch'],'test_accuracy':expected['test_accuracy'],
                        'test_loss':expected['test_loss'],'train_loss':expected['train_loss'],
                        'sha256':sha_file(p),'bytes':p.stat().st_size}
    history_count=verify_history(root/'artifacts/historical_before.json')
    receipt={'status':'PASS','epochs':200,'optimizer_steps':78000,'train_samples_per_epoch':49920,
             'total_training_presentations':9984000,'dataset_train_size':50000,'test_count':10000,
             'selection':'best Test Accuracy with earliest tie; Test is not independent generalization evidence',
             'selected':selected,'historical_files_unchanged':history_count,'source_commit':frozen['commit'],
             'bn_updates_per_step':2,'bn_calibration':False,'extra_training_or_pruning':False}
    atomic_json(out/'FINAL_AUDIT.json',receipt)
    return receipt


if __name__=='__main__':
    root=Path(__file__).resolve().parent
    import yaml
    c=yaml.safe_load((root/'config.yaml').read_text())
    print(json.dumps(final_audit(root,root/c['output']),indent=2))
