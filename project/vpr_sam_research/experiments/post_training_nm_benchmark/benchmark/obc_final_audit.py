"""CPU-only completed-run audit; Final is the primary endpoint, never Best Test."""
import json
from pathlib import Path
import torch
from . import artifacts
from .audit_inputs import load_original
from .common import sha_file,finite
from .model import build_model
from .nm import eligible_layers

def audit(out,parent_path,parent_sha,identity):
    out=Path(out);torch.set_num_threads(4)
    completion=json.loads((out/'completion.json').read_text())
    if completion['status']!='COMPLETE' or completion['identity']!=identity or list(out.glob('failure*')):
        raise ValueError('Completed healthy run required')
    parent,digest=load_original(parent_path,parent_sha)
    if parent['identity']['seed']!=identity['seed'] or 'geometry' in parent or parent['identity']['data_manifest_sha256']!=identity['data_manifest_sha256']:
        raise ValueError('Dense parent identity')
    native_path=out/'native/native_complete.pt';native_sha=sha_file(native_path)
    native=artifacts.load(native_path,identity);mask_sha=native['mask_sha256']
    model=build_model();names=[n for n,_ in model.named_parameters()]
    expected=[n+'.weight' for n in eligible_layers(model)]
    if len(expected)!=19 or set(native['masks'])!=set(expected) or native['parent']['sha256']!=digest or native['dense_model_sha256']!=parent['model_sha256']:
        raise ValueError('Full native scope/parent')
    changed=[n for n in parent['model'] if not torch.equal(parent['model'][n],native['model'][n])]
    if not set(changed)<=set(expected):raise ValueError('Native changed excluded weights or BN')
    replay=json.loads((out/'calibration_replay.json').read_text())
    if len(replay)!=19 or any(x!=replay[0] for x in replay) or (replay[0]['samples_including_rounds'],replay[0]['batches'])!=(10240,80):
        raise ValueError('Calibration replay/coverage')
    if len(native['layer_receipts'])!=19 or any(x['samples_including_rounds']!=10240 or x['algorithm']!='official_TrueOBS_nmprune' or x['cpu_timing_fence_only'] is not False for x in native['layer_receipts']):
        raise ValueError('Official layer receipts')
    if (out/'common_bn.pt').exists() or completion['metrics']['common_bn']['status']!='SKIPPED_BY_USER':raise ValueError('BN-only not authorized')
    history=json.loads((out/'recovery/epochs.json').read_text())
    if len(history)!=20 or any((x['epoch'],x['steps'],x['train_samples'],x['test_count'])!=(i,390*i,49920,10000) for i,x in enumerate(history,1)):
        raise ValueError('Recovery budget/coverage')
    if completion['recovery']['history']!=history or completion['recovery']['final']!=history[-1]:raise ValueError('Completion metrics mismatch')
    best=max(history,key=lambda x:x['test_accuracy'])
    if completion['recovery']['best']!={k:best[k] for k in ('epoch','test_accuracy','test_loss')}:raise ValueError('Auxiliary Best selection')
    paths=[out/'native/native_boundary.pt',native_path]+[out/f'recovery/epoch_{i:03d}.pt' for i in range(1,21)]+[out/'recovery/best_test.pt',out/'recovery/last.pt']
    records=[];epoch_models={}
    for path in paths:
        v=artifacts.load(path,identity)
        if v['mask_sha256']!=mask_sha or any(not torch.equal(v['masks'][n],native['masks'][n]) for n in expected):raise ValueError('Mask changed')
        record={'path':str(path.relative_to(out)),'sha256':sha_file(path),'model_sha256':v['model_sha256'],'mask_sha256':v['mask_sha256'],'stage':v['stage']}
        if v['stage']=='recovery':
            epoch=v['epoch'];record.update(epoch=epoch,steps=v['steps'])
            if v['parent']!={'sha256':native_sha,'stage':'native_complete'} or v['steps']!=390*epoch or v['history']!=history[:epoch]:raise ValueError('Recovery source/history')
            groups=[i for g in v['optimizer']['param_groups'] for i in g['params']]
            for state in v['optimizer']['state'].values():finite((x for x in state.values() if isinstance(x,torch.Tensor)),'Optimizer state')
            for name,mask in native['masks'].items():
                if torch.count_nonzero(v['optimizer']['state'][groups[names.index(name)]]['momentum_buffer'][~mask]):raise ValueError('Deleted momentum nonzero')
            if v['scheduler']['last_epoch']!=epoch or set(v['rng'])!={'python','numpy','cpu','cuda'}:raise ValueError('Scheduler/RNG state')
            if path.name.startswith('epoch_'):epoch_models[epoch]=v['model_sha256']
            if path.name=='best_test.pt' and (epoch!=best['epoch'] or v['model_sha256']!=epoch_models[epoch]):raise ValueError('Best publication mismatch')
            if path.name=='last.pt' and (epoch!=20 or v['model_sha256']!=epoch_models[20]):raise ValueError('Final publication mismatch')
        elif v['model_sha256']!=native['model_sha256']:raise ValueError('Native publication mismatch')
        records.append(record);del v
    published=json.loads((out/'recovery/checkpoint_hashes.json').read_text())
    if any(published[name]!=sha_file(out/'recovery'/name) for name in ('best_test.pt','last.pt')):raise ValueError('Published hashes')
    return {'status':'PASS','seed':identity['seed'],'identity':identity,'primary_checkpoint':'recovery/last.pt','primary_epoch':20,
            'metrics':completion['metrics'],'recovery_final':history[-1],'auxiliary_best':best,'history':history,
            'mask_sha256':mask_sha,'artifacts':records,'calibration':replay[0],
            'peak_allocated_bytes':completion['gpu_peak_allocated_bytes'],
            'small_file_sha256':{str(p.relative_to(out)):sha_file(p) for p in out.rglob('*') if p.is_file() and p.suffix in ('.json','.jsonl')}}
