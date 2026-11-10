"""Stage1 hard gate: original local15 Best files, aliases and exact CIFAR model."""
import argparse,json
from pathlib import Path
import torch
from .common import atomic_json,environment,finite,object_sha,sha_file,tensor_sha
from .model import build_model
from .nm import eligible_layers,validate_layer_mask

def load_original(path,expected_sha=None):
    if not Path(path).is_file():raise FileNotFoundError('Required existing checkpoint missing: '+str(path))
    digest=sha_file(path)
    if expected_sha is not None and digest!=expected_sha:raise ValueError('Original checkpoint SHA256 mismatch: '+str(path))
    value=torch.load(path,map_location='cpu',weights_only=False)
    if value['schema']!=2 or value['checkpoint_status']!='healthy' or value['parameter_location']!='unperturbed_after_sgd':
        raise ValueError('Original schema/health/location')
    if tensor_sha(value['model'])!=value['model_sha256']:raise ValueError('Original model component digest')
    finite(value['model'].values(),'Original checkpoint')
    if 'geometry' in value:
        g=value['geometry'];mapping=[(n,tuple(s)) for n,s in g['mapping']]
        model=build_model();expected=[(n,tuple(p.shape)) for n,p in model.named_parameters()]
        if mapping!=expected or g['d']!=sum(p.numel() for p in model.parameters()) or g['method']!='fixed_l2':
            raise ValueError('APBR geometry parameter identity')
        logs={n:s for (n,_),s in zip(g['mapping'],g['logs'])}
        if len(logs)!=len(mapping) or any(s.dtype!=torch.float64 or s.shape!=value['model'][n].shape for n,s in logs.items()):
            raise ValueError('FP64 geometry coordinates')
        finite(logs.values(),'Original geometry')
        if tensor_sha(logs)!=value['geometry_sha256']:raise ValueError('Original geometry component digest')
    return value,digest

def audit(root,output):
    root=Path(root);manifest=json.loads((root/'best_checkpoints/verified_manifest.json').read_text(encoding='utf-8'))
    if manifest['status']!='PASS' or manifest['count']!=15:raise ValueError('Existing verified manifest missing')
    records=[];by_role={};forward=[];identities={}
    torch.set_num_threads(4)
    for item in manifest['checkpoints']:
        path=root/'best_checkpoints'/item['relative_path'];value,digest=load_original(path,item['sha256'])
        seed=item['seed'];method=item['method'];stage=item['stage']
        if value['identity']['seed']!=seed:raise ValueError('Checkpoint seed')
        if stage=='dense':
            model=build_model().eval();model.load_state_dict(value['model'],strict=True)
            before=tensor_sha(model.state_dict())
            with torch.no_grad():scores=model(torch.zeros(1,3,32,32))
            if scores.shape!=(1,10) or not torch.isfinite(scores).all() or tensor_sha(model.state_dict())!=before:
                raise ValueError('Exact architecture forward / BN changed')
            role='standard_dense_sam' if method=='dense_sam' else 'apbr_dense_with_geometry'
            if (method=='dense_sam')==('geometry' in value):raise ValueError('Dense SAM/APBR mislabeled')
            if sum(p.numel() for p in model.parameters())!=11173962:raise ValueError('Wrong architecture')
            forward.append({'seed':seed,'method':method,'synthetic_cpu_forward':'PASS','not_dataset_accuracy':True})
            identities[seed,method]=value['identity']
        else:
            role={'sam_magnitude':'sam_weight_only_mask_recovery','vpr_magnitude':'apbr_weight_only_mask_recovery',
                  'vpr_guided':'full_apbr_mask_recovery'}[method]
            if tensor_sha(value['masks'])!=value['mask_sha256']:raise ValueError('Stored Mask bytes')
            for name,mask in value['masks'].items():validate_layer_mask(value['model'][name],mask)
        record={'seed':seed,'role':role,'historical_label':method,'stage':stage,'path':str(path.resolve()),'sha256':digest,
            'epoch':value['epoch'],'steps':value['steps'],'model_sha256':value['model_sha256'],'producer_commit':value['identity']['commit'],
            'geometry_sha256':value.get('geometry_sha256'),'geometry_state':'embedded_in_same_dense_checkpoint' if 'geometry' in value else None,
            'mask_sha256':value.get('mask_sha256'),'geometry_reference':value.get('geometry_reference')}
        records.append(record);by_role.setdefault(role,[]).append(record)
    for seed in (0,1,2):
        a,b=identities[seed,'dense_sam'],identities[seed,'vpr_l2']
        if a['initialization_hash']!=b['initialization_hash'] or a['data_manifest_sha256']!=b['data_manifest_sha256']:
            raise ValueError('Historical pair mismatch')
    model=build_model();layers=eligible_layers(model)
    layout={'architecture':'SSAM2022 CIFAR-ResNet18; not torchvision ImageNet ResNet18',
        'parameters':sum(p.numel() for p in model.parameters()),'geometry_d':11173962,
        'geometry_parameter_tensors':len(list(model.named_parameters())),
        'eligible_layers':{n:list(m.weight.shape) for n,m in layers.items()},'eligible_coordinates':sum(m.weight.numel() for m in layers.values()),
        'all_eligible_rows_divisible_by4':all(m.weight[0].numel()%4==0 for m in layers.values()),
        'snows_masked_parameters_above100k':[n for n,m in layers.items() if m.weight.numel()//2>100000]}
    result={'status':'PASS','stage1_complete':True,'apbr_alias_confirmed_by_user':True,
        'alias':'APBR-SAM = existing Full VPR-L2+Geometry-guided; historical labels and formulas untouched',
        'checkpoint_records':records,'roles':by_role,'cpu_forwards':forward,'layout':layout,'environment':environment(),
        'limitations':['Raw/BN/Last/all-epoch large checkpoints remain on shutdown server; existing metrics/logs/Mask states and all15Best are local',
                      'No formal dataset evaluation, pruning or new training performed by this audit']}
    atomic_json(output,result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input-root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    r=audit(a.input_root,a.output);print(json.dumps({'status':r['status'],'checkpoint_count':len(r['checkpoint_records']),'architecture':r['layout']}))
