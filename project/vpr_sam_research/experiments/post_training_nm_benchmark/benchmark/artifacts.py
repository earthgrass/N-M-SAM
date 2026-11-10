"""Atomic new-baseline artifacts; original checkpoint files are never rewritten."""
from pathlib import Path
import torch
from .common import finite,sha_file,tensor_sha

def validate(value,expected_identity=None):
    if value['schema']!=2 or value['kind']!='post_training_nm_baseline' or value['checkpoint_status']!='healthy':
        raise ValueError('New baseline schema/kind/health')
    if expected_identity is not None and value['identity']!=expected_identity:raise ValueError('Resume identity mismatch')
    finite(value['model'].values(),'Baseline artifact weights')
    if tensor_sha(value['model'])!=value['model_sha256'] or tensor_sha(value['masks'])!=value['mask_sha256']:
        raise ValueError('Artifact weight/Mask digest')
    if value['stage'] not in ('native_partial','native_complete','native_bn','common_bn','recovery'):raise ValueError('Artifact stage')
    params=value['model']
    from .nm import validate_layer_mask
    for name,mask in value['masks'].items():validate_layer_mask(params[name],mask)

def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    value=dict(value,schema=2,kind='post_training_nm_baseline',checkpoint_status='healthy',
        model_sha256=tensor_sha(value['model']),mask_sha256=tensor_sha(value['masks']))
    validate(value);tmp=path.with_name(path.name+'.tmp')
    import os
    with tmp.open('wb') as f:torch.save(value,f);f.flush();os.fsync(f.fileno())
    validate(torch.load(tmp,map_location='cpu',weights_only=False),value['identity']);tmp.replace(path)
    validate(torch.load(path,map_location='cpu',weights_only=False),value['identity'])
    return sha_file(path)

def load(path,identity=None):
    value=torch.load(path,map_location='cpu',weights_only=False);validate(value,identity);return value
