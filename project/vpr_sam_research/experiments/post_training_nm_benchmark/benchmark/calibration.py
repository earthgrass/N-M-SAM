"""Explicit train-only calibration. Pending budgets fail closed."""
import hashlib,json
from pathlib import Path
import numpy as np

def validate_indices(indices,train_size=50000):
    if not indices or any(type(i) is not int for i in indices):raise ValueError('Nonempty integer training IDs required')
    if len(set(indices))!=len(indices) or min(indices)<0 or max(indices)>=train_size:raise ValueError('Training IDs duplicate/out of range')
    return indices

def index_sha(indices):
    validate_indices(indices);return hashlib.sha256(np.asarray(indices,dtype='<i8').tobytes()).hexdigest()

def deterministic_indices(train_size,count,seed):
    if not 0<count<=train_size:raise ValueError('Invalid calibration count')
    return np.random.default_rng(seed).choice(train_size,count,replace=False).tolist()

def approved_indices(config):
    c=config['calibration']
    required=('samples','index_file','index_sha256','rounds','batch_size','forward_passes_per_layer',
              'optimization_epochs','random_seed','augmentation','preprocessing')
    if c['approved'] is not True or any(c.get(key) is None for key in required):
        raise ValueError('Calibration budget/indices/rounds are pending user decision')
    if c.get('dataset')!='official_cifar10_train_50000':raise ValueError('Calibration must use training images only')
    expected={'to_tensor':True,'normalize_mean':config['data']['normalize_mean'],'normalize_std':config['data']['normalize_std']}
    if c['preprocessing']!=expected:raise ValueError('Calibration preprocessing not frozen to dense input convention')
    if c['augmentation'] not in ('none','obc_random_resized_crop32_flip'):
        raise ValueError('Unreviewed calibration augmentation')
    if c['augmentation']=='obc_random_resized_crop32_flip':
        if (c['samples'],c['rounds'],c['batch_size'],c['random_seed'])!=(1024,10,128,0):
            raise ValueError('Reviewed OBC pilot budget must be1024/10/128/seed0')
        if c.get('transform_spec')!={'size':32,'scale':[0.2,1.0],'ratio':[0.75,4/3],'flip_probability':0.5}:
            raise ValueError('Reviewed OBC CIFAR transform changed')
    ids=json.loads(Path(c['index_file']).read_text(encoding='utf-8'))
    if c.get('index_file_sha256') is not None and hashlib.sha256(Path(c['index_file']).read_bytes()).hexdigest()!=c['index_file_sha256']:
        raise ValueError('Calibration index file byte identity changed')
    validate_indices(ids)
    if len(ids)!=c['samples'] or index_sha(ids)!=c['index_sha256']:raise ValueError('Calibration count/identity')
    if any(type(c[key]) is not int for key in ('rounds','batch_size','forward_passes_per_layer','optimization_epochs','random_seed')):
        raise ValueError('Calibration counters/seed must be integers')
    if c['rounds']<1 or c['batch_size']<1 or c['optimization_epochs']!=0:raise ValueError('Unsupported calibration rounds/batch/optimization epochs')
    expected_passes=c['rounds']*((len(ids)+c['batch_size']-1)//c['batch_size'])
    if c['forward_passes_per_layer']!=expected_passes:raise ValueError('Calibration forward budget mismatch')
    return ids
