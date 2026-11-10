"""Validate frozen32-image smoke identity independently of CUDA availability."""
import json
from pathlib import Path
from .calibration import approved_indices
from .common import object_sha,sha_file
from .preflight import ROOT

def verify(c,formal):
    ids=approved_indices(dict(c,data=formal['data']))
    if (len(ids),c['calibration']['random_seed'],c['calibration']['batch_size'],c['calibration']['rounds'])!=(32,0,8,1):
        raise ValueError('Smoke differs from approved32/seed0/batch8/1round')
    if c.get('precision')!={'model':'float32','amp':False,'tf32':False,'compile':False}:raise ValueError('Smoke precision changed')
    if c.get('model')!='cifar_resnet18_ssam2022':raise ValueError('Smoke model changed')
    proof_path=ROOT/c['freeze_record']
    if sha_file(proof_path)!=c['freeze_record_sha256']:raise ValueError('Smoke freeze-record digest')
    proof=json.loads(proof_path.read_text(encoding='utf-8'))
    if sha_file(c['calibration']['index_file'])!=c['calibration']['index_file_sha256'] or ids!=proof['ids']:raise ValueError('Frozen index bytes/order')
    if proof['runtime_data_source_sha256']!=sha_file(ROOT/'benchmark/data.py'):raise ValueError('Actual preprocessing source changed')
    if any(c['calibration']['preprocessing'][k]!=v for k,v in proof['parent_normalization'].items()):raise ValueError('Parent Normalize mismatch')
    manifest=dict(proof['dataset_manifest_compact'],train_indices=list(range(50000)),test_indices=list(range(10000)))
    if object_sha(manifest)!=proof['dataset_manifest_sha256']:raise ValueError('Frozen dataset manifest changed')
    if c['input']['sha256']!=proof['parent_checkpoint_sha256'] or c['input']['role']!='standard_dense_sam':raise ValueError('Smoke parent mismatch')
    return proof
