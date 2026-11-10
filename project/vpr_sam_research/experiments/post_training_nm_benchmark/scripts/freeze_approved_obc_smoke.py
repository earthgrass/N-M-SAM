"""One-time user-approved32train-image freeze; reads only existing evidence."""
import argparse,ast,json,sys
from pathlib import Path
import numpy as np,yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from benchmark.audit_inputs import load_original
from benchmark.calibration import deterministic_indices,index_sha
from benchmark.common import atomic_json,object_sha,sha_file

def definition(text,name):
    return ast.dump(next(n for n in ast.parse(text).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name==name),include_attributes=False)

def main():
    p=argparse.ArgumentParser();p.add_argument('--input-root',required=True);a=p.parse_args();delivery=Path(a.input_root)
    output=ROOT/'configs/obc_smoke_train_ids.json'
    if output.exists():raise FileExistsError('Do not resample/overwrite frozen production IDs')
    items=json.loads((ROOT/'configs/existing_inputs.json').read_text(encoding='utf-8'))['checkpoints']
    item=next(r for r in items if r['role']=='standard_dense_sam' and r['seed']==0)
    parent,checkpoint_sha=load_original(delivery/item['relative_path'],item['sha256'])
    old=delivery/'full_evidence/sam_seed0';source=json.loads((old/'SOURCE_FREEZE.json').read_text(encoding='utf-8'))
    if source['source_sha256']!=parent['identity']['source_sha256']:raise ValueError('Parent source snapshot differs')
    for name in ('config.yaml','data.py'):
        if sha_file(old/name)!=source['source_sha256'][name]:raise ValueError('Parent source bytes changed')
    pc=parent['identity']['config'];old_config=yaml.safe_load((old/'config.yaml').read_text(encoding='utf-8'))
    if object_sha(old_config)!=parent['identity']['config_sha256'] or old_config!=pc:raise ValueError('Parent config mismatch')
    actual=(ROOT/'benchmark/data.py').read_text(encoding='utf-8');historic=(old/'data.py').read_text(encoding='utf-8')
    if definition(actual,'Images')!=definition(historic,'Images'):raise ValueError('Actual preprocessing class changed')
    manifest_path=old/'artifacts/dense_sam_ssam50k_20261024/seed0/data_manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    if object_sha(manifest)!=parent['identity']['data_manifest_sha256']:raise ValueError('Parent dataset identity mismatch')
    if manifest['train_indices']!=list(range(50000)) or manifest['test_indices']!=list(range(10000)):raise ValueError('Nonofficial sample order')
    ids=deterministic_indices(50000,32,0);atomic_json(output,ids)
    files=[f'data_batch_{i}' for i in range(1,6)]
    provenance={'approval':'explicit user authorization;32train/seed0/batch8/noaug/1round',
        'parent_checkpoint_sha256':checkpoint_sha,'parent_commit':parent['identity']['commit'],
        'parent_config_sha256':parent['identity']['config_sha256'],'parent_config_file_sha256':sha_file(old/'config.yaml'),
        'parent_data_source_sha256':sha_file(old/'data.py'),'runtime_data_source_sha256':sha_file(ROOT/'benchmark/data.py'),
        'images_class_AST_matches_parent':True,'parent_normalization':{k:pc[k] for k in ('normalize_mean','normalize_std')},
        'preprocessing':'Images(training=False): ToTensor -> Normalize; no crop/flip/Cutout/TTA',
        'sampling':{'algorithm':'numpy.random.default_rng(0).choice(50000,32,replace=False)',
            'bit_generator':'PCG64','numpy_at_generation':np.__version__,'output_order':'draw order retained; SequentialSampler over frozen subset'},
        'index_file_sha256':sha_file(output),'index_canonical_int64le_sha256':index_sha(ids),'ids':ids,
        'dataset_manifest_sha256':object_sha(manifest),'dataset_manifest_file_sha256':sha_file(manifest_path),
        'dataset_files_sha256':manifest['files_sha256'],'training_file_order':files,
        'sample_identity':[{'dataset_index':i,'file':files[i//10000],'row_zero_based':i%10000,
                            'file_sha256':manifest['files_sha256'][files[i//10000]],'pixel_layout':'serialized row CHW(3,32,32)'} for i in ids],
        'dataset_manifest_compact':{k:v for k,v in manifest.items() if k not in ('train_indices','test_indices')},
        'dataset_manifest_expand':'add train_indices=range(50000),test_indices=range(10000)',
        'data_availability':'file hashes attested by original checkpoint-bound manifest; server raw files rechecked at future smoke, not read here'}
    proof=ROOT/'configs/obc_smoke_freeze.json';atomic_json(proof,provenance)
    path=ROOT/'configs/obc_cuda_smoke.yaml';c=yaml.safe_load(path.read_text(encoding='utf-8'))
    c['review']={'approved':True,'status':'APPROVED_OFFLINE_PREPARATION_ONLY_NO_GPU_EXECUTION_THIS_TURN'}
    c['model']='cifar_resnet18_ssam2022';c['input']={'role':'standard_dense_sam','seed':0,'relative_path':item['relative_path'],'sha256':checkpoint_sha}
    c['precision']={'model':'float32','amp':False,'tf32':False,'compile':False}
    c['calibration'].update(approved=True,status='APPROVED_FROZEN',samples=32,index_file='configs/obc_smoke_train_ids.json',
        index_sha256=index_sha(ids),index_file_sha256=sha_file(output),batch_size=8,rounds=1,forward_passes_per_layer=4,
        optimization_epochs=0,random_seed=0,augmentation='none',
        preprocessing={'to_tensor':True,'normalize_mean':pc['normalize_mean'],'normalize_std':pc['normalize_std']})
    c['freeze_record']='configs/obc_smoke_freeze.json';c['freeze_record_sha256']=sha_file(proof)
    path.write_text(yaml.safe_dump(c,sort_keys=False),encoding='utf-8',newline='')
    print(json.dumps({'ids':ids,'index_sha256':index_sha(ids),'index_file_sha256':sha_file(output),'checkpoint_sha256':checkpoint_sha}))

if __name__=='__main__':main()
