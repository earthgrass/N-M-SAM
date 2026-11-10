"""One-time preparation from verified local evidence; not an experiment launcher."""
import json,shutil,sys
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from benchmark.common import atomic_json,sha_file
from benchmark.calibration import index_sha

def main():
    inventory=json.loads((ROOT/'artifacts/stage1_inventory.json').read_text(encoding='utf-8'))
    old=ROOT.parent/'vpr_v2_unified50k'
    target=ROOT/'configs';target.mkdir(exist_ok=True)
    shutil.copyfile(old/'config.yaml',target/'historical_protocol.yaml')
    shutil.copyfile(old/'reference/bn_train_ids.json',target/'bn_train_ids.json')
    original=yaml.safe_load((target/'historical_protocol.yaml').read_text(encoding='utf-8'))
    ids=json.loads((target/'bn_train_ids.json').read_text(encoding='utf-8'))
    inputs=[]
    for row in inventory['checkpoint_records']:
        inputs.append({k:v for k,v in row.items() if k!='path'})
        inputs[-1]['relative_path']=str(Path(row['path']).relative_to(Path(row['path']).parents[3])).replace('\\','/')
    # Paths are relative to delivery/best_checkpoints; runtime location is an explicit CLI argument.
    atomic_json(target/'existing_inputs.json',{'status':'historically_verified','checkpoints':inputs})
    config={'name':'APBR-post-training-NM-controlled-benchmark','version':'preexecution-acceptance-20261025','purpose':'formal','device':'cuda:0',
        'gpu_uuid':original['gpu_uuid'],'threads':4,'seeds':[0,1,2],
        'execution':{'approved':False,'cuda_smoke_receipt':None,'pilot_seed':0,'full_queue_requires_reviewed_pilot':True},
        'dense_training':'prohibited; load existing Dense-SAM only for new competitors',
        'apbr_alias':'Full VPR-L2 + Geometry-guided; fixed_l2/vpr_guided unchanged',
        'calibration':{'approved':False,'dataset':'official_cifar10_train_50000',
                       'samples':None,'index_file':None,'index_sha256':None,'batch_size':None,
                       'rounds':None,'forward_passes_per_layer':None,'optimization_epochs':None,'random_seed':None,
                       'augmentation':None,'preprocessing':None,'status':'PENDING_APPROVAL',
                       'method_native':{'OBC':{'samples':1024,'default_rounds':10,'noaug_rounds':1,'approved_for_local_run':False},
                                        'SNOWS':{'nm_parser_ngrads':128,'readme_example_ngrads':3000,'approved_for_local_run':False}},
                       'common_budget':{'status':'PENDING_APPROVAL','approved':False}},
        'nm_acceptance':{'OBC':'CPU_ADAPTER_PASS_CUDA_PENDING','SNOWS':'BLOCKED','CAP':'CONDITIONAL_BLOCKED'},
        'pruning':{'n':2,'m':4,'layout':'oihw_output_row_contiguous_v1','scope':'exclude_first_last',
                   'tail':'dense','bn_affine':'excluded','tie_break':'native OBC argmin first index; existing magnitude stable lower index'},
        'data':{k:original[k] for k in ('train_size','test_size','validation_size','batch_size','evaluation_batch_size',
             'workers','pin_memory','drop_last','normalize_mean','normalize_std','crop_padding','horizontal_flip_probability',
             'cutout_size','cutout_probability')},
        'recovery':original['recovery'],
        'common_bn':{'samples':len(ids),'index_file':'configs/bn_train_ids.json','index_sha256':index_sha(ids),
                     'augmentation':'none','batch_size':128,'rounds':1,'momentum':None,'separate_copy':True},
        'OBC':{'status':'CPU_ADAPTER_TESTED_GPU_PENDING','parallel':32,'relative_damping':0.,
               'native_bn':{'default_enabled':False,'optional_postproc_bnt_batches':100,'momentum':.1},
               'output':'native compensated post-pruning; not Raw Hard'},
        'SNOWS':{'status':'BLOCKED_PINNED_SOURCE','reasons':['module syntax error','chunked HVP error','zero-RHS CG nonfinite']},
        'CAP':{'status':'BLOCKED_PINNED_NM_CONSTRUCTORS'},
        'MOONSHOT':{'status':'LITERATURE_ONLY_CODE_NOT_RELEASED_AT_PIN'},
        'UniPTS':{'status':'LITERATURE_ONLY_NO_EQUIVALENT_NM_PATH'},
        'minimum_free_disk_bytes':22*1024**3,
        'precision':{'model':'float32','amp':False,'tf32':False,'compile':False,'deterministic':True},
        'selection':'maximum Test Accuracy earliest tie; Test reused for selection; biased endpoint disclosed',
        'recovery_parent':'native_complete before any BN recalibration',
        'provenance':{'historical_protocol_sha256':sha_file(target/'historical_protocol.yaml'),
                      'base_commit':'d0025b48664091444cc1ad363253010c336b23e6'}}
    (target/'post_training_nm_benchmark.yaml').write_text(yaml.safe_dump(config,sort_keys=False,allow_unicode=True),encoding='utf-8')
    print('Prepared configuration; calibration and GPU execution remain unapproved.')

if __name__=='__main__':main()
