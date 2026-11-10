"""Offline method-specific status. Never contacts a GPU server or launches jobs."""
import argparse,json,subprocess
from pathlib import Path
import yaml
from .common import atomic_json,sha_file
from .obc_smoke import review_errors
from .preflight import ROOT,source_identity

def statuses(formal,smoke,config_frozen=False,input_verified=False):
    errors=review_errors(smoke,formal)
    if not config_frozen:errors.append('BLOCKED_CONFIG_REVIEW: current source/config not reviewed and frozen')
    if not input_verified:errors.append('BLOCKED_CONFIG_REVIEW: current pilot input hash not freshly verified')
    status='READY_FOR_GPU_SMOKE' if not errors else ('BLOCKED_CALIBRATION_APPROVAL' if any('CALIBRATION' in e for e in errors) else 'BLOCKED_CONFIG_REVIEW')
    return {'OBC':{'status':status,'reasons':errors,'cuda_acceptance':'NOT_RUN'},
        'SNOWS':{'status':'BLOCKED_SOURCE','reasons':['controlled kernels are not the complete solver',
            'minimal rejected-step fix is controlled; Armijo batch/gradient mismatch unchanged','CIFAR18 hook bridge and explicit-zero Mask unverified',
            'calibration/optimization budget unapproved'],'cuda_acceptance':'NOT_RUN'}}

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='artifacts/smoke_readiness_status.json')
    p.add_argument('--verify-input-root');p.add_argument('--expected-commit');p.add_argument('--expected-config-sha256');args=p.parse_args()
    c=yaml.safe_load((ROOT/'configs/post_training_nm_benchmark.yaml').read_text(encoding='utf-8'))
    smoke_path=ROOT/'configs/obc_cuda_smoke.yaml';smoke=yaml.safe_load(smoke_path.read_text(encoding='utf-8'))
    current_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    clean=not subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip()
    frozen=clean and args.expected_commit==current_commit and args.expected_config_sha256==sha_file(smoke_path)
    verified=False
    if args.verify_input_root:
        records=json.loads((ROOT/'configs/existing_inputs.json').read_text(encoding='utf-8'))['checkpoints']
        item=next(r for r in records if r['role']=='standard_dense_sam' and r['seed']==0)
        verified=sha_file(Path(args.verify_input_root)/item['relative_path'])==item['sha256']
        if not verified:raise ValueError('Pilot input SHA256 mismatch')
    report={'methods':statuses(c,smoke,frozen,verified),'source':source_identity(),
        'smoke_config_sha256':sha_file(smoke_path),'formal_config_sha256':sha_file(ROOT/'configs/post_training_nm_benchmark.yaml'),
        'snapshot_commit':current_commit,'source_tree_clean':clean,'input_hash_rechecked_this_command':verified,
        'calibration_index_file_sha256':smoke['calibration'].get('index_file_sha256'),
        'calibration_index_canonical_sha256':smoke['calibration'].get('index_sha256'),
        'freeze_record_sha256':smoke.get('freeze_record_sha256'),
        'data_identity_scope':'checkpoint-bound historical file manifest/preprocessing verified; actual server bytes checked by future smoke',
        'prior_inputs':'existing original15-file verified inventory retained; no redundant full audit',
        'server_contacted':False,'gpu_started':False}
    atomic_json(args.output,report);print(json.dumps(report['methods']))

if __name__=='__main__':main()
