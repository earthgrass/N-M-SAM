"""Package this authorized phase without modifying any earlier evidence."""
import ast,hashlib,json,shutil,sys,xml.etree.ElementTree as ET
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from benchmark.common import atomic_json,sha_file
from benchmark.preflight import source_identity
from benchmark.snows_patch import candidate_source
from benchmark.snows_step_patch import sources
from benchmark.upstream import source

def main():
    origin=ROOT/'artifacts/approved_smoke_stepfix_20261025'
    target=ROOT/'docs/research/evidence/approved_smoke_stepfix_20261025_final';target.mkdir(exist_ok=False)
    batches={};cases={}
    for name in ('snows_step_tests','snows_step_retry_tests','obc_freeze_tests','snows_actual_interface_tests'):
        tree=ET.parse(origin/(name+'.xml'));suites=tree.getroot().findall('testsuite')
        batches[name]={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
        for n in tree.findall('.//testcase'):
            cases[n.attrib['classname']+'::'+n.attrib['name']]='PASS' if n.find('failure') is None and n.find('error') is None else 'FAIL'
    if any(v!='PASS' for v in cases.values()):raise ValueError('A targeted test remains failing')
    for p in origin.iterdir():
        if p.is_file():shutil.copyfile(p,target/p.name)
    raw=[]
    for name in batches:
        tree=ET.parse(origin/(name+'.xml'))
        for n in tree.findall('.//testcase'):
            for failure in list(n.findall('failure'))+list(n.findall('error')):
                raw.extend([name+'::'+n.attrib['name'],failure.text or ''])
    (target/'RAW_JUNIT_TRACEBACKS.txt').write_text('\n\n'.join(raw),encoding='utf-8',newline='')
    proof=json.loads((ROOT/'configs/obc_smoke_freeze.json').read_text(encoding='utf-8'))
    before,_=candidate_source();after,caller,diff=sources()
    digest=lambda s:hashlib.sha256(s.encode()).hexdigest()
    report={'phase':'user-approved32-image smoke freeze and minimal SNOWS rejection repair',
        'snapshot_before_final_commit':True,'parent_commit':'4002ab362938d2d4fc24a46370d0164b24b9fb22',
        'source':source_identity(),'smoke_config_sha256':sha_file(ROOT/'configs/obc_cuda_smoke.yaml'),
        'index_file_sha256':proof['index_file_sha256'],'canonical_index_sha256':proof['index_canonical_int64le_sha256'],
        'checkpoint_sha256':proof['parent_checkpoint_sha256'],'dataset_manifest_sha256':proof['dataset_manifest_sha256'],
        'raw_dataset_validation':'historical checkpoint-bound data manifest verified offline; server file bytes must pass future runtime gate',
        'test_batches':batches,'latest_cases':cases,'unique_new_or_affected_cases':len(cases),
        'initial_harness_failures_preserved':7,'old29_plus18_evidence_reused_not_rerun':True,
        'patches':{'math_patch_unchanged_sha256':sha_file(ROOT/'patches/snows_minimal_diagnostic.patch'),
            'step_patch_sha256':sha_file(ROOT/'patches/snows_step_acceptance.patch'),
            'original_official_solver_sha256':digest(source('SNOWS','CNN_pruning/prune/SNOWS.py')),
            'after_old_math_patch_sha256':digest(before),'after_approved_step_patch_sha256':digest(after),
            'original_official_caller_sha256':digest(source('SNOWS','CNN_pruning/prune/Layer_pruner_new.py')),
            'controlled_caller_sha256':digest(caller)},
        'SNOWS':'BLOCKED_SOURCE: full representations/explicit Mask and native Armijo data mismatch remain unproved',
        'OBC':'approved offline preparation; final clean-commit readiness command must run after this source snapshot commits',
        'runtime_receipt':'artifacts/approved_smoke_stepfix_20261025/FINAL_READINESS.json',
        'GPU_executed':False,'SSH_executed':False,'formal_calibration_approved':False,'historical_artifacts_modified':False}
    atomic_json(target/'PHASE_EVIDENCE.json',report)
    atomic_json(target/'SHA256SUMS.json',{p.name:sha_file(p) for p in sorted(target.iterdir()) if p.is_file() and p.name!='SHA256SUMS.json'})
    lines=['# OBC approved smoke freeze —2026-10-25','',
        'Approved scope: CIFAR10 official train32IDs, seed0,batch8,noaug,1round, parent ToTensor/Normalize; exact CIFAR-ResNet18; partial layer1.0.conv1; FP32/no AMP/TF32/compile; independent BN copy;2fixed-Mask SGD steps. No GPU run authorized this turn.',
        '', '## Frozen sample IDs (draw order, never resampled)','', '```json',json.dumps(proof['ids']),'```','',
        f'Index JSON byte SHA256: `{report["index_file_sha256"]}`.',
        f'Canonical int64-LE index SHA256: `{report["canonical_index_sha256"]}`.',
        f'Parent checkpoint SHA256: `{report["checkpoint_sha256"]}`.',
        f'Checkpoint-bound full data-manifest SHA256: `{report["dataset_manifest_sha256"]}`.',
        f'Smoke config SHA256: `{report["smoke_config_sha256"]}`.',
        f'Source/config/scripts/patch digest: `{report["source"]["sha256"]}`.',
        '', 'Generation: existing deterministic_indices helper, NumPy default_rng(0)/PCG64 choice(50000,32,replace=False), draw order retained. The index file is loaded at execution and is never regenerated. Freeze helper refuses an existing production index file.',
        '', '## Parent source and preprocessing evidence','',
        'The actual seed0 Dense-SAM file passed byte SHA/schema/healthy/model digest checks. Its embedded source hashes matched SOURCE_FREEZE; original data.py/config.yaml bytes and embedded config digest matched. The actual Images class AST matches the runtime independent copy. Training=False executes only ToTensor then Normalize, no crop/flip/Cutout/TTA.',
        f'Normalize mean={proof["parent_normalization"]["normalize_mean"]}; std={proof["parent_normalization"]["normalize_std"]}.',
        'Five training files are ordered data_batch_1..5,10000rows each. Every selected ID maps to file/row and file SHA in configs/obc_smoke_freeze.json; no test_batch row is selected. Full dataset-file SHA map is retained there. Raw server files are not locally re-read while server is off; their checkpoint-bound identity is verified, and future runner validates actual full file manifest/order before pruning.',
        '', '## Execution acceptance','',
        'Source/config and index file bytes are included in the source digest. The new clean Git commit is obtained with git rev-parse HEAD after committing this report; the post-commit FINAL_READINESS receipt records that exact commit, source digest, config digest and fresh pilot checkpoint byte check. This avoids a self-referential commit hash inside its own content.',
        'New/affected OBC9tests pass:5freeze/preprocessing/tamper/exact optimizer checks and4affected smoke/gate checks. Prior compensation/N:M/BN/recovery proofs are reused. SNOWS9step tests plus2actual-interface checks pass after preserving7test-harness failures and targeted retries; no old full suite rerun.',
        'Evidence packet: docs/research/evidence/approved_smoke_stepfix_20261025_final. Earlier preparation snapshot remains separately preserved.',
        'Expected planning reservation:8GiB free VRAM/4GiB disk, approved RTX5090 UUID. No GPU peak/time is measured. CUDA smoke command and strict failure/no-overwrite/logging policy: post_training_nm_gpu_runbook.md. Formal OBC and SNOWS budgets remain PENDING_APPROVAL.']
    (ROOT/'docs/research/OBC_SMOKE_FREEZE.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'phase_cases':len(cases),'latest_cases':'PASS','initial_failures_preserved':7,
        'source_sha256':report['source']['sha256'],'config_sha256':report['smoke_config_sha256']}))

if __name__=='__main__':main()
