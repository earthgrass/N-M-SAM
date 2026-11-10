"""Seal supplemental read-only acceptance evidence; no experiment execution."""
import json,shutil,subprocess,sys,xml.etree.ElementTree as ET
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from benchmark.common import atomic_json,sha_file
from benchmark.preflight import source_identity,gates

def main():
    evidence=ROOT/'docs/research/evidence';cases=set();batches={}
    for name in ('cpu_tests','acceptance_tests','final_gate_tests','cuda_closed_gate_tests'):
        path=ROOT/'artifacts'/(name+'.xml');tree=ET.parse(path)
        suites=tree.getroot().findall('testsuite');counts={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
        if counts['failures'] or counts['errors']:raise ValueError('Acceptance evidence contains failure: '+name)
        for item in tree.findall('.//testcase'):cases.add((item.attrib.get('classname'),item.attrib['name']))
        batches[name]=counts
        shutil.copyfile(path,evidence/path.name)
        environment=path.with_suffix('.environment.json')
        if environment.exists():shutil.copyfile(environment,evidence/environment.name)
    shutil.copyfile(ROOT/'artifacts/preexecution_acceptance/tracebacks.json',evidence/'acceptance_tracebacks.json')
    c=yaml.safe_load((ROOT/'configs/post_training_nm_benchmark.yaml').read_text(encoding='utf-8'))
    inventory=json.loads((ROOT/'configs/existing_inputs.json').read_text(encoding='utf-8'))
    report={'status':'CPU_PREPARATION_ACCEPTED_FORMAL_EXECUTION_BLOCKED','snapshot_before_commit':True,
        'branch':subprocess.check_output(['git','-C',str(ROOT),'branch','--show-current'],text=True).strip(),
        'parent_HEAD':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        'source':source_identity(),'config_sha256':sha_file(ROOT/'configs/post_training_nm_benchmark.yaml'),
        'test_batches':batches,'distinct_test_cases':len(cases),'initial_suite_not_repeated_for_new_request':True,
        'tests':{p.name:sha_file(p) for p in sorted((ROOT/'tests').glob('*.py'))},
        'calibration':c['calibration'],'startup_failures':gates(c),
        'gpu_environment_checked':False,'cuda_smoke':'PENDING','formal_pruning':'NOT_RUN',
        'recovery_queue':'NOT_RUN','historical_artifacts_modified':False,
        'original_checkpoint_audit':'15 files previously verified; no redundant re-load requested',
        'SNOWS':'BLOCKED_AT_PIN; whole import not operational; numeric witness preserved',
        'CAP':'CONDITIONAL_BLOCKED; full SparseML stack not validated',
        'resume_boundary':'epoch end only; no mid-batch sampler state; next epoch order derived from restored RNG and same loader',
        'limits':['CPU synthetic determinism only, not all real-data worker/CUDA behavior',
                  'GPU resource monitor integration and full-size acceptance pending',
                  'No official source patch applied; no claim of general paper algorithm defect']}
    atomic_json(evidence/'acceptance_state.json',report)
    hashes={p.name:sha_file(p) for p in sorted(evidence.iterdir()) if p.is_file() and p.name!='SHA256SUMS.json'}
    atomic_json(evidence/'SHA256SUMS.json',hashes)
    lines=['# Pre-execution audit and acceptance — 2026-10-25','',
        'CPU preparation accepted within the stated synthetic/inspection scope. **Formal execution remains BLOCKED.**',
        'Server not contacted; no formal pruning, recovery queue or dense retraining. Historical files untouched.','',
        '## Actual state inspected','',
        f'Branch `{report["branch"]}`, parent HEAD `{report["parent_HEAD"]}`. Preparation changes were staged/uncommitted when inspected; this report captures pre-commit source hashes, not a formal-run freeze.',
        'Read existing upstream lock/config, OBC adapter/pipeline, common BN/evaluation/recovery and their tests, initial15-case JUnit and runtime metadata, all15 checkpoint inventory/geometry/mask records, copied source protocol/BN IDs, historical per-seed CSV/statistics, original Mask/postprocess call chain, and pinned native pruning/scorer code.',
        'No redundant full checkpoint reload or rerun of the historical results was needed.','',
        '## Tests and evidence','',
        f'Prior full suite:15PASS. This request added14 distinct cases and reran only affected paths: targeted18PASS, final source/group checks2PASS, closed-CUDA-gate1PASS. Combined evidence covers{len(cases)}distinct tests; repeated assertions are not independent experiments.',
        'Raw JUnit files, exact test runtime, full tracebacks and hashes are in evidence/. Earlier tests were inspected first; the old numerical HVP/CG counterexample was not rerun merely to recreate its report.',
        'New coverage: native permutation/inverse/quadratic correspondence, differing feasible Mask, noncontiguous3x3 weights, explicit native ties, exact BN cumulative mean/unbiased variance/counter and unrelated buffers, every SGD-step fixed Mask/zero weight/zero momentum, kept-weight update, every epoch reload, earliest accuracy tie, Python/NumPy/CPU RNG restoration, and independent missing calibration/config/grouping/CUDA gates.','',
        '## OBC and grouping acceptance','',
        '[Full source-chain comparison table and proof](NM_GROUPING_ACCEPTANCE.md). Both exact-N legality and conditional quadratic compensation passed CPU tests. Original Conv Unfold H goes through native Linear identity coordinates; original score/compensation/inverse code is unchanged. This intentionally adapts the grouping constraint and is not a strict reproduction of published native-convention numbers.',
        'Native OBC removes earliest tied score; historical Magnitude/Guided keeps earliest tied score. This difference is explicit, not silently normalized. GPU memory, full-size numerical behavior and performance are unverified.','',
        '## Candidate blockers','',
        'SNOWS pin750497fe: parse failure before dependency/model loading; isolated unchanged function tests show default chunk HVP error1 on an analytic quadratic and zero-RHS CG nonfinite. CAP pinf4cbe696: actual N:M scorer AST call reproduces unexpected n keyword; direct CAPNMHandle uses self.m before assignment. Complete tracebacks, inputs, commits and actual local versions are in acceptance_tracebacks.json.',
        'These are locked-source/integration blockers, not claims that the published algorithms are invalid. An independent model bridge cannot fix parsing or constructors. CAP remains conditional; full SparseML execution was not validated. No patch was applied. Official corrected revision or separately reviewed controlled patch is a future decision.','',
        '## BN / recovery acceptance and limits','',
        'BN-only copies are independent. Only running_mean/running_var/num_batches_tracked change; original model, weights and unrelated sentinel buffer stay equal. Eval changes no model state.',
        'Synthetic interruption/restoration matches model parameters and buffers, optimizer momentum, scheduler/LR, Python/NumPy/CPU RNG, sample-order hashes, epoch/global steps, Best metadata and Mask/config identity. Every SGD step preserves pruned zeroes and fixed Mask; all epoch files load. Best=maximum Test Accuracy earliest tie; Last=latest healthy epoch, Final=epoch20 when complete.',
        'Resume is at epoch boundaries: batch within next epoch is0, not an arbitrary mid-batch continuation. Sample order derives from restored RNG and the same RandomSampler/loader. Actual multiworker/CUDA continuation remains pending; do not describe this as universal whole-training determinism.','',
        '## Calibration and execution gates','',
        'Status=PENDING_APPROVAL. Calibration dataset is explicitly train-only. Sample count/IDs/hash/batch/forward passes/rounds/optimization epochs/seed/augmentation/preprocessing remain null. Method-native1024x10 OBC and SNOWS128 parser/3000 example are informational, not approved choices. Common BN4096 remains separate.',
        'Missing calibration fields, unaccepted N:M layout/scope, wrong input or reviewed config SHA, wrong frozen Git commit, missing CUDA receipt, unavailable/wrong/busy GPU and insufficient disk reject execution. execution.approved=false, cuda_smoke_receipt=null. No formal task starts this round even if future prerequisites become available.','',
        '## Future commands and pending work','',
        '[Single-seed pilot, serial seeds1/2, exact resume commands, inputs/SHA256, logs/outputs, estimates and failure policy](post_training_nm_gpu_runbook.md). Config is configs/post_training_nm_benchmark.yaml.',
        f'Current **unapproved** config SHA256: `{report["config_sha256"]}`.',
        'Commands require a later reviewed configuration hash and frozen commit; current placeholders are intentional because calibration is not approved. GPU smoke receipt must bind source/config and pass before formal pilot. Resource monitoring integration, full-size pilot, and any SNOWS/CAP fixes remain unimplemented/unaccepted GPU work. No three-seed GPU queue is authorized or running.']
    (ROOT/'docs/research/PRE_EXECUTION_ACCEPTANCE.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'status':report['status'],'distinct_test_cases':len(cases),'batches':batches,'config_sha256':report['config_sha256']}))

if __name__=='__main__':main()
