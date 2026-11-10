"""Read-only source snapshot and deterministic reference/smoke tables. No Test access."""
from pathlib import Path
import hashlib, json, shutil, subprocess

ROOT = Path(__file__).resolve().parents[1]
HQ = ROOT.parent / 'HQ-APBR-V1'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p, value): p.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
def load(p): return json.loads(p.read_text(encoding='utf-8-sig'))

def main():
    selected = {
        'config.json': 'configs/hq_apbr_v1_seed0_gpu_v1.json',
        'gpu_smoke_summary.json': 'reports/gpu_3090_v1/SUMMARY.json',
        'smoke_history.json': 'reports/gpu_3090_v1/remote_run_v1/run_v1/smoke/history.json',
        'historical_references.json': 'reports/gpu_3090_v1/reference_results.json',
        'initialization.json': 'reports/gpu_3090_v1/formal_start/INITIALIZATION_PROOF.json',
        'formal_identity.json': 'reports/gpu_3090_v1/formal_start/identity.json',
        'execution_approval.json': 'approvals/seed0_performance_exception_20261105.json',
        'math_definition.md': 'docs/hq_apbr_v1_math.md',
    }
    records=[]
    for name, rel in selected.items():
        src=HQ/rel; dest=ROOT/'evidence'/name
        if dest.exists() and sha(dest)!=sha(src):
            raise RuntimeError(f'Refusing to overwrite frozen evidence: {name}')
        shutil.copyfile(src,dest)
        records.append({'snapshot':str(dest.relative_to(ROOT)), 'source':str(src), 'sha256':sha(src)})
    # Source is captured as hashes; do not copy or mutate the training implementation.
    for rel in ['hq_apbr_v1/math_core.py','hq_apbr_v1/training.py','hq_apbr_v1/calibration.py',
                'hq_apbr_v1/evaluation.py','scripts/run_authorized_seed0.py']:
        records.append({'source':str(HQ/rel),'sha256':sha(HQ/rel)})
    dump(ROOT/'evidence/source_snapshot_manifest.json',records)
    git={k:subprocess.check_output(['git','-C',str(HQ),*v],text=True,encoding='utf-8').strip()
         for k,v in {'head':['rev-parse','HEAD'],'branch':['branch','--show-current'],'status':['status','--short']}.items()}
    dump(ROOT/'evidence/repository_snapshot.json',git)
    ref=load(ROOT/'evidence/historical_references.json')
    rs=ref['references']; sam, replay, apbr=rs
    lines=[r'\begin{table*}[t]',r'\centering\small',r'\begin{tabular}{llrrrrr}',r'\toprule',
           r'Training & Evidence cohort & Dense Acc. & Dense CE & Native Acc. & Native CE & Drop (pp)\\',r'\midrule']
    for label,r in [('SAM',sam),('Original APBR',apbr)]:
        lines.append(f"{label} & Historical reference & {r['dense_accuracy_percent']:.2f} & {r['dense_ce']:.4f} & {r['accuracy_percent']:.2f} & {r['ce']:.4f} & {-r['accuracy_delta_vs_dense_pp']:.2f}"+r'\\')
    lines += [r'\midrule',r'Direct-Q SAM & Not run & \pending & \pending & \pending & \pending & \pending\\',
              r'\HCResultRow',r'\bottomrule',r'\end{tabular}',
              r'\caption{Dense and native SparseGPT endpoints (accuracy in \%). Existing reference records are not a matched HC-SAM cohort. The native endpoint includes dynamic selection and OBS compensation, with zero recovery epochs. Historical SAM replay is listed separately in the appendix.}',
              r'\label{tab:main}',r'\end{table*}']
    (ROOT/'tables/main_results.tex').write_text(r'\newcommand{\MainResultsTable}{'+'\n'+'\n'.join(lines)+'\n}\n',encoding='utf-8')
    pending=ROOT/'tables/hc_result_row.tex'
    if not pending.exists(): pending.write_text(r'\newcommand{\HCResultRow}{HC-SAM & Current exploratory run & \pending & \pending & \pending & \pending & \pending\\}'+'\n',encoding='utf-8')
    hist=[r'\begin{table}[h]',r'\centering\small',r'\begin{tabular}{lrrr}',r'\toprule Execution & Dense & Native & Native CE\\',r'\midrule']
    for label,r in [('SAM formal',sam),('SAM paired replay',replay),('Original APBR',apbr)]:
        hist.append(f"{label} & {r['dense_accuracy_percent']:.2f} & {r['accuracy_percent']:.2f} & {r['ce']:.4f}"+r'\\')
    hist += [r'\bottomrule\end{tabular}',r'\caption{Separate historical single-seed records, accuracy in \%. No averaging of duplicate executions and no selection of the better SAM reference.}\label{tab:historical}',r'\end{table}']
    (ROOT/'tables/historical.tex').write_text(r'\newcommand{\HistoricalTable}{'+'\n'+'\n'.join(hist)+'\n}\n',encoding='utf-8')
    smoke=load(ROOT/'evidence/smoke_history.json')
    summary=load(ROOT/'evidence/gpu_smoke_summary.json')
    train=sum(x['train_seconds'] for x in smoke)/len(smoke)
    cal=sum(x['calibration_duration'] for x in smoke)/len(smoke)
    # Summary contains peak across gate and smoke, not only the three training epochs.
    profile={'train_seconds_mean':train,'calibration_seconds_mean':cal,'calibration_to_train_percent':100*cal/train,
             'peak_allocated_gib':summary['resources']['peak_allocated_bytes']/1024**3, 'peak_reserved_gib':summary['resources']['peak_reserved_bytes']/1024**3, 'resources_source':summary['resources'],'performance_source':summary['performance'],
             'scope':'three temporary smoke epochs; full training pending; not SAM speed comparison'}
    dump(ROOT/'evidence/performance_derived.json',profile)
    cost=r'''\begin{table}[t]
\centering\small
\begin{tabular}{lr}
\toprule Measurement (three-epoch smoke) & Value\\
\midrule
Training per epoch & TRAIN s\\
Calibration per epoch & CAL s\\
Calibration / training & RATIO\%\\
Peak allocated (gate + smoke) & ALLOC GiB\\
Peak reserved (gate + smoke) & RESERVE GiB\\
Numerical / resume checks & Pass\\
30\% performance gate & Fail\\
Full-run cost & \pending\\
\bottomrule
\end{tabular}
\caption{Synchronized RTX 3090 measurements. Checkpoint I/O and final evaluation are outside the training/calibration timers. An explicit research exception permits training without changing the failed performance gate.}
\label{tab:cost}
\end{table}
'''.replace('TRAIN',f'{train:.2f}').replace('CAL',f'{cal:.2f}').replace('RATIO',f'{100*cal/train:.2f}').replace('ALLOC',f"{profile['peak_allocated_gib']:.3f}").replace('RESERVE',f"{profile['peak_reserved_gib']:.3f}")
    (ROOT/'tables/performance.tex').write_text(r'\newcommand{\PerformanceTable}{'+'\n'+cost+'}\n',encoding='utf-8')
    entries=[]
    def entry(value, source, pointer, note=''):
        if isinstance(value,(int,float)) and not isinstance(value,bool):
            entries.append({'value':value,'source':source,'selector':{'kind':'json-pointer','pointer':pointer},'aggregate':'identity','tolerance':1e-8,'note':note})
    for i,r in enumerate(rs):
        for k in ['accuracy_percent','ce','dense_accuracy_percent','dense_ce','accuracy_delta_vs_dense_pp','test_samples','calibration_count']:
            entry(r[k],'evidence/historical_references.json',f'/references/{i}/{k}','Historical only')
    for k,v in load(ROOT/'evidence/config.json').items(): entry(v,'evidence/config.json','/'+k,'Frozen protocol')
    for k in ['train_seconds_mean','calibration_seconds_mean','calibration_to_train_percent','peak_allocated_gib','peak_reserved_gib']:
        entry(profile[k],'evidence/performance_derived.json','/'+k,'Recomputed from smoke_history.json by this script')
    entry(78200,'evidence/historical_references.json','/parents/dense_sam/steps','200 * 391 frozen step budget, also verified in reference')
    entry(5578752,'evidence/historical_references.json','/references/0/mask_audit/deleted_coordinates','Frozen eligible half-density count')
    entry(4,'evidence/historical_references.json','/references/1/algorithm/m','Physical group size')
    dump(ROOT/'numeric_evidence.json',{'schema_version':'ai-research-writing/numeric-evidence-v2','entries':entries})
    print('Evidence frozen; reference and smoke tables generated; no evaluation executed.')

if __name__=='__main__': main()
