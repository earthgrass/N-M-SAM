"""Evidence-bound single-seed report; no optimizer or dataset execution."""
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from vpr_sam.l2_protocol import OUTPUT
from vpr_sam.repro import atomic_json, file_hash

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/vpr_l2'


def main():
    result = json.loads((OUT/'evaluation.json').read_text())
    if result['status'] != 'PASS': raise RuntimeError('Incomplete evaluation')
    run = ROOT/OUTPUT
    summary = json.loads((run/'summary.json').read_text())
    epochs = json.loads((run/'epochs.json').read_text())
    steps = json.loads((run/'steps.json').read_text())
    if len(epochs)!=200 or len(steps)!=70400: raise RuntimeError('Incomplete history')
    refs = ROOT/'docs/vpr_l2_reference'
    old = json.loads((refs/'v1_final_audit.json').read_text())
    rows=[]
    bn_old={r['method']:r for r in csv.DictReader((refs/'bn_recalibration.csv').open()) if r['seed']=='0'}
    for r in old['records']:
        if r['seed'] != 0: continue
        rows.append(dict(method=r['method'],seed=0,dense_loss=r['dense']['loss'],dense_accuracy=r['dense']['accuracy'],
            raw_hard_loss=r['hard']['loss'],raw_hard_accuracy=r['hard']['accuracy'],
            bn_hard_loss=float(bn_old[r['method']]['bn_post_loss']),bn_hard_accuracy=float(bn_old[r['method']]['bn_post_accuracy'])))
    dense=result['dense']['metrics'];hard=result['raw_hard']['hard_validation'];bn=result['bn_calibrated_hard']['metrics']
    rows.append(dict(method='fixed_l2',seed=0,dense_loss=dense['loss'],dense_accuracy=dense['accuracy'],
        raw_hard_loss=hard['loss'],raw_hard_accuracy=hard['accuracy'],bn_hard_loss=bn['loss'],bn_hard_accuracy=bn['accuracy']))
    for row in rows:
        row['raw_loss_increase']=row['raw_hard_loss']-row['dense_loss']
        row['raw_accuracy_drop']=row['dense_accuracy']-row['raw_hard_accuracy']
    with (OUT/'seed0_comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    plots=OUT/'plots';plots.mkdir(exist_ok=True)
    def plot(name,x,series,ylabel,xlabel='Epoch'):
        fig,ax=plt.subplots(figsize=(7,4))
        for label,values in series.items():ax.plot(x,values,label=label)
        ax.set(xlabel=xlabel,ylabel=ylabel);ax.legend();ax.grid(alpha=.2);fig.tight_layout()
        for suffix in ('png','pdf'):fig.savefig(plots/f'{name}.{suffix}',dpi=180)
        plt.close(fig)
    x=[r['epoch'] for r in epochs]
    plot('training_loss',x,{'center train':[r['train_loss'] for r in epochs],
        'adversarial train':[r['adversarial_train_loss'] for r in epochs],
        'validation':[r['validation']['loss'] for r in epochs]},'Cross entropy')
    plot('validation_accuracy',x,{'fixed_l2 final-protocol':[r['validation']['accuracy'] for r in epochs]},'Validation accuracy')
    sx=[r['step'] for r in steps]
    plot('epsilon_ratio',sx,{'epsilon L2/rho':[r['epsilon_l2_over_rho'] for r in steps]},'Ratio','Step')
    plot('log_radius',x,{k:[r['geometry'][k] for r in epochs] for k in ('log_radius_min','log_radius_max','log_radius_rms')},'Log radius')
    plot('log_volume',sx,{'global log-volume':[r['log_volume'] for r in steps]},'Sum log-radius','Step')
    measured=[r for r in steps if r.get('direction_cosine') is not None]
    plot('linear_direction',[r['step'] for r in measured],{k:[r[k] for r in measured] for k in ('direction_cosine','linear_strength_over_spherical')},'Ratio / cosine','Step')
    plot('validation_comparison',[0,1,2],{k:[r[k] for r in rows] for k in ('dense_accuracy','raw_hard_accuracy','bn_hard_accuracy')},'Validation accuracy','SAM / VPR / fixed_l2 (seed 0)')
    header=f"Frozen commit: `{result['commit']}`\n\nConfig SHA256: `{result['config_sha256']}`\n\n"
    def report(name,text):(ROOT/name).write_text('# '+name.removesuffix('.md').replace('_',' ')+'\n\n'+header+text+'\n',encoding='utf8')
    report('VPR_L2_TRAINING_REPORT.md',f"Completed exactly 200 epochs / 70400 steps from random initialization.\n\n"
        f"Dense validation: {dense}.\n\nElapsed: {summary['elapsed_seconds']} seconds; peak CUDA: {summary['peak_cuda_bytes']} bytes.\n\n"
        'Center/adversarial losses, gradients, epsilon, actual parameter delta and FP32 addition error are in steps.json. '
        'Direction/linear strength are measured on step 1+352k without extra forward/backward or RNG consumption. '
        'No early-accuracy stopping, warm start, recovery, extra seed or test-set evaluation.\n\n![](artifacts/vpr_l2/plots/training_loss.png)')
    ratios=np.array([r['epsilon_l2_over_rho'] for r in steps])
    endpoint=result['endpoint']
    means={key:float(np.mean([r[key] for r in endpoint])) for key in ('R_epsilon','direction_cosine','linear_actual_ratio','actual_loss_increase','spherical_loss_increase')}
    historical=[r for r in csv.DictReader((refs/'endpoint_probes.csv').open()) if r['seed']=='0']
    old_means={m:{k:float(np.mean([float(r[k]) for r in historical if r['method']==m])) for k in ('R_epsilon','cosine','linear_actual_ratio','ellipsoid_adversarial_loss','center_loss')} for m in ('sam','vpr')}
    for values in old_means.values():
        values['actual_loss_increase']=values['ellipsoid_adversarial_loss']-values['center_loss']
    atomic_json(OUT/'endpoint_comparison.json',{'fixed_l2':means,'v1':old_means,'rule':'same fixed 1024 validation IDs, eval BN, batch128; different trained endpoints'})
    report('VPR_L2_GEOMETRY_ANALYSIS.md',f"epsilon L2/rho range: [{ratios.min()}, {ratios.max()}].\n\nFinal geometry: {json.dumps(result['geometry'])}.\n\n"
        f"Fixed endpoint probes (1024 validation samples): {means}.\n\nV1 same-seed historical endpoint values: {old_means}.\n\n"
        'Radius normalization is a numerical property, not evidence of better generalization. '
        'Linear strength is g1 dot epsilon; its ratio to rho||g1|| equals cosine for fixed L2 and is at most 1 by Cauchy-Schwarz. '
        'True finite-perturbation loss increase is separately measured. Endpoint comparisons use different trained models; '
        'within each endpoint the spherical counterfactual shares that endpoint. No causal attribution from one seed. '
        'The ellipsoid boundary norm is allowed to differ from rho.\n\n![](artifacts/vpr_l2/plots/epsilon_ratio.png)')
    public={}
    for label,key in [('dense','dense'),('raw_hard','hard_probe')]:
        batches=result.get(key,{}).get('probe_batches',[])
        if batches:
            total=sum(b['count'] for b in batches)
            public[label]={k:sum(b[k]*b['count'] for b in batches)/total for k in ('center_loss','adversarial_loss','increase')}
            public[label]['count']=total
    atomic_json(OUT/'public_probe_summary.json',public)
    report('VPR_L2_PRUNING_EVALUATION.md',f"Final-only Dense: {dense}\n\nRaw Hard (original BN): {hard}\n\n"
        f"Raw loss increase: {result['raw_hard']['hard_loss_increase']}; accuracy drop: {result['raw_hard']['accuracy_drop']}.\n\n"
        f"Independent train-only BN-calibrated Hard: {bn}.\n\nPublic common sphere probes: {public}.\n\n"
        'Magnitude 2:4, exclude_first_last, dense tail. BN calibration resets buffers on a model copy, uses exactly the frozen 4096 train IDs, '
        '32 unaugmented batches of 128, BN-only train mode and momentum=None. Weights unchanged; no optimizer steps. '
        'The calibrated value does not replace the Raw Hard outcome. Public rho=.02 sphere probes use the frozen first 10 validation batches '
        'of 256; hard probes retain the masked subspace. Full per-batch values: evaluation.json.')
    comparison='| method | dense acc | dense loss | raw hard acc | raw hard loss | loss damage | acc drop | BN hard acc | BN hard loss |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
    for r in rows:comparison+='| '+r['method']+' | '+' | '.join(f'{r[k]:.9f}' for k in ('dense_accuracy','dense_loss','raw_hard_accuracy','raw_hard_loss','raw_loss_increase','raw_accuracy_drop','bn_hard_accuracy','bn_hard_loss'))+' |\n'
    baseline=next(r for r in rows if r['method']=='vpr');current=rows[-1]
    verdict={k:current[k]-baseline[k] for k in ('dense_accuracy','dense_loss','raw_loss_increase','raw_accuracy_drop')}
    report('VPR_L2_SEED0_COMPARISON.md',comparison+'\nFixed_l2 minus V1 VPR: '+json.dumps(verdict)+'\n\n'
        'All rows use seed0, matching initialization/split/order and validation definitions; source snapshots differ and are explicitly recorded. '
        'Historical values come from audited immutable JSON/CSV, never loaded under V2 provenance. '
        'Positive accuracy difference means higher accuracy; positive loss/damage difference means higher loss/damage. '
        'This is one exploratory seed, with no multi-seed significance claim. Numerical L2 correction, first-order strength, '
        'finite loss increase, dense performance and pruning damage must be assessed separately.\n\n![](artifacts/vpr_l2/plots/validation_comparison.png)')
    hashes={str(p.relative_to(ROOT)):file_hash(p) for p in (run/'final.pt',run/'best_validation.pt',run/'last.pt')}
    hard_path=run/'pruning/n2_m4_exclude_first_last_magnitude_dense_final/pruned.pt'
    hashes[str(hard_path.relative_to(ROOT))]=file_hash(hard_path)
    if hashes[str((run/'final.pt').relative_to(ROOT))] != result['checkpoint_sha256']['final']:
        raise RuntimeError('Final SHA changed during report')
    audit=dict(status='PASS',commit=result['commit'],config_sha256=result['config_sha256'],checkpoint_sha256=hashes,
        completed_epochs=200,completed_steps=70400,paired_identity=True,source_schema=2,
        raw_hard_bn_unchanged=True,bn_calibration_weights_unchanged=True,
        recovery_executed=False,test_used=False,extra_seed_executed=False,
        limits='single exploratory seed; no significance or guaranteed performance advantage')
    atomic_json(OUT/'final_audit.json',audit)
    report('VPR_L2_FINAL_AUDIT.md','PASS. Exactly one authorized fixed_l2 seed0 200-epoch trajectory and approved non-training evaluations completed.\n\n'
        +json.dumps(audit,indent=2)+'\n\nAll primary results use final checkpoint. No recovery or additional GPU task is queued. '
        'Large checkpoints and datasets stay on the server.\n')
    files=[p for p in OUT.rglob('*') if p.is_file() and p.name!='DELIVERY_SHA256.json']
    files += list(ROOT.glob('VPR_L2_*.md'))
    atomic_json(OUT/'DELIVERY_SHA256.json',{str(p.relative_to(ROOT)):file_hash(p) for p in files})
    print('VPR-L2 final reports and figures complete',flush=True)


if __name__ == '__main__':main()
