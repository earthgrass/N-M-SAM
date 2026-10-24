"""Read existing three-seed metrics, hashes and audits; no model execution."""
import csv,io,json,math,statistics
from pathlib import Path
from overnight import ROOT,HOME,SEED0,save,sha,verify_bundle

METHODS=('sam_magnitude','vpr_magnitude','vpr_guided')

def mean_sd_ci(values):
    if len(values)!=3 or any(not math.isfinite(v) for v in values):raise ValueError('Three finite independent seeds required')
    mean=statistics.mean(values);sd=statistics.stdev(values);half=4.302652729911275*sd/math.sqrt(3)
    return {'n_seeds':3,'mean':mean,'sample_sd':sd,'exploratory_95pct_t_interval_df2':[mean-half,mean+half]}

def read_seed(seed):
    project=SEED0 if seed==0 else HOME/'runs'/f'vpr_seed{seed}'/'project'
    comparison=project/'artifacts/comparison'
    audit=json.loads((comparison/'FINAL_AUDIT.json').read_text())
    if audit['status']!='PASS':raise ValueError('Comparison audit failed')
    data=json.loads((comparison/'summary.json').read_text())
    if {r['method'] for r in data}!=set(METHODS):raise ValueError('Methods incomplete')
    for row in data:
        r=row['recovery']['epochs']
        if len(r)!=20 or r[-1]['steps']!=7800:raise ValueError('Recovery budget')
        best=max(r,key=lambda e:(e['test_accuracy'],-e['epoch']))
        if best['epoch']!=row['recovery_best']['epoch'] or best['test_accuracy']!=row['recovery_best']['test_accuracy']:
            raise ValueError('Best Test rule mismatch')
        if sha(row['parent']['path'])!=row['parent']['sha256']:raise ValueError('Dense checkpoint changed')
    by={r['method']:r for r in data}
    if by['vpr_magnitude']['parent']!=by['vpr_guided']['parent']:raise ValueError('VPR weights/geometry mismatch')
    orders=[[e['order_sha256'] for e in r['recovery']['epochs']] for r in data]
    if orders[0]!=orders[1] or orders[0]!=orders[2]:raise ValueError('Recovery pairing')
    return data,{'seed':seed,'summary_path':str(comparison/'summary.json'),
                 'summary_sha256':sha(comparison/'summary.json'),'audit_sha256':sha(comparison/'FINAL_AUDIT.json'),
                 'checkpoint_sha256':audit['checkpoint_sha256']}

def main():
    freeze=verify_bundle();rows=[];evidence=[];curves={}
    for seed in (0,1,2):
        data,receipt=read_seed(seed);evidence.append(receipt)
        for r in data:
            b,f=r['recovery_best'],r['recovery_final'];curves[seed,r['method']]=r['recovery']['epochs']
            dense_root=Path(r['parent']['path']).parent
            dense_history=[json.loads(s) for s in (dense_root/'epoch_metrics.jsonl').read_text().splitlines()]
            producer=json.loads((dense_root/'source_manifest.json').read_text())['identity']
            if len(dense_history)!=200 or dense_history[-1]['steps']!=78000 or producer['seed']!=seed:
                raise ValueError('Dense source or budget mismatch')
            rows.append({'seed':seed,'method':r['method'],'dense_accuracy_pct':100*r['dense']['accuracy'],
                'dense_loss':r['dense']['loss'],'raw_accuracy_pct':100*r['raw_hard']['accuracy'],'raw_loss':r['raw_hard']['loss'],
                'bn_accuracy_pct':100*r['bn_only']['accuracy'],'bn_loss':r['bn_only']['loss'],
                'recovery_best_accuracy_pct':100*b['test_accuracy'],'recovery_best_epoch':b['epoch'],
                'recovery_best_loss':b['test_loss'],'recovery_final_accuracy_pct':100*f['test_accuracy'],
                'recovery_final_loss':f['test_loss'],'raw_damage_pp':r['accuracy_damage_pp'],
                'recovery_best_gain_pp':r['recovery_best_gain_pp'],'parent_sha256':r['parent']['sha256'],
                'mask_sha256':r['mask_sha256'],'recovery_best_sha256':r['recovery']['best_sha256'],
                'recovery_final_sha256':r['recovery']['last_sha256'],
                'dense_commit':producer['commit'],'dense_config_sha256':producer['config_sha256'],
                'dense_config_file_sha256':producer['config_file_sha256'],
                'dense_elapsed_before_final_publication_seconds':dense_history[-1]['elapsed_seconds_before_publication'],
                'dense_peak_cuda_allocated_bytes':dense_history[-1]['peak_cuda_allocated_bytes'],
                'recovery_elapsed_seconds':f['elapsed_seconds'],
                'recovery_peak_cuda_allocated_bytes':f['peak_cuda_allocated_bytes']})
    out=ROOT/'artifacts/final_seeds012';out.mkdir(parents=True,exist_ok=False)
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (out/'per_seed_metrics.csv').write_text(stream.getvalue())
    metrics=['dense_accuracy_pct','raw_accuracy_pct','raw_loss','bn_accuracy_pct','bn_loss',
             'recovery_best_accuracy_pct','recovery_best_loss','recovery_final_accuracy_pct','recovery_final_loss','raw_damage_pp','recovery_best_gain_pp']
    stats={m:{k:mean_sd_ci([r[k] for r in rows if r['method']==m]) for k in metrics} for m in METHODS}
    paired={}
    indexed={(r['seed'],r['method']):r for r in rows}
    for a,b in [('vpr_magnitude','sam_magnitude'),('vpr_guided','sam_magnitude'),('vpr_guided','vpr_magnitude')]:
        paired[a+' minus '+b]={k:mean_sd_ci([indexed[s,a][k]-indexed[s,b][k] for s in (0,1,2)]) for k in metrics}
    save(out/'statistics.json',{'method_statistics':stats,'within_seed_paired_differences':paired,
        'unit':'independent_seed','selection_bias':'Test used for checkpoint selection; not independent unbiased evaluation',
        'uncertainty':'Exploratory paired t intervals assume independent approximately normal seed differences; n=3 is small'})
    save(out/'evidence.json',{'aggregation_commit':freeze['commit'],'seeds':evidence,'seed0_training_commit':'cea8f8c051221929e6572f431adb4fd15f727551'})
    lines=['# Paired seeds0/1/2 SAM and VPR-L2 report','','All endpoints use official Test; Best Test participates in selection and is not an independent unbiased estimate.',
           'Seed0 keeps its original frozen snapshot; seeds1/2 use the pre-frozen extension with identical research definitions.','',
           '| Method | Raw Acc mean±SD | BN-only Acc mean±SD | Recovery Best Acc mean±SD | Recovery Final Acc mean±SD |',
           '|---|---|---|---|---|']
    for method in METHODS:
        entries=[f"{stats[method][k]['mean']:.4f} ± {stats[method][k]['sample_sd']:.4f}" for k in
                 ['raw_accuracy_pct','bn_accuracy_pct','recovery_best_accuracy_pct','recovery_final_accuracy_pct']]
        lines.append('| '+method+' | '+' | '.join(entries)+' |')
    lines+=['','See per_seed_metrics.csv for every Accuracy/Loss, Best Epoch, checkpoint and Mask hash.',
            'statistics.json retains paired differences and exploratory95% t intervals, df2. Epochs and batches are not statistical repeats.',
            'BN-only is separate from untouched-Raw recovery. Dense Best weights and geometry are matched. No Test-independent or SOTA claim.']
    (out/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n')
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for method in METHODS:
        epochs=range(1,21)
        for ax,key,scale in [(axes[0],'test_accuracy',100),(axes[1],'test_loss',1)]:
            mean=[statistics.mean(curves[s,method][i][key]*scale for s in (0,1,2)) for i in range(20)]
            sd=[statistics.stdev(curves[s,method][i][key]*scale for s in (0,1,2)) for i in range(20)]
            ax.plot(epochs,mean,label=method);ax.fill_between(epochs,[m-d for m,d in zip(mean,sd)],[m+d for m,d in zip(mean,sd)],alpha=.15)
    for ax,label in zip(axes,['Test Accuracy (%)','Test CE Loss']):ax.set(xlabel='Recovery epoch',ylabel=label);ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'recovery_three_seed_curves.png',dpi=180);plt.close(fig)
    save(out/'FINAL_AUDIT.json',{'status':'PASS','seeds':[0,1,2],'methods':list(METHODS),
        'rows':len(rows),'gpu_tasks_after_completion':False,'evidence_sha256':sha(out/'evidence.json')})
    (out/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.name}\n' for p in sorted(out.iterdir()) if p.is_file() and p.name!='SHA256SUMS'))

if __name__=='__main__':main()
