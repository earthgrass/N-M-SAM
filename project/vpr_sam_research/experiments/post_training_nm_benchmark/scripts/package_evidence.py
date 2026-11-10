"""Package small CPU evidence and reused historical metrics, never large weights."""
import argparse,csv,json,shutil,statistics,sys,xml.etree.ElementTree as ET
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from benchmark.common import atomic_json,sha_file,environment
from benchmark.preflight import source_identity

def main():
    p=argparse.ArgumentParser();p.add_argument('--delivery-root',required=True);a=p.parse_args()
    source=Path(a.delivery_root)/'final_seeds012';out=ROOT/'docs/research/evidence';out.mkdir(parents=True,exist_ok=True)
    for name in ('per_seed_metrics.csv','statistics.json'):
        destination=out/('historical_'+name);shutil.copyfile(source/name,destination)
        assert sha_file(source/name)==sha_file(destination)
    for name in ('stage1_inventory.json','compatibility_probes.json','PREFLIGHT_PENDING.json','cpu_tests.xml','test_environment.json'):
        shutil.copyfile(ROOT/'artifacts'/name,out/name)
    rows=list(csv.DictReader((out/'historical_per_seed_metrics.csv').open(encoding='utf-8-sig')))
    lines=['# Accuracy-only comparison draft — new baselines pending','',
        'Historical metrics are reused, not rerun. Percent, mean ± sample standard deviation; n=3 independent seeds.',
        'Official Test participates in Best selection. Native compensated outputs will be distinguished from unmodified Raw Hard.',
        '', '| Method | Dense Acc (%) | Native post-pruning / Raw Hard Acc (%) | Recovery Best Acc (%) |',
        '|---|---:|---:|---:|']
    for key,label in [('sam_magnitude','Dense-SAM + Magnitude'),('vpr_magnitude','APBR training + Magnitude (ablation)'),
                      ('vpr_guided','APBR-SAM (Full VPR-L2 + Geometry-guided)')]:
        subset=[r for r in rows if r['method']==key]
        if sorted(int(r['seed']) for r in subset)!=[0,1,2]:raise ValueError('Historical per-seed coverage')
        stats=[]
        for field in ('dense_accuracy_pct','raw_accuracy_pct','recovery_best_accuracy_pct'):
            values=[float(r[field]) for r in subset];stats.append(f'{statistics.mean(values):.4f} ± {statistics.stdev(values):.4f}')
        lines.append('| '+label+' | '+' | '.join(stats)+' |')
    lines+=['| Dense-SAM + OBC (controlled layout adapter) | existing same parents | PENDING GPU/calibration | PENDING |',
            '| Dense-SAM + SNOWS | existing same parents | BLOCKED official source | BLOCKED |','',
            'Final, Loss, best epochs, Mask/parent hashes and per-seed paired confidence intervals remain in the accompanying historical CSV/statistics JSON.',
            'Existing intervals are exploratory and based on only three independent seeds; no SOTA or stable superiority conclusion is drawn.',
            'No external published accuracy is merged into this common-protocol table.']
    (ROOT/'docs/research/accuracy_table.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    tree=ET.parse(out/'cpu_tests.xml');suites=tree.getroot().findall('testsuite')
    result={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    test_env=json.loads((out/'test_environment.json').read_text(encoding='utf-8'))
    result.update(status='PASS' if result['failures']==result['errors']==0 else 'FAIL',test_runtime=test_env,
                  packaging_environment=environment(),
                  historical_inputs_reused=True,formal_experiments_run=False,gpu_contacted=False)
    atomic_json(out/'cpu_validation_summary.json',result)
    hashes={f.name:sha_file(f) for f in sorted(out.iterdir()) if f.is_file() and f.name!='SHA256SUMS.json'}
    atomic_json(out/'SHA256SUMS.json',hashes)
    print(json.dumps(result))

if __name__=='__main__':main()
