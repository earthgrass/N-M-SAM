"""Read frozen result JSONs after one authorized job; never opens any dataset."""
import argparse,hashlib,json,os,time,traceback
from pathlib import Path

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--references',type=Path,required=True)
    p.add_argument('--reference-sha256',required=True);p.add_argument('--training-pid',type=int,required=True);a=p.parse_args()
    # Keep this post-run report outside the training evidence manifest so its
    # later writes cannot invalidate the frozen main-run SHA256 inventory.
    out=a.run.parent/(a.run.name+'_historical_comparison')
    if out.exists():raise FileExistsError(out)
    out.mkdir()
    try:
        if sha(a.references)!=a.reference_sha256:raise ValueError('Reference snapshot hash mismatch')
        write(out/'status.json',{'status':'WAITING_FOR_FINAL_EVALUATION','new_test_access':False})
        started=time.monotonic()
        while True:
            state=json.loads((a.run/'status.json').read_text())
            if state['stage']=='complete':break
            if state['stage']=='failed':raise RuntimeError('Training/evaluation failed; comparison not run')
            os.kill(a.training_pid,0)
            if time.monotonic()-started>86400:raise TimeoutError('Comparison wait exceeded24h')
            time.sleep(30)
        if sha(a.references)!=a.reference_sha256:raise ValueError('Reference snapshot changed')
        result_path=a.run/'SEED0_EVIDENCE_REPORT.json';result=json.loads(result_path.read_text())
        if result['completed_epochs']!=200 or result['steps']!=78200 or result['seed']!=0:raise ValueError('Final contract mismatch')
        refs=json.loads(a.references.read_text());hq=result['results']['sparsegpt_native'];rows=[]
        for r in refs['references']:
            rows.append({'reference_id':r['reference_id'],'reference_accuracy_percent':r['accuracy_percent'],'reference_ce':r['ce'],
              'hq_accuracy_percent':hq['accuracy_percent'],'hq_ce':hq['ce'],
              'hq_minus_reference_accuracy_pp':hq['accuracy_percent']-r['accuracy_percent'],
              'hq_minus_reference_ce':hq['ce']-r['ce'],
              'reference_evaluation_sha256':r['evaluation']['sha256'],'reference_checkpoint_sha256':r['checkpoint']['sha256']})
        report={'status':'REFERENCE_COMPARISON_COMPLETE','comparison_status':'REFERENCE_ONLY','strict_ranking_allowed':False,
          'reason':'Historical hardware/environment and execution records differ; same initialization alone does not establish a paired causal experiment.',
          'no_dataset_opened':True,'no_evaluation_repeated':True,'source_script_sha256':sha(__file__),
          'reference_snapshot_sha256':a.reference_sha256,'hq_evidence_sha256':sha(result_path),
          'rows':rows,'hq_dense_accuracy_percent':result['results']['dense']['accuracy_percent'],
          'hq_dense_to_native_drop_pp':result['dense_to_native_sparse_drop_pp']}
        write(out/'comparison.json',report)
        lines=['# Seed0 historical reference comparison','','All rows are reference-only. No Test evaluation was repeated.','',
          '| Historical reference | Native accuracy % | HQ minus reference pp | CE |','|---|---:|---:|---:|']
        lines += [f"| {r['reference_id']} | {r['reference_accuracy_percent']:.2f} | {r['hq_minus_reference_accuracy_pp']:+.2f} | {r['reference_ce']:.6f} |" for r in rows]
        lines += ['',f"HQ/HC-SAM native SparseGPT: {hq['accuracy_percent']:.2f}%, CE {hq['ce']:.6f}.",report['reason']]
        (out/'COMPARISON.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
        write(out/'status.json',{'status':'COMPLETE','new_test_access':False,'source_script_sha256':sha(__file__)})
        write(out/'SHA256.json',{p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='SHA256.json'})
    except BaseException as e:
        write(out/'failure.json',{'status':'BLOCKED','error':str(e),'traceback':traceback.format_exc(),'automatic_retry':False});raise

if __name__=='__main__':main()
