"""One detached sequential seed1->audit->seed2->audit queue; fail closed, no retry."""
import argparse,json,os,subprocess,sys,time,traceback,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import yaml
from benchmark.common import atomic_json,sha_file
from benchmark.preflight import ROOT,source_identity,require_config_hash
from benchmark.audit_inputs import load_original
from benchmark.obc_final_audit import audit

def protocol_equal(a,b):
    a=copy.deepcopy(a);b=copy.deepcopy(b)
    for c in (a,b):
        for field in ('name','seeds','reporting'):c.pop(field,None)
    return a==b

def main():
    p=argparse.ArgumentParser();p.add_argument('--expected-commit',required=True);p.add_argument('--expected-config-sha256',required=True)
    p.add_argument('--input-root',required=True);p.add_argument('--data-root',required=True)
    p.add_argument('--seed0-audit',required=True);p.add_argument('--seed0-audit-sha256',required=True);p.add_argument('--seed0-output',required=True)
    p.add_argument('--output',required=True);args=p.parse_args();os.chdir(ROOT)
    control=Path(args.output).resolve();control.mkdir(parents=True,exist_ok=False)
    cfg_path=ROOT/'configs/obc_seeds12_4090_final.yaml';active=None
    try:
        commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        if commit!=args.expected_commit or subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise ValueError('Clean frozen commit required')
        require_config_hash(cfg_path,args.expected_config_sha256)
        cfg=yaml.safe_load(cfg_path.read_text());base=yaml.safe_load((ROOT/'configs/obc_seed0_4090_pilot.yaml').read_text())
        if cfg['seeds']!=[1,2] or not protocol_equal(base,cfg) or cfg['reporting']['primary_checkpoint']!='last.pt':raise ValueError('Only reviewed seed/Final reporting changes accepted')
        if sha_file(args.seed0_audit)!=args.seed0_audit_sha256:raise ValueError('Reviewed pilot audit SHA')
        pilot=json.loads(Path(args.seed0_audit).read_text());old=Path(args.seed0_output)
        if pilot['status']!='PASS' or not all(v['pass'] for v in pilot['checks'].values()):raise ValueError('Pilot audit not PASS')
        if pilot['identity']['config_sha256']!=sha_file(ROOT/'configs/obc_seed0_4090_pilot.yaml'):raise ValueError('Pilot protocol config mismatch')
        if sha_file(old/'completion.json')!=pilot['small_file_sha256']['completion.json']:raise ValueError('Pilot completion changed')
        src=source_identity()
        for name,digest in pilot['identity']['source']['files'].items():
            if name.startswith('benchmark/') and src['files'][name]!=digest:raise ValueError('Audited implementation changed: '+name)
        review={'approved':True,'source_sha256':src['sha256'],'config_sha256':sha_file(cfg_path),
                'basis':'User approved seed1/2 after seed0 audited PASS; only Final reporting changes',
                'original_pilot_commit':pilot['identity']['commit'],'pilot_audit_sha256':sha_file(args.seed0_audit)}
        atomic_json(control/'pilot_review.json',review)
        records=json.loads((ROOT/'configs/existing_inputs.json').read_text())['checkpoints']
        for seed in (1,2):
            item=next(x for x in records if x['seed']==seed and x['role']=='standard_dense_sam')
            parent,_=load_original(Path(args.input_root)/item['relative_path'],item['sha256'])
            if parent['identity']['seed']!=seed or 'geometry' in parent or parent['identity']['data_manifest_sha256']!=pilot['identity']['data_manifest_sha256']:raise ValueError('Wrong seed parent/data')
        atomic_json(control/'status.json',{'status':'PRECHECK_PASS','seeds':[1,2],'reporting':'Final epoch20','source_sha256':src['sha256'],'commit':commit})
        for seed in (1,2):
            active=seed
            if source_identity()!=src:raise ValueError('Source changed during queue')
            preflight=[sys.executable,'-B','-m','benchmark.preflight','--config',str(cfg_path),'--gpu','--output',str(control/f'preflight_seed{seed}.json')]
            subprocess.run(preflight,check=True)
            out=ROOT/'artifacts'/f'full_obc_seed{seed}_4090_final_v1'
            if out.exists():raise FileExistsError('Refuse existing output')
            cmd=[sys.executable,'-B','-u','-m','benchmark.run_obc','--config',str(cfg_path),'--expected-commit',commit,
                 '--expected-config-sha256',sha_file(cfg_path),'--input-root',args.input_root,'--data-root',args.data_root,
                 '--seed',str(seed),'--pilot-review',str(control/'pilot_review.json'),'--output',str(out)]
            atomic_json(control/'status.json',{'status':'RUNNING','seed':seed,'command':cmd,'out':str(out),'started_unix':time.time()})
            with (control/f'seed{seed}.log').open('x') as log:
                child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                atomic_json(control/f'seed{seed}_process.json',{'pid':child.pid,'command':cmd,'output':str(out)})
                code=child.wait()
            if code!=0:raise RuntimeError(f'seed{seed} exited{code}; stopping queue without retry')
            completion=json.loads((out/'completion.json').read_text())
            ci=completion['identity']
            if ci['seed']!=seed or ci['commit']!=commit or ci['source']!=src or ci['config_sha256']!=sha_file(cfg_path):
                raise ValueError('Completed run identity changed')
            atomic_json(control/'status.json',{'status':'READONLY_AUDIT','seed':seed})
            receipt=audit(out,Path(args.input_root)/next(x['relative_path'] for x in records if x['seed']==seed and x['role']=='standard_dense_sam'),
                          next(x['sha256'] for x in records if x['seed']==seed and x['role']=='standard_dense_sam'),completion['identity'])
            atomic_json(control/f'FINAL_AUDIT_seed{seed}.json',receipt)
        atomic_json(control/'status.json',{'status':'COMPLETE','seeds':[1,2],'primary_checkpoint':'last.pt','all_audits':'PASS','ended_unix':time.time()})
    except Exception:
        atomic_json(control/'FAILURE.json',{'seed':active,'traceback':traceback.format_exc(),'no_automatic_retry':True})
        atomic_json(control/'status.json',{'status':'FAILED','seed':active});raise

if __name__=='__main__':main()
