"""Import completed local artifacts only. Does not load data/models or run Test."""
import argparse, hashlib, json, math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
EXPECTED='56ff197b0860dfd0f4a0768c5609031214e6405c51557009a9977582b38400e7'
def load(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()
def require(value,msg):
    if not value: raise ValueError(msg)
def validate(run):
    report=load(run/'SEED0_EVIDENCE_REPORT.json')
    require(report['status']=='SEED0_EXPLORATORY_COMPLETE','Incomplete run')
    require((report['seed'],report['completed_epochs'],report['steps'],report['recovery_epochs'])==(0,200,78200,0),'Protocol mismatch')
    require(report['performance_gate']=='FAIL','Original performance failure must be retained')
    ev=run/'evaluation'; manifest=load(ev/'all_endpoints_frozen.json')
    require(manifest['status']=='ALL_ENDPOINTS_FROZEN' and manifest['test_accessed'] is False,'Invalid pre-test freeze')
    require(manifest['config_sha256']==EXPECTED,'Config mismatch')
    require(manifest['parent_sha256']==report['final_checkpoint_sha256'],'Parent mismatch')
    require(load(ev/'test_started.json')['frozen_manifest_sha256']==sha(ev/'all_endpoints_frozen.json'),'Pre-test manifest identity mismatch')
    require(set(manifest['endpoints'])=={'dense','final_q_mask_only','final_q_fixed_obs','sparsegpt_native'},'Missing/extra endpoint')
    parent=run/'training'/'final_epoch_199.pt'
    # Locate the exact parent by hash, without guessing or accepting a name alone.
    candidates=list((run/'training').glob('*.pt'))
    require(any(sha(p)==report['final_checkpoint_sha256'] for p in candidates),'Verified final dense parent file missing')
    for name,item in manifest['endpoints'].items():
        checkpoint=ev/Path(item['checkpoint_path']).name
        require(checkpoint.exists() and sha(checkpoint)==item['checkpoint_sha256'],f'{name}: checkpoint hash mismatch')
        result=load(ev/f'{name}_evaluation.json')
        require(result==report['results'][name],f'{name}: report differs from evaluation JSON')
        require(result['checkpoint_sha256']==item['checkpoint_sha256'] and result['mask_sha256']==item['mask_sha256'],f'{name}: evaluation identity mismatch')
        require(result['parent_sha256']==manifest['parent_sha256'] and result['config_sha256']==EXPECTED,f'{name}: parent/config mismatch')
        require(result.get('samples',result.get('count'))==10000,f'{name}: test count mismatch')
        require(math.isfinite(result['accuracy_percent']) and 0<=result['accuracy_percent']<=100 and math.isfinite(result['ce']) and result['ce']>=0,'Invalid metrics')
    return report
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-dir',type=Path,required=True);ap.add_argument('--write',action='store_true');args=ap.parse_args()
    report=validate(args.run_dir)
    d=report['results']['dense'];s=report['results']['sparsegpt_native']
    row=f"HC-SAM & Current exploratory run & {d['accuracy_percent']:.2f} & {d['ce']:.4f} & {s['accuracy_percent']:.2f} & {s['ce']:.4f} & {d['accuracy_percent']-s['accuracy_percent']:.2f}"+r'\\'+'\n'
    print(row)
    if args.write:
        dest=ROOT/'evidence/imported_final'
        require(not dest.exists(),'Final import already exists; create a new version instead of overwriting')
        dest.mkdir()
        source=args.run_dir/'SEED0_EVIDENCE_REPORT.json'
        (dest/'report.json').write_bytes(source.read_bytes())
        (dest/'import.json').write_text(json.dumps({'source':str(source.resolve()),'sha256':sha(source),'config_sha256':EXPECTED,'status':'HASH_VALIDATED_IMPORT','scope':'single-seed exploratory; not a matched cohort'},indent=2)+'\n')
        (ROOT/'tables/hc_result_row.tex').write_text(r'\newcommand{\HCResultRow}{'+row.strip()+'}\n',encoding='utf-8')
        print('Imported. Rebuild and review the prose: it still explicitly describes a pending draft.')
if __name__=='__main__': main()
