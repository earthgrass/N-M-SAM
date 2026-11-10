"""Read downloaded checkpoints and JSONs; no dataset loader or model evaluation."""
import argparse,json,math,hashlib,sys,gc
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from hq_apbr_v1.training import validate_checkpoint,tensor_sha,file_sha
from hq_apbr_v1.vendor.nm import validate_keep_mask
EXPECTED='56ff197b0860dfd0f4a0768c5609031214e6405c51557009a9977582b38400e7'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def check(x,msg):
    if not x:raise AssertionError(msg)
def finite(x):
    if isinstance(x,float):check(math.isfinite(x),'Nonfinite JSON value')
    elif isinstance(x,dict):
        for v in x.values():finite(v)
    elif isinstance(x,list):
        for v in x:finite(v)
def main():
    a=argparse.ArgumentParser();a.add_argument('root',type=Path);arg=a.parse_args();root=arg.root;run=root/'seed0_full_override_v1';ev=run/'evaluation'
    report=read(run/'SEED0_EVIDENCE_REPORT.json');finite(report)
    check(read(root/'LOCAL_SHA256_VERIFICATION.json')['status']=='LOCAL_TRANSFER_SHA256_PASS','Transfer incomplete')
    check((report['status'],report['seed'],report['completed_epochs'],report['steps'],report['recovery_epochs'])==('SEED0_EXPLORATORY_COMPLETE',0,200,78200,0),'Final budget violation')
    check(report['performance_gate']=='FAIL','Historical performance failure changed')
    check(report['identity']['config_sha256']==EXPECTED,'Config mismatch')
    history=read(run/'training/history.json');check(len(history)==200,'Incomplete history')
    for i,row in enumerate(history):
        finite(row);check(row['epoch']==i and row['steps']==391*(i+1) and row['train_samples']==50000 and row['batches']==391,'Epoch coverage')
        check(row['test_accessed'] is False,'Training Test access')
        check(abs(row['eps_norm_min']-.05)<1e-6 and abs(row['eps_norm_max']-.05)<1e-6,'Perturbation radius')
        check(row['skipped_calibration_epochs']==[],'Unexpected skipped refresh')
        cal=read(run/'training'/f'calibration_epoch_{i:03d}.json');finite(cal)
        check(cal['buffers_sha256_before']==cal['buffers_sha256_after'],'BN mutation')
        check(cal['parameters_sha256_before']==cal['parameters_sha256_after'],'Calibration parameter mutation')
        check(len(cal['layers'])==19,'Calibration layer count')
        for layer in cal['layers'].values():
            for diag in layer['groups_diagnostics']:check(diag['cholesky_status']=='PASS','Cholesky failure')
    parent=torch.load(run/'training/final_epoch_199.pt',map_location='cpu',weights_only=False)
    validate_checkpoint(parent,report['identity']);check(parent['steps']==78200 and parent['epoch_index']==199,'Parent epoch')
    check(file_sha(run/'training/final_epoch_199.pt')==report['final_checkpoint_sha256'],'Parent hash')
    dense=parent['model'];geom=parent['stepper']['geometry']
    geometry_sum=sum(float(x.double().sum()) for x in geom.values());geometry_max=max(float(x.abs().max()) for x in geom.values())
    check(all(bool(torch.isfinite(x).all()) for x in geom.values()),'Geometry nonfinite')
    freeze=read(ev/'all_endpoints_frozen.json');check(freeze['test_accessed'] is False and freeze['status']=='ALL_ENDPOINTS_FROZEN','Freeze identity')
    check(read(ev/'test_started.json')['frozen_manifest_sha256']==file_sha(ev/'all_endpoints_frozen.json'),'Pre-Test manifest')
    check(freeze['parent_sha256']==report['final_checkpoint_sha256'],'Freeze parent')
    rows=[]
    for name,rec in freeze['endpoints'].items():
        path=ev/Path(rec['checkpoint_path']).name;check(file_sha(path)==rec['checkpoint_sha256'],'Endpoint SHA')
        ck=torch.load(path,map_location='cpu',weights_only=False)
        check(tensor_sha(ck['model'])==ck['model_sha256']==rec['model_sha256'],'Model tensor hash')
        check(tensor_sha(ck['masks'])==ck['mask_sha256']==rec['mask_sha256'],'Mask tensor hash')
        check(ck['parent_sha256']==report['final_checkpoint_sha256'],'Endpoint parent')
        check(ck['test_accessed'] is False and ck['recovery_epochs']==0,'Endpoint provenance')
        check(all(bool(torch.isfinite(v).all()) for v in ck['model'].values()),'Nonfinite model')
        count=deleted=0
        for key,keep in ck['masks'].items():
            validate_keep_mask(keep.reshape(keep.shape[0],-1),2,4);check(keep.dtype==torch.bool,'Mask dtype')
            check(torch.count_nonzero(ck['model'][key][~keep])==0,'Deleted nonzero')
            count+=keep.numel();deleted+=int((~keep).sum())
        if name!='dense':check((len(ck['masks']),count,deleted)==(19,11157504,5578752),'Strict scope count')
        for key,w in dense.items():
            if key not in ck['masks']:check(torch.equal(w,ck['model'][key]),'Noneligible / BN changed')
        if name=='final_q_mask_only':
            for key,keep in ck['masks'].items():check(torch.equal(dense[key]*keep,ck['model'][key]),'Mask-only branch mismatch')
        metric=read(ev/f'{name}_evaluation.json');check(metric==report['results'][name] and metric['samples']==10000,'Evaluation record mismatch')
        check(metric['checkpoint_sha256']==rec['checkpoint_sha256'],'Evaluated checkpoint mismatch')
        rows.append({'endpoint':name,'accuracy_percent':metric['accuracy_percent'],'ce':metric['ce'],'mask_layers':len(ck['masks']),'eligible_coordinates':count,'deletions':deleted,'checkpoint_sha256':rec['checkpoint_sha256'],'mask_sha256':rec['mask_sha256'],'audit':'PASS'})
        del ck;gc.collect()
    out={'status':'COMPLETED_SEED0_LOCAL_AUDIT_PASS','test_repeated':False,'dataset_opened':False,'training_epochs':200,'steps':78200,'refreshes':200,'layers_per_refresh':19,'geometry_sum':geometry_sum,'geometry_max_abs':geometry_max,'results':rows,'train_seconds':report['train_seconds'],'calibration_seconds':report['calibration_seconds'],'performance_gate':'FAIL','override':report['research_override'],'parent_sha256':report['final_checkpoint_sha256'],'config_sha256':EXPECTED,'peak_training_allocated_bytes':max(r['peak_allocated_bytes'] for r in history),'peak_training_reserved_bytes':max(r['peak_reserved_bytes'] for r in history)}
    (root/'LOCAL_FINAL_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    labels={'dense':'HC-SAM Dense Final','final_q_mask_only':'Final-Q Mask-only','final_q_fixed_obs':'Final-Q + Fixed-mask OBS','sparsegpt_native':'HC-SAM + Native SparseGPT'}
    lines=['# HC-SAM Seed0 local receipt and audit','','| Endpoint | Accuracy |','|---|---:|']+[f"| {labels[r['endpoint']]} | {r['accuracy_percent']:.2f}% |" for r in rows]
    lines+=['','200 epochs / 78,200 steps; 200 training-only calibration refreshes. Strict 2:4 checked for all three sparse endpoints. No Recovery or new Test evaluation.','Original performance gate FAIL retained; explicit research override used.','All remote copies preserved; local SHA256 and checkpoint/Mask audits PASS.']
    (root/'LOCAL_RECEIPT_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(out))
if __name__=='__main__':main()
