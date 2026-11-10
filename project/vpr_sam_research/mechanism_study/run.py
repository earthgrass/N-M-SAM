"""Pre-registered matched-random mechanism study; no optimizer is constructed."""
import argparse, copy, csv, hashlib, json, math, subprocess, time, traceback
from pathlib import Path
import numpy as np
import torch
from torch import nn
from vpr_sam.training.engine import read_checkpoint
from vpr_sam.models import build_model
from vpr_sam.optim.geometry import Geometry
from vpr_sam.pruning import make_masks, apply_masks, validate_masks
from vpr_sam.data import build_data, Images, loader
from vpr_sam.evaluation import evaluate
from vpr_sam.repro import seed_all, seed_for, file_hash, tensor_hash, source_hashes

def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')

def table(path, rows):
    with path.open('w', newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def quant(x):
    return dict(zip(['min','q25','median','q75','max'], map(float,np.quantile(x,[0,.25,.5,.75,1])))) if len(x) else None

def bn_calibrate(model, train, device):
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    before=tensor_hash(dict(model.named_parameters()))
    modules=[m for m in model.modules() if isinstance(m,nn.modules.batchnorm._BatchNorm)]
    for m in modules: m.reset_running_stats(); m.momentum=None; m.train()
    with torch.no_grad():
        for x,_,_ in loader(train,128,0,workers=0): model(x.to(device))
    assert all(int(m.num_batches_tracked)==32 for m in modules)
    assert before==tensor_hash(dict(model.named_parameters()))
    model.eval()

def main(root,out,protocol_path=None,v1_root=None):
    protocol_path=protocol_path or Path(__file__).with_name('PROTOCOL.json')
    protocol=json.loads(protocol_path.read_text())
    out.mkdir(parents=True,exist_ok=False)
    dump(out/'PROTOCOL_FREEZE.json',dict(protocol=protocol,protocol_sha256=file_hash(protocol_path),script_sha256=file_hash(Path(__file__)),time=time.time()))
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()==protocol['producer_commit']
    assert subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip()==''
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    run=root/'runs/vpr_l2_seed0_v2/fixed_l2_seed0'
    check=run/'final.pt'
    old=root/'artifacts/geometry_guided_mask_seed0'
    original=[p for d in [run,root/'artifacts/vpr_l2',old] for p in d.rglob('*') if p.is_file()]
    before={str(p.relative_to(root)):file_hash(p) for p in original}
    sources=source_hashes(root)
    assert file_hash(check)==protocol['dense_sha256']
    state,_=read_checkpoint(check,root)
    assert state['epoch']==200 and state['stepper']['steps']==70400
    c=state['config']; torch.set_num_threads(c['threads']); seed_all(seed_for(0,'evaluation'),True)
    assert torch.cuda.get_device_name(0)=='NVIDIA GeForce RTX 5090'
    model=build_model(c['model']); model.load_state_dict(state['model'])
    geom=Geometry(model.named_parameters(),c['geometry'],'fixed_l2',0); geom.load_state_dict(state['stepper']['geometry'])
    logs={name:s for (name,_),s in zip(geom.mapping,geom.logs)}
    masks,_=make_masks(model,c['pruning'])
    frozen=torch.load(old/'masks.pt',map_location='cpu',weights_only=False)
    guided=frozen['geometry_guided']
    assert tensor_hash(masks)==tensor_hash(frozen['magnitude'])
    params=dict(model.named_parameters()); info={}; layers=[]; flips=[]; construction=[]; geometry_rows=[]
    for name in sorted(masks):
        w=params[name].detach().double(); s=logs[name]
        flat=w.reshape(w.shape[0],-1); head=flat.shape[1]//4*4
        assert head==flat.shape[1], 'Current study expects no incomplete tails; no silent mapping change'
        wb=flat.reshape(-1,4); sb=s.reshape(-1,4)
        a=masks[name].reshape(-1,4); b=guided[name].reshape(-1,4)
        order=wb.abs().argsort(dim=-1,descending=True,stable=True)
        ranks=wb.abs().gather(1,order)
        gap=((ranks[:,1]-ranks[:,2])/ranks[:,1].clamp_min(torch.finfo(torch.float64).tiny)).numpy()
        edges=(np.array(protocol['gap_edges'],dtype=np.float64) if 'gap_edges' in protocol else np.quantile(gap,np.linspace(0,1,21)))
        bins=np.searchsorted(edges[1:-1],gap,side='right')
        diff=(a!=b); counts=diff.sum(1); changed=counts>0
        srange=(sb.max(1).values-sb.min(1).values).numpy(); ratio=np.exp(srange)
        assert np.isfinite(ratio).all()
        patterns={}
        for idx in changed.nonzero().flatten().tolist():
            oldrank=a[idx].gather(0,order[idx]); newrank=b[idx].gather(0,order[idx])
            rm=tuple((oldrank & ~newrank).nonzero().flatten().tolist()); add=tuple((~oldrank & newrank).nonzero().flatten().tolist())
            key=(int(bins[idx]),rm,add); patterns.setdefault(key,[]).append(idx)
            flips.append(dict(layer=name,block=idx,flipped_coordinates=int(counts[idx]),gap=float(gap[idx]),s_range=float(srange[idx]),radius_ratio=float(ratio[idx]),bin=int(bins[idx]),removed_ranks=str(rm),added_ranks=str(add)))
        # Verify frozen geometry-guided score against coordinates and tie rule.
        scores=wb.abs()*(-sb).exp(); chosen=scores.argsort(dim=-1,descending=True,stable=True)[:,:2]
        assert torch.equal(torch.zeros_like(b).scatter_(1,chosen,True),b)
        layer=dict(layer=name,total_blocks=len(wb),flipped_coordinates=int(diff.sum()),changed_blocks=int(changed.sum()),swap2_blocks=int((counts==2).sum()),swap4_blocks=int((counts==4).sum()),flip_rate=float(diff.sum())/a.numel())
        layers.append(layer)
        geometry_rows.append(dict(layer=name,flipped_gap=quant(gap[changed.numpy()]),unflipped_gap=quant(gap[~changed.numpy()]),flipped_s_range=quant(srange[changed.numpy()]),unflipped_s_range=quant(srange[~changed.numpy()]),flipped_radius_ratio=quant(ratio[changed.numpy()]),unflipped_radius_ratio=quant(ratio[~changed.numpy()]),bin_edges=edges.tolist()))
        info[name]=dict(order=order,bins=bins,patterns=patterns,shape=w.shape,base=a,changed=changed)
    table(out/'MASK_FLIP_ANALYSIS.csv',layers); table(out/'flip_blocks.csv',flips); dump(out/'block_geometry_distributions.json',geometry_rows)
    nonmask=[]
    for name,s in logs.items():
        nonmask.append(dict(parameter=name,mask_eligible=name in masks,numel=s.numel(),sum_log_radius=float(s.sum()),log_radius_rms=float(s.square().mean().sqrt())))
    table(out/'geometry_allocation.csv',nonmask)
    controls=[]
    # Freeze every random mask before any new validation measurement.
    for seed in protocol['random_seeds']:
        rng=np.random.Generator(np.random.PCG64(seed)); result={}; overlaps=0
        for name,meta in info.items():
            b=meta['base'].clone(); used=set(); selected=[]
            for (bin_id,rm,add),guided_ids in sorted(meta['patterns'].items()):
                pool=np.array([i for i in np.flatnonzero(meta['bins']==bin_id) if i not in used],dtype=np.int64)
                ids=rng.choice(pool,size=len(guided_ids),replace=False)
                for idx in ids.tolist():
                    used.add(idx); selected.append(idx); b[idx,meta['order'][idx,list(rm)]]=False; b[idx,meta['order'][idx,list(add)]]=True
                    construction.append(dict(seed=seed,layer=name,block=idx,bin=bin_id,removed_ranks=str(rm),added_ranks=str(add)))
                assert len(ids)==len(guided_ids)
            assert bool((b.sum(1)==2).all())
            assert sorted((b!=meta['base']).sum(1).tolist())==sorted((guided[name].reshape(-1,4)!=meta['base']).sum(1).tolist())
            overlaps+=sum(bool(meta['changed'][i]) for i in selected)
            result[name]=b.reshape(meta['shape'])
        controls.append((seed,result,overlaps))
    table(out/'random_mask_construction.csv',construction)
    identities=[dict(seed=s,mask_sha256=tensor_hash(m),overlapping_guided_blocks=o) for s,m,o in controls]
    assert len(set(x['mask_sha256'] for x in identities))==len(controls)
    dump(out/'RANDOM_MASK_FREEZE.json',dict(protocol_sha256=file_hash(protocol_path),construction_sha256=file_hash(out/'random_mask_construction.csv'),masks=identities,strict_strata_matching=True,random_masks=len(controls),validation_observations_so_far=0))
    datasets,manifest=build_data(c,root); assert manifest==state['manifest']
    val=datasets['validation']; ids=json.loads((root/'docs/vpr_l2_reference/fixed_sample_ids.json').read_text())['0']['bn_train']; assert ids==manifest['train_indices'][:4096]
    train=Images(val.images,val.labels,ids,0,augment=False,cutout=0); device=torch.device(c['device'])
    prior=json.loads((old/'final_audit.json').read_text()); expected={r['method']:r for r in prior['methods']}
    results=[]
    for label,ms in [('magnitude',masks),('geometry_guided',guided)]+[(str(s),m) for s,m,_ in controls]:
        started=time.time(); sparse=build_model(c['model']).to(device); sparse.load_state_dict(state['model']); apply_masks(sparse,ms); validate_masks(sparse,ms,c['pruning'])
        raw=evaluate(sparse,loader(val,256,0,workers=0),device)
        # Preserve per-sample correctness for the two pre-existing methods only.
        if label in expected:
            preds=[]
            with torch.no_grad():
                sparse.eval()
                for x,y,i in loader(val,256,0,workers=0):
                    pred=sparse(x.to(device)).argmax(1).cpu(); preds.extend(dict(id=int(ii),target=int(yy),pred=int(pp),correct=bool(pp==yy)) for ii,yy,pp in zip(i,y,pred))
            table(out/(label+'_raw_predictions.csv'),preds)
        bn_calibrate(sparse,train,device); bn=evaluate(sparse,loader(val,256,0,workers=0),device)
        if label in expected:
            assert raw==expected[label]['raw'], 'Raw replay mismatch'
            assert bn==expected[label]['bn_calibrated'], 'BN replay mismatch'
        d2=da=0.
        for name,mask in ms.items():
            w=params[name].detach().double()[~mask]; s=logs[name][~mask]; d2+=float(w.square().sum()); da+=float((w*(-s).exp()).square().sum())
        row=dict(method=label,raw_accuracy=raw['accuracy'],raw_loss=raw['loss'],raw_loss_damage=raw['loss']-prior['dense']['loss'],bn_accuracy=bn['accuracy'],bn_loss=bn['loss'],bn_loss_damage=bn['loss']-prior['dense']['loss'],D2=d2,DA=da,mask_sha256=tensor_hash(ms),seconds=time.time()-started)
        assert all(math.isfinite(v) for v in row.values() if isinstance(v,float))
        results.append(row); table(out/'MATCHED_RANDOM_RESULTS.csv',results); print(json.dumps(row),flush=True)
        del sparse; torch.cuda.empty_cache()
    # V1 SAM BN-only baseline: original hard model, identical calibration IDs.
    if v1_root is None: raise ValueError('Explicit original V1 project root required')
    v1=v1_root
    hp=v1/'runs/main200_seeds012_v1/sam_seed0/pruning/n2_m4_exclude_first_last_magnitude_dense_final/pruned.pt'
    v1sha=file_hash(hp); oldstate,_=read_checkpoint(hp,v1); assert oldstate['manifest']==manifest
    baseline=build_model(c['model']).to(device); baseline.load_state_dict(oldstate['model'])
    samraw=evaluate(baseline,loader(val,256,0,workers=0),device); assert samraw['accuracy']==.854
    bn_calibrate(baseline,train,device); sambn=evaluate(baseline,loader(val,256,0,workers=0),device)
    dump(out/'DENSE_SAM_BN.json',dict(raw=samraw,bn=sambn,checkpoint_sha256=v1sha,train_ids_sha256=hashlib.sha256(np.asarray(ids,dtype=np.int64).tobytes()).hexdigest(),weights_unchanged=True))
    assert file_hash(hp)==v1sha
    after={str(p.relative_to(root)):file_hash(p) for p in original}; assert before==after and sources==source_hashes(root)
    dump(out/'FINAL_AUDIT.json',dict(status='PASS',producer_commit=protocol['producer_commit'],dense_sha256=file_hash(check),geometry_sha256=tensor_hash(logs),bn_train_ids_sha256=hashlib.sha256(np.asarray(ids,dtype=np.int64).tobytes()).hexdigest(),historical_files_unchanged=len(before),original_file_sha256=before,source_hashes=sources,optimizer_steps=0,test_used=False,random_controls=32,strict_matching=True,replay_exact=True))
    dump(out/'status.json',dict(status='COMPLETE',evaluations=len(results),optimizer_steps=0))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--project',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--protocol',type=Path); p.add_argument('--v1-project',type=Path,required=True); a=p.parse_args()
    try: main(a.project.resolve(),a.output.resolve(),a.protocol,a.v1_project.resolve())
    except BaseException as e:
        if a.output.exists(): dump(a.output/'failure.json',dict(error=type(e).__name__,message=str(e),traceback=traceback.format_exc(),automatic_retry=False))
        raise
