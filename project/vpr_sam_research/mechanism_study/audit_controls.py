"""Independent CPU reconstruction of every pre-frozen random Mask."""
import argparse,collections,csv,json
from pathlib import Path
import numpy as np
import torch
from vpr_sam.training.engine import read_checkpoint
from vpr_sam.repro import tensor_hash

def main(root,folders):
    torch.set_num_threads(4)
    state,_=read_checkpoint(root/'runs/vpr_l2_seed0_v2/fixed_l2_seed0/final.pt',root)
    frozen=torch.load(root/'artifacts/geometry_guided_mask_seed0/masks.pt',map_location='cpu',weights_only=False)
    mags=frozen['magnitude'];guided=frozen['geometry_guided'];params=state['model']
    for out in folders:
        freeze=json.loads((out/'PROTOCOL_FREEZE.json').read_text());protocol=freeze['protocol']
        rows=list(csv.DictReader((out/'random_mask_construction.csv').open()))
        expected=collections.Counter(); gaps={}; orders={}; bins={}; guide_gaps=[]
        for name,mask in mags.items():
            w=params[name].double().reshape(-1,4).abs();order=w.argsort(dim=-1,descending=True,stable=True);rank=w.gather(1,order)
            gap=((rank[:,1]-rank[:,2])/rank[:,1].clamp_min(torch.finfo(torch.float64).tiny)).numpy();gaps[name]=gap;orders[name]=order
            edges=np.array(protocol['gap_edges']) if 'gap_edges' in protocol else np.quantile(gap,np.linspace(0,1,21))
            bins[name]=np.searchsorted(edges[1:-1],gap,side='right')
            old=mask.reshape(-1,4);new=guided[name].reshape(-1,4)
            for idx in (old!=new).any(1).nonzero().flatten().tolist():
                a=old[idx].gather(0,order[idx]);b=new[idx].gather(0,order[idx]);rm=tuple((a & ~b).nonzero().flatten().tolist());add=tuple((~a & b).nonzero().flatten().tolist())
                expected[(name,int(bins[name][idx]),str(rm),str(add))]+=1;guide_gaps.append(gap[idx])
        audits=[]; pooled=[]
        for seed in protocol['random_seeds']:
            rr=[r for r in rows if int(r['seed'])==seed];counter=collections.Counter((r['layer'],int(r['bin']),r['removed_ranks'],r['added_ranks']) for r in rr)
            assert counter==expected and len(set((r['layer'],r['block']) for r in rr))==len(rr)
            ms={n:m.clone() for n,m in mags.items()};selected_gaps=[]
            import ast
            for r in rr:
                n=r['layer'];idx=int(r['block']);assert int(bins[n][idx])==int(r['bin'])
                b=ms[n].reshape(-1,4);rm=ast.literal_eval(r['removed_ranks']);add=ast.literal_eval(r['added_ranks'])
                b[idx,orders[n][idx,list(rm)]]=False;b[idx,orders[n][idx,list(add)]]=True;selected_gaps.append(gaps[n][idx])
            for n,m in ms.items():assert bool((m.reshape(-1,4).sum(1)==2).all())
            identity=next(r for r in json.loads((out/'RANDOM_MASK_FREEZE.json').read_text())['masks'] if r['seed']==seed)
            assert tensor_hash(ms)==identity['mask_sha256'];pooled.extend(selected_gaps)
            audits.append(dict(seed=seed,sha256=identity['mask_sha256'],strict_stratum_rank_counts_verified=True,mask_legal=True,changed_blocks=len(rr)))
        def summary(a):return dict(zip(['min','q25','median','q75','max'],map(float,np.quantile(a,[0,.25,.5,.75,1]))))
        record=dict(status='PASS',cpu_only=True,controls=audits,guided_gap=summary(guide_gaps),random_gap=summary(pooled),comparison='pooled controls, same size each; excludes validation metrics',independent_hash_reconstruction=True)
        target=out/'CONTROL_INDEPENDENT_AUDIT.json';assert not target.exists();target.write_text(json.dumps(record,indent=2)+'\n')
    print('CPU CONTROL RECONSTRUCTION PASS')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--folders',type=Path,nargs='+',required=True);a=p.parse_args();main(a.project,a.folders)
