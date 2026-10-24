"""CPU research candidate only. No formal configuration or training entry point.

Frozen Global Geometry/Stepper files are imported unchanged. Only complete
pruning blocks learn radii; all excluded coordinates retain identity geometry.
"""
import copy
import math
import torch
from vpr_sam.optim.geometry import Geometry
from vpr_sam.optim.step import Stepper


class BlockwiseGeometry(Geometry):
    def __init__(self,named_parameters,config,eligible_names,*,m=4,n=2,gamma_block):
        if not 0<n<m or not math.isfinite(gamma_block) or gamma_block<=0:
            raise ValueError('Explicit finite positive gamma_block and valid n:m required')
        if config['dtype']!='float64': raise ValueError('FP64 geometry required')
        super().__init__(named_parameters,config,'fixed_l2',0)
        self.eligible=tuple(sorted(eligible_names)); self.m=m; self.n=n; self.gamma_block=gamma_block
        names={name:shape for name,shape in self.mapping}
        if not set(self.eligible)<=names.keys(): raise ValueError('Unknown eligible parameter')
        if any(len(names[k]) not in (2,4) for k in self.eligible): raise ValueError('Only Conv/Linear weights may learn block geometry')

    def _blocks(self,tensor):
        flat=tensor.reshape(tensor.shape[0],-1)
        head=flat.shape[1]//self.m*self.m
        return flat,head,flat[:,:head].reshape(-1,self.m)

    def check_local(self,logs):
        for (name,_),s in zip(self.mapping,logs):
            if not bool(torch.isfinite(s).all()): raise FloatingPointError('Nonfinite geometry')
            if name not in self.eligible:
                if bool((s!=0).any()): raise ValueError('Excluded coordinates require identity geometry')
                continue
            flat,head,blocks=self._blocks(s)
            if bool((flat[:,head:]!=0).any()): raise ValueError('Dense tail must retain identity geometry')
            tolerance=1e-12+1e-12*blocks.abs().sum(1)
            if bool((blocks.sum(1).abs()>tolerance).any()): raise FloatingPointError('Block log-volume drift; no recentering')
        self.check_volume(logs)

    @torch.no_grad()
    def prepare(self,eps,successful_steps,zero=False):
        if self.candidate is not None: raise RuntimeError('Uncommitted geometry')
        if len(eps)!=len(self.logs): raise ValueError('Feedback coordinate mapping mismatch')
        if any(e.shape!=s.shape for e,s in zip(eps,self.logs)): raise ValueError('Feedback shape mismatch')
        if any(not bool(torch.isfinite(e).all()) for e in eps): raise FloatingPointError('Nonfinite feedback')
        if zero or successful_steps<self.config['warmup_steps']:
            self.candidate=(None,zero)
            return dict(p_sum_error=0.,d_p_max=0.,max_abs_delta_s=0.,zero_energy_blocks=0)
        updates=[]; maxdelta=0.; maxshare=0.; error=0.; skipped=0
        for (name,_),s,e in zip(self.mapping,self.logs,eps):
            delta=torch.zeros_like(s)
            if name in self.eligible:
                _,head,eb=self._blocks(e.double()); df=delta.reshape(delta.shape[0],-1)
                scale=eb.abs().max(1,keepdim=True).values
                active=scale[:,0]>0; skipped+=int((~active).sum())
                # Scale before squaring: exact zero is the only skip threshold.
                normalized=eb[active]/scale[active]
                energy=normalized.square(); shares=energy/energy.sum(1,keepdim=True)
                db=torch.zeros_like(eb); db[active]=self.gamma_block*(1-self.m*shares)
                df[:,:head]=db.reshape(df.shape[0],head)
                if len(shares):
                    error=max(error,float((shares.sum(1)-1).abs().max()))
                    maxshare=max(maxshare,float((self.m*shares).max()))
                maxdelta=max(maxdelta,float(db.abs().max()) if db.numel() else 0.)
            updates.append(s+delta)
        self.check_local(updates); self.candidate=(updates,False)
        return dict(p_sum_error=error,d_p_max=maxshare,max_abs_delta_s=maxdelta,zero_energy_blocks=skipped)

    def state_dict(self):
        self.check_local(self.logs)
        return dict(block_version=1,m=self.m,n=self.n,gamma_block=self.gamma_block,eligible=self.eligible,global_state=super().state_dict())

    def load_state_dict(self,state):
        if (state['block_version'],state['m'],state['n'],state['gamma_block'],tuple(state['eligible']))!=(1,self.m,self.n,self.gamma_block,self.eligible):
            raise ValueError('Block geometry protocol mismatch')
        backup=copy.deepcopy(super().state_dict())
        try:
            super().load_state_dict(state['global_state']); self.check_local(self.logs)
        except BaseException:
            super().load_state_dict(backup)
            raise


class CandidateStepper(Stepper):
    """Reuses frozen two-forward/BN/SGD transactions; CPU smoke only."""
    def __init__(self,model,config,eligible_names,*,gamma_block,n=2,m=4):
        if config['device']!='cpu' or config['purpose']!='smoke':
            raise ValueError('Prototype has no formal/CUDA training authorization')
        super().__init__(model,config)
        self.geometry=BlockwiseGeometry(model.named_parameters(),config['geometry'],eligible_names,m=m,n=n,gamma_block=gamma_block)
