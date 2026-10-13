from dataclasses import replace
from functools import partial
from time import perf_counter
import torch
from nmsam.core.loss import classification_loss
from nmsam.sam.train_step import train_step
from nmsam.sam.reference_sgd import create_optimizer
from nmsam.sparsity.registry import MaskRegistry
from nmsam.sparsity.training import sparse_step
from nmsam.sparsity.schedule import update_after_step
from .metrics import step_metrics

METHODS=("Dense-SGD","Dense-SAM","2:4-SGD-Magnitude","2:4-SAM-Magnitude",
         "2:4-SAM-Taylor-g1","2:4-SAM-Taylor-g2")


class BaselineTrainer:
    def __init__(self,model,method,*,lr=.05,rho=.05,l2_reg=.001,mask_interval=1,
                 beta=.9,nesterov=True,gradient_clipping=5.,ste=True,
                 scheduler="constant",total_steps=1):
        if method not in METHODS or mask_interval < 1:
            raise ValueError("unsupported method or mask interval")
        self.model,self.method,self.step=model,method,0
        self.rho=rho if "SAM" in method else 0.
        self.l2_reg,self.mask_interval=l2_reg,mask_interval
        self.gradient_clipping=gradient_clipping
        self.optimizer=create_optimizer(model.parameters(),lr=lr,beta=beta,nesterov=nesterov)
        if scheduler not in ("constant","cosine") or total_steps<1:raise ValueError("invalid scheduler")
        self.scheduler=(torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,T_max=total_steps)
                        if scheduler=="cosine" else torch.optim.lr_scheduler.LambdaLR(self.optimizer,lambda _:1.))
        self.registry=MaskRegistry(model,ste=ste) if method.startswith("2:4") else None
        self.scoring="taylor_g1" if method.endswith("g1") else "taylor_g2" if method.endswith("g2") else "magnitude"

    def train(self,batch):
        start=perf_counter()
        bn_before={k:v.clone() for k,v in self.model.named_buffers()}
        if self.registry:
            report=sparse_step(self.model,self.optimizer,batch,self.registry,rho=self.rho,
                               l2_reg=self.l2_reg,gradient_clipping=self.gradient_clipping,capture_trace=True)
        else:
            report=train_step(self.model,self.optimizer,batch,partial(classification_loss,l2_reg=self.l2_reg),
                              rho=self.rho,gradient_clipping=self.gradient_clipping,capture_trace=True)
        self.step+=1
        flip=update_after_step(self.registry,report.trace,self.step,self.mask_interval,self.scoring) if self.registry else 0.
        self.scheduler.step()
        metrics=step_metrics(replace(report,trace=None),self.step)
        inactive=0.; count=0; retained=0
        if self.registry:
            for n,m in self.registry.masks.items():
                p=dict(self.model.named_parameters())[n]
                inactive+=(p.detach()[~m].square().sum().item())
                count+=m.numel(); retained+=m.sum().item()
        metrics.update(method=self.method,objective="task_CE|optimizer_J",
            derivative=("STE_surrogate" if self.registry.ste else "pure_masked_true_J") if self.registry else "dense_true_J",
            state_mode="train_fixed_batch",mask_id=self.registry.version if self.registry else "dense",
            snapshot_id=f"pre_step_{self.step}",regularization_space="master",
            score_weight_time="W_pre",score_gradient_time="g_pre",application_time="next_step",
            mask_flip_rate=flip,target_sparsity=1-retained/count if count else 0.,
            mask_legal_rate=1.,inactive_master_norm=inactive**.5,
            bn_buffer_change=sum((v-bn_before[k]).float().norm().item() for k,v in self.model.named_buffers()),
            elapsed_seconds=perf_counter()-start)
        metrics["mask_score_objective"]="weight_magnitude" if self.scoring=="magnitude" else "optimizer_J"
        if self.registry and self.step%self.mask_interval==0:
            metrics["per_layer_legal_rate"]={name:1. for name in self.registry.masks}
            metrics["scope_exclusions"]=dict(self.registry.excluded)
        return metrics
