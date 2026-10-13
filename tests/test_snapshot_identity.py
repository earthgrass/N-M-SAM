import torch
from torch import nn
from nmsam.core.batch import StepBatch
from nmsam.core.functional_model import EffectiveLoss
from nmsam.sparsity.registry import MaskRegistry


def test_snapshot_binds_batch_and_inactive_master_not_only_effective_z():
    model=nn.Linear(4,2).double();registry=MaskRegistry(model)
    batch=StepBatch(torch.ones(2,4,dtype=torch.float64),torch.tensor([0,1]))
    first=EffectiveLoss(model,batch,registry)
    same=EffectiveLoss(model,batch,registry)
    assert first.labels["snapshot_id"]==same.labels["snapshot_id"]
    changed=EffectiveLoss(model,StepBatch(batch.inputs*2,batch.targets),registry)
    assert changed.labels["snapshot_id"]!=first.labels["snapshot_id"]
    inactive=(~registry.masks["weight"]).nonzero()[0]
    with torch.no_grad():model.weight[inactive[0],inactive[1]].add_(1)
    master_changed=EffectiveLoss(model,batch,registry)
    assert torch.equal(master_changed.z,first.z)
    assert master_changed.labels["snapshot_id"]!=first.labels["snapshot_id"]
