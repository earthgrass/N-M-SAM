import torch
from nmsam.models.tiny_mlp import TinyMLP
from nmsam.core.batch import StepBatch
from nmsam.experiment.training import BaselineTrainer,METHODS
from nmsam.sparsity.registry import MaskRegistry
from nmsam.sparsity.schedule import masks_from_trace,update_after_step
from nmsam.core.step_types import StepTrace


def test_taylor_uses_pre_update_master_not_post():
    m=TinyMLP();r=MaskRegistry(m)
    ps=tuple(torch.ones_like(p) for p in m.parameters())
    gs=tuple(torch.arange(p.numel(),dtype=p.dtype).reshape_as(p)+1 for p in m.parameters())
    trace=StepTrace(ps,gs,ps,ps,gs,ps,gs)
    first=masks_from_trace(r,trace,"taylor_g1")
    with torch.no_grad():
        for p in m.parameters():p.fill_(100)
    assert all(torch.equal(v,masks_from_trace(r,trace,"taylor_g1")[k]) for k,v in first.items())
    assert update_after_step(r,trace,1,2,"magnitude")==0 and r.version==0
    update_after_step(r,trace,2,2,"taylor_g2");assert r.version==1


def test_six_methods_share_train_core_and_semantics():
    b=StepBatch(torch.randn(4,1,4,4),torch.tensor([0,1,2,0]))
    for method in METHODS:
        trainer=BaselineTrainer(TinyMLP(),method)
        metrics=trainer.train(b)
        assert metrics["regularization_space"]=="master" and metrics["snapshot_id"]=="pre_step_1"
        assert metrics["restoration_error"]==0
        if trainer.registry:
            assert trainer.registry.version==1 and metrics["target_sparsity"]==.5
