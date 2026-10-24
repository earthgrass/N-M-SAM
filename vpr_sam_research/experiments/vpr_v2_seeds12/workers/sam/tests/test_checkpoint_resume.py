import copy
import random
import numpy as np
import pytest
import torch
from torch import nn
from core import sam_step
from state import atomic_checkpoint,checkpoint_payload,load_checkpoint,validate_checkpoint
from train import choose_best
from scheduler import CosineLRscheduler


def test_checkpoint_exact_resume_and_rng(tmp_path,device):
    torch.manual_seed(15)
    def factory():
        return nn.Sequential(nn.Linear(3,4),nn.BatchNorm1d(4),nn.ReLU(),nn.Linear(4,2)).to(device)
    model=factory()
    opt=torch.optim.SGD(model.parameters(),lr=.05,momentum=.9,weight_decay=.001)
    scheduler=CosineLRscheduler(opt,T_max=200,eta_min=0.,warmup_epoch=0,warmup_init_lr=0.)
    x=torch.randn(5,3,device=device); y=torch.arange(5,device=device)%2
    sam_step(model,opt,x,y); scheduler.step(0)
    identity={'config':'fixed','source':'test-only','device':device}
    payload=checkpoint_payload(model,opt,scheduler,1,[{'steps':1}],identity,{'epoch':1})
    path=tmp_path/'last.pt'; atomic_checkpoint(path,payload)
    expected_rng=(random.random(),np.random.rand(),torch.rand(3))
    sam_step(model,opt,x,y); scheduler.step(1)
    expected=copy.deepcopy(model.state_dict())
    restored=factory()
    opt2=torch.optim.SGD(restored.parameters(),lr=.05,momentum=.9,weight_decay=.001)
    sched2=CosineLRscheduler(opt2,T_max=200,eta_min=0.,warmup_epoch=0,warmup_init_lr=0.)
    load_checkpoint(path,restored,opt2,sched2,identity)
    assert random.random()==expected_rng[0] and np.random.rand()==expected_rng[1]
    assert torch.equal(torch.rand(3),expected_rng[2])
    sam_step(restored,opt2,x,y); sched2.step(1)
    assert all(torch.equal(v,restored.state_dict()[k]) for k,v in expected.items())
    assert sched2.state_dict()==scheduler.state_dict()
    assert not (tmp_path/'last.pt.tmp').exists()
    with pytest.raises(ValueError): validate_checkpoint(payload,{'config':'changed'})


def test_best_accuracy_earliest_tie_not_lowest_loss():
    best=None
    for epoch,acc,loss in [(1,.9,.4),(2,.9,.3),(3,.91,.5),(4,.90,.2)]:
        best,changed=choose_best(best,{'epoch':epoch,'test_accuracy':acc,'test_loss':loss,'train_loss':.1})
        assert changed==(epoch in (1,3))
    assert best=={'epoch':3,'test_accuracy':.91,'test_loss_at_best_accuracy':.5,'train_loss':.1}


def test_corrupt_checkpoint_tensor_rejected(tmp_path,device):
    model=nn.Linear(2,2).to(device); opt=torch.optim.SGD(model.parameters(),lr=.1)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,200)
    payload=checkpoint_payload(model,opt,sched,0,[],{},None)
    payload['model']['weight']=torch.zeros_like(payload['model']['weight'])
    with pytest.raises(ValueError): validate_checkpoint(payload)
