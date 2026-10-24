import copy
import torch
from torch import nn
from core import bn_snapshot,evaluate,restore_bn_buffers,sam_step


def small_model(device):
    return nn.Sequential(nn.Conv2d(3,4,3,padding=1),nn.BatchNorm2d(4),nn.ReLU(),nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(4,2)).to(device)


def test_second_forward_keeps_batch_stats_and_all_buffers(device):
    torch.manual_seed(0); model=small_model(device); model.train()
    x=torch.randn(8,3,8,8,device=device)+2
    before=bn_snapshot(model); model(x); first=bn_snapshot(model)
    assert not torch.equal(before['1']['running_mean'],first['1']['running_mean'])
    assert not torch.equal(before['1']['running_var'],first['1']['running_var'])
    assert first['1']['num_batches_tracked'].item()==1
    with restore_bn_buffers(model):
        train_output=model(x)
        assert model.training and model[1].training
        torch.nn.functional.cross_entropy(train_output,torch.arange(8,device=device)%2).backward()
    assert all(torch.equal(v,bn_snapshot(model)[n][k]) for n,b in first.items() for k,v in b.items())
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_complete_step_updates_bn_exactly_once_and_eval_never_updates(device):
    torch.manual_seed(8); model=small_model(device); reference=copy.deepcopy(model).train()
    x=torch.randn(8,3,8,8,device=device); y=torch.arange(8,device=device)%2
    reference(x); expected=bn_snapshot(reference)
    sam_step(model,torch.optim.SGD(model.parameters(),lr=.05),x,y)
    assert all(torch.equal(v,bn_snapshot(model)[n][k]) for n,b in expected.items() for k,v in b.items())
    assert all(torch.isfinite(v).all() for v in model.state_dict().values())
    evaluate(model,[(x,y,torch.arange(8))],device)
    assert all(torch.equal(v,bn_snapshot(model)[n][k]) for n,b in expected.items() for k,v in b.items())
