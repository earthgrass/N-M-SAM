import ast
from pathlib import Path

import torch
from torch import nn
from core import bn_snapshot,evaluate,sam_step


def small(device):
    return nn.Sequential(nn.Conv2d(3,4,3,padding=1),nn.BatchNorm2d(4),nn.ReLU(),
                         nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(4,2)).to(device)


def test_two_bn_updates_and_eval_no_updates(device):
    model=small(device); x=torch.randn(8,3,8,8,device=device)+2; y=torch.arange(8,device=device)%2
    before=bn_snapshot(model)
    result=sam_step(model,torch.optim.SGD(model.parameters(),lr=.05),x,y,audit_bn=True)
    audit=result['bn_audit']['1']; after=bn_snapshot(model)
    assert audit['first_counter']==1 and audit['second_counter']==2
    assert audit['first_mean_change']>0 and audit['second_mean_change']>0
    assert audit['first_var_change']>0 and audit['second_var_change']>0
    assert not torch.equal(before['1']['running_mean'],after['1']['running_mean'])
    assert all(torch.isfinite(v).all() for v in model.state_dict().values())
    assert all(m.track_running_stats and m.momentum==.1 for m in model.modules() if isinstance(m,nn.BatchNorm2d))
    evaluate(model,[(x,y,torch.arange(8))],device)
    assert all(torch.equal(v,bn_snapshot(model)[n][k]) for n,b in after.items() for k,v in b.items())


def test_actual_fsam_helper_freezes_mean_var_but_not_counter(device):
    path=Path(__file__).resolve().parents[1]/'reference/fsam/train.py'
    tree=ast.parse(path.read_text())
    selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['disable_running_stats','enable_running_stats']]
    ns={'_BatchNorm':nn.modules.batchnorm._BatchNorm}
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),ns)
    model=small(device).train(); x=torch.randn(8,3,8,8,device=device)
    model(x); first=bn_snapshot(model)
    ns['disable_running_stats'](model); output=model(x); output.sum().backward(); second=bn_snapshot(model)
    assert torch.equal(first['1']['running_mean'],second['1']['running_mean'])
    assert torch.equal(first['1']['running_var'],second['1']['running_var'])
    assert second['1']['num_batches_tracked'].item()==first['1']['num_batches_tracked'].item()+1
    ns['enable_running_stats'](model)
    assert model[1].momentum==.1
