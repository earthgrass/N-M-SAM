import ast
import math
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
import yaml

from data import Images,make_loader
from model import build_model
from scheduler import CosineLRscheduler

ROOT=Path(__file__).resolve().parents[1]


def test_model_classes_match_official_ast():
    def classes(path):
        return {n.name:ast.dump(n) for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name in ['BasicBlock','ResNet']}
    assert classes(ROOT/'model.py')==classes(ROOT/'reference/ssam2022/models/resnet.py')
    model=build_model()
    assert sum(p.numel() for p in model.parameters())==11173962
    assert model.conv1.kernel_size==(3,3) and model.conv1.stride==(1,1)
    assert [len(getattr(model,f'layer{i}')) for i in range(1,5)]==[2,2,2,2]


def test_all_200_lr_values_match_exact_official_call_order():
    namespace={'math':math,'optim':torch.optim}
    source=ast.parse((ROOT/'reference/ssam2022/solver/lr_scheduler.py').read_text())
    nodes=[]
    for n in source.body:
        if isinstance(n,ast.ClassDef) and n.name in ['LRscheduler','CosineLRscheduler']:
            n.decorator_list=[]
            for f in n.body:
                if isinstance(f,ast.FunctionDef): f.decorator_list=[d for d in f.decorator_list if not(isinstance(d,ast.Name) and d.id=='configurable')]
            nodes.append(n)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'official_scheduler','exec'),namespace)
    a=torch.optim.SGD([torch.nn.Parameter(torch.tensor(1.))],lr=.05)
    b=torch.optim.SGD([torch.nn.Parameter(torch.tensor(1.))],lr=.05)
    args=dict(T_max=200,eta_min=0.,warmup_epoch=0,warmup_init_lr=0.)
    ours=CosineLRscheduler(a,**args); official=namespace['CosineLRscheduler'](b,**args)
    used=[]
    for epoch in range(200):
        used.append(a.param_groups[0]['lr'])
        assert a.param_groups[0]['lr']==b.param_groups[0]['lr']
        ours.step(epoch); official.step(epoch)
        assert ours.state_dict()==official.state_dict()
    assert used[0]==used[1]==.05
    assert used[-1]==.05*(1+math.cos(math.pi*198/200))/2


class FakeImages:
    def __len__(self): return 50000
    def __getitem__(self,i): return Image.fromarray(np.full((32,32,3),i%256,np.uint8)),i%10


def test_full50k_train_and_full10k_eval_batch_sizes():
    c=yaml.safe_load((ROOT/'config.yaml').read_text()); data=Images(FakeImages(),c,True)
    loader=make_loader(data,c,True,workers=0)
    batches=list(iter(loader.batch_sampler))
    assert len(data)==50000 and len(batches)==390
    assert all(len(b)==128 for b in batches)
    assert len(set(i for b in batches for i in b))==49920
    class TestImages(FakeImages):
        def __len__(self): return 10000
    test=make_loader(Images(TestImages(),c),c,workers=0)
    batch_ids=list(iter(test.batch_sampler))
    assert len(batch_ids)==79 and len(batch_ids[-1])==16
    assert sum(map(len,batch_ids))==10000
    assert c['seed'] in (0,1,2) and c['validation_size']==0


def test_ssam_cutout_matches_reference_on_fixed_rng():
    c=yaml.safe_load((ROOT/'config.yaml').read_text())
    dataset=Images(FakeImages(),c,True)
    assert [type(t).__name__ for t in dataset.transform.transforms]==['RandomCrop','RandomHorizontalFlip','ToTensor','Normalize','Cutout']
    assert np.array_equal(np.asarray(c['normalize_mean']),np.array([125.3,123.,113.9])/255)
    assert np.array_equal(np.asarray(c['normalize_std']),np.array([63.,62.1,66.7])/255)
    source=ast.parse((ROOT/'reference/ssam2022/data/dataset.py').read_text())
    node=next(n for n in source.body if isinstance(n,ast.ClassDef) and n.name=='Cutout')
    ns={'torch':torch,'np':np}; exec(compile(ast.Module(body=[node],type_ignores=[]),'cutout','exec'),ns)
    from data import Cutout
    for seed in range(20):
        torch.manual_seed(seed); np.random.seed(seed)
        a=Cutout()(torch.ones(3,32,32))
        torch.manual_seed(seed); np.random.seed(seed)
        b=ns['Cutout']()(torch.ones(3,32,32))
        assert torch.equal(a,b)


def test_worker_python_numpy_torch_seeding():
    from data import verify_worker_rng
    torch.manual_seed(0)
    before=torch.get_rng_state().clone()
    receipt=verify_worker_rng()
    assert receipt['status']=='PASS'
    assert torch.equal(before,torch.get_rng_state())
