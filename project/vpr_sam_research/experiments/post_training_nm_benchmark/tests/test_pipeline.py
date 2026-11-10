import copy,json
import numpy as np
import pytest,torch
from torch import nn
from benchmark import artifacts
from benchmark.calibration import approved_indices,deterministic_indices,index_sha,validate_indices
from benchmark.common import seed_all,tensor_sha
from benchmark.evaluation import evaluate,recalibrate_copy
from benchmark.nm import eligible_layers,validate_masks
from benchmark.obc import accumulate,prune_with_hessian
from benchmark.recovery import recover

class Tiny(nn.Module):
    def __init__(self):
        super().__init__();self.first=nn.Conv2d(3,4,1);self.bn=nn.BatchNorm2d(4)
        self.middle=nn.Conv2d(4,4,1,bias=False);self.last=nn.Linear(4,2)
    def forward(self,x):return self.last(self.middle(self.bn(self.first(x))).mean((2,3)))

def batches():
    # RandomSampler and random augmentation both consume the restored global RNG.
    x=torch.arange(12*3*4*4,dtype=torch.float32).reshape(12,3,4,4)/100
    y=torch.arange(12)%2
    ids=torch.randperm(12)
    for group in ids.split(4):yield x[group]+torch.rand(())*.01,y[group],group

def test_calibration_pending_and_invalid_train_ids(tmp_path):
    with pytest.raises(ValueError,match='pending'):approved_indices({'calibration':{'approved':False}})
    for ids in ([],[0,0],[50000],[True],[1.1],[-1]):
        with pytest.raises(ValueError):validate_indices(ids)
    ids=deterministic_indices(50000,8,0)
    assert ids==deterministic_indices(50000,8,0) and len(set(ids))==8
    path=tmp_path/'ids.json';path.write_text(json.dumps(ids),encoding='utf-8')
    preprocessing={'to_tensor':True,'normalize_mean':[0.,0.,0.],'normalize_std':[1.,1.,1.]}
    c={'data':{'normalize_mean':[0.,0.,0.],'normalize_std':[1.,1.,1.]},
       'calibration':{'approved':True,'samples':8,'index_file':str(path),'rounds':1,
                     'augmentation':'none','index_sha256':index_sha(ids),'dataset':'official_cifar10_train_50000',
                     'batch_size':4,'forward_passes_per_layer':2,'optimization_epochs':0,'random_seed':0,'preprocessing':preprocessing}}
    assert approved_indices(c)==ids
    c['calibration']['index_sha256']='wrong'
    with pytest.raises(ValueError,match='identity'):approved_indices(c)

def prepare():
    seed_all(0,2);model=Tiny().eval();initial=copy.deepcopy(model.state_dict())
    images=torch.randn(4,4,4,4);h,_=accumulate(model.middle,[(images,model.middle(images))])
    out,mask,_=prune_with_hessian(model.middle,h,parallel=2)
    model.middle.weight.data.copy_(out);masks={'middle.weight':mask}
    validate_masks(model,masks)
    for k in initial:
        if k!='middle.weight':assert torch.equal(initial[k],model.state_dict()[k])
    return model,masks

def test_separate_bn_and_native_artifact_integrity(tmp_path):
    model,masks=prepare();before=tensor_sha(model.state_dict())
    clone,info=recalibrate_copy(model,list(batches()),'cpu')
    assert info['batches']==3 and int(clone.bn.num_batches_tracked)==3
    assert tensor_sha(model.state_dict())==before
    assert tensor_sha(dict(model.named_parameters()))==tensor_sha(dict(clone.named_parameters()))
    assert not torch.equal(model.bn.running_var,clone.bn.running_var)
    a=evaluate(clone,list(batches()),'cpu');assert a['count']==12
    value={'identity':{'source':'locked'},'stage':'native_complete','model':model.state_dict(),'masks':masks}
    path=tmp_path/'native.pt';artifacts.save(path,value);artifacts.load(path,{'source':'locked'})
    with pytest.raises(ValueError,match='identity'):artifacts.load(path,{'source':'changed'})
    damaged=torch.load(path,weights_only=False);damaged['model']['last.bias'][0]+=1
    torch.save(damaged,path)
    with pytest.raises(ValueError,match='digest'):artifacts.load(path)

def test_recovery_resume_matches_uninterrupted_weights_lr_order_and_momentum(tmp_path):
    model,masks=prepare();initial=copy.deepcopy(model.state_dict())
    c={'device':'cpu','purpose':'synthetic_test','recovery':{'epochs':3,'learning_rate':.005,
       'momentum':.9,'weight_decay':.0005,'nesterov':False,'eta_min':0}}
    identity={'config':'synthetic-resume-test'};parent={'stage':'native_complete','sha256':'synthetic'}
    seed_all(5,2);full=Tiny();full.load_state_dict(initial)
    recover(full,masks,batches,batches,c,tmp_path/'full',identity,parent)
    seed_all(5,2);partial=Tiny();partial.load_state_dict(initial);calls=0
    def interrupted():
        nonlocal calls
        calls+=1
        if calls==2:raise RuntimeError('simulated process loss after healthy epoch1')
        return batches()
    with pytest.raises(RuntimeError,match='simulated'):
        recover(partial,masks,interrupted,batches,c,tmp_path/'resume',identity,parent)
    # Fresh process/model construction consumes RNG, then checkpoint restores it.
    seed_all(99,2);resumed=Tiny()
    recover(resumed,masks,batches,batches,c,tmp_path/'resume',identity,parent,resume=True)
    a=artifacts.load(tmp_path/'full/last.pt');b=artifacts.load(tmp_path/'resume/last.pt')
    assert a['model_sha256']==b['model_sha256'] and a['scheduler']==b['scheduler']
    assert a['best']==b['best'] and a['epoch']==b['epoch'] and a['steps']==b['steps'] and a['mask_sha256']==b['mask_sha256']
    assert a['rng']['python']==b['rng']['python'] and torch.equal(a['rng']['cpu'],b['rng']['cpu'])
    assert a['rng']['numpy'][0]==b['rng']['numpy'][0] and np.array_equal(a['rng']['numpy'][1],b['rng']['numpy'][1])
    assert a['rng']['numpy'][2:]==b['rng']['numpy'][2:]
    if a['rng']['cuda'] is None:assert b['rng']['cuda'] is None
    else:assert all(torch.equal(x,y) for x,y in zip(a['rng']['cuda'],b['rng']['cuda']))
    for ra,rb in zip(a['history'],b['history']):
        assert {k:v for k,v in ra.items() if k!='elapsed_seconds'}=={k:v for k,v in rb.items() if k!='elapsed_seconds'}
    for key,state in a['optimizer']['state'].items():
        assert torch.equal(state['momentum_buffer'],b['optimizer']['state'][key]['momentum_buffer'])
    validate_masks(resumed,masks)

def test_scope_rejects_bn_first_and_last_masks():
    model,masks=prepare();assert list(eligible_layers(model))==['middle']
    masks['first.weight']=torch.ones_like(model.first.weight,dtype=torch.bool)
    with pytest.raises(ValueError,match='Scope'):validate_masks(model,masks)
