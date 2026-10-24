import copy,json
from pathlib import Path
import pytest
import torch
from torch import nn
import anchor
from state import checkpoint_payload,atomic_checkpoint,atomic_json

def test_dynamic_paired_anchor_identity_and_publication(tmp_path,monkeypatch,device):
    model=nn.Linear(2,2).to(device);opt=torch.optim.SGD(model.parameters(),lr=.05)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,200)
    history=[{'epoch':i,'steps':390*i,'test_accuracy':.9 if i in (8,9) else .8,
              'test_loss':.2,'train_loss':.1} for i in range(1,201)]
    config={'seed':1,'learning_rate':.05,'output':'sam','experiment':'SAM'}
    identity={'commit':'frozen','seed':1,'config':config,'initialization_hash':'initial','data_manifest_sha256':'data'}
    run=tmp_path/'artifacts/sam';run.mkdir(parents=True)
    payload=checkpoint_payload(model,opt,scheduler,8,history[:8],identity,None)
    digest=atomic_checkpoint(run/'best_test.pt',payload)
    atomic_json(run/'status.json',{'status':'completed','steps':78000})
    atomic_json(run/'checkpoint_hashes.json',{'best_test.pt':digest})
    atomic_json(run/'FINAL_AUDIT.json',{'status':'PASS','selected':{'best_test.pt':{'sha256':digest}}})
    (run/'epoch_metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in history))
    current=dict(config,experiment='VPR',output='vpr',anchor={'root':str(tmp_path),'run':'artifacts/sam'})
    monkeypatch.setattr(anchor,'source_freeze',lambda root:{'commit':'frozen'})
    _,rows,receipt=anchor.verify_anchor(current)
    assert receipt['seed']==1 and receipt['epoch']==8 and len(rows)==200
    wrong=dict(current,seed=2)
    with pytest.raises(ValueError):anchor.verify_anchor(wrong)
    with pytest.raises(ValueError):anchor.verify_anchor(dict(current,learning_rate=.1))
    atomic_json(run/'checkpoint_hashes.json',{'best_test.pt':'wrong'})
    with pytest.raises(ValueError):anchor.verify_anchor(current)
