from copy import deepcopy
import torch
from torch import nn
from nmsam.models.tiny_mlp import TinyMLP
from nmsam.core.batch import StepBatch
from nmsam.experiment.training import BaselineTrainer
from nmsam.experiment.checkpoint import save_checkpoint,load_checkpoint
from nmsam.data.sampler import StatefulBatchSampler
from nmsam.utils.seed import seed_everything


def test_single_step_full_state_checkpoint_restore(tmp_path):
    seed_everything(5)
    x=torch.randn(12,1,4,4);y=torch.arange(12)%3
    factory=lambda:nn.Sequential(TinyMLP(),nn.Dropout(.2))
    trainer=BaselineTrainer(factory(),"2:4-SAM-Taylor-g2")
    sampler=StatefulBatchSampler(12,4,seed=9)
    indices=next(iter(sampler));trainer.train(StepBatch(x[indices],y[indices]))
    path=tmp_path/"resume.pt";meta={"git_sha":"test","config_sha256":"config","split_sha256":"split"}
    save_checkpoint(path,trainer,sampler,meta)
    expected=deepcopy(trainer.model.state_dict());expected_masks=trainer.registry.state_dict()
    rand=torch.rand(3)
    resumed=BaselineTrainer(factory(),"2:4-SAM-Taylor-g2");rs=StatefulBatchSampler(12,4,seed=9)
    load_checkpoint(path,resumed,rs,meta)
    for k,v in resumed.model.state_dict().items():assert torch.equal(v,expected[k])
    assert resumed.registry.version==expected_masks["version"] and resumed.step==1
    assert rs.cursor==4 and torch.equal(rs.order,sampler.order)
    for k,v in resumed.registry.masks.items():assert torch.equal(v,expected_masks["masks"][k])
    assert torch.equal(torch.rand(3),rand)
    assert resumed.scheduler.state_dict()==trainer.scheduler.state_dict()
    for p,q in zip(resumed.model.parameters(),trainer.model.parameters()):
        assert torch.equal(resumed.optimizer.state[p]["momentum_buffer"],trainer.optimizer.state[q]["momentum_buffer"])
