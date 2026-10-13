from copy import deepcopy
import pytest
import torch
from nmsam.models.tiny_mlp import TinyMLP
from nmsam.experiment.training import BaselineTrainer
from nmsam.experiment.checkpoint import save_checkpoint,load_checkpoint
from nmsam.data.sampler import StatefulBatchSampler


def test_malformed_sampler_checkpoint_rolls_back_all_loaded_state(tmp_path):
    trainer=BaselineTrainer(TinyMLP(),"2:4-SAM-Magnitude")
    sampler=StatefulBatchSampler(8,4);metadata={"git_sha":"unit","config_sha256":"cfg","split_sha256":"split"}
    path=tmp_path/"bad.pt";save_checkpoint(path,trainer,sampler,metadata)
    saved=torch.load(path,weights_only=False)
    for tensor in saved["model"].values():
        if tensor.is_floating_point():tensor.add_(2)
    saved["sampler"]["cursor"]=100;torch.save(saved,path)
    before=deepcopy(trainer.model.state_dict());rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError,match="sampler"):load_checkpoint(path,trainer,sampler,metadata)
    for key,value in trainer.model.state_dict().items():assert torch.equal(value,before[key])
    assert trainer.step==0 and sampler.cursor==0 and torch.equal(torch.get_rng_state(),rng)
