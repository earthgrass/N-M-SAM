"""Minimal additional actual-model interface checks, not a SNOWS reproduction."""
import pytest,torch
from benchmark.model import build_model
from benchmark.snows_step_patch import load_step_candidate

def test_native_get_blocks_does_not_accept_actual_cifar18_without_fake_name():
    module=load_step_candidate();model=build_model().eval()
    with pytest.raises(AttributeError,match='name'):module.get_blocks(model)

def test_native_block_hooks_miss_functional_relu_and_projection_branch():
    module=load_step_candidate();block=build_model().eval().layer2[0]
    handles=[];outputs=[]
    count=module.register_hooks_on_basicblock(block,handles,outputs,cur_module='conv1',is_first_block=True,K=20)
    hooked=[n for n,m in block.named_modules() if m._forward_hooks]
    try:
        with torch.no_grad():result=block(torch.zeros(1,64,32,32))
        assert result.shape==(1,128,16,16)
        assert hooked==['conv1','conv2'] and count==len(outputs)==2
        assert isinstance(block.shortcut[0],torch.nn.Conv2d) and not block.shortcut[0]._forward_hooks
        assert not any(isinstance(m,torch.nn.ReLU) for m in block.modules())
    finally:
        for h in handles:h.remove()
