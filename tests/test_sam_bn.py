from copy import deepcopy
import random
import numpy as np
import pytest
import torch
from torch import nn
from nmsam.core.batch import StepBatch
from nmsam.core.loss import classification_loss
from nmsam.sam.reference_sgd import create_optimizer
from nmsam.sam.train_step import train_step
from nmsam.utils.seed import capture_rng, restore_rng


@pytest.mark.parametrize("momentum", [0.1, None])
@pytest.mark.parametrize("tracking", [True, False])
def test_bn_first_update_second_frozen_batch_statistics_and_modes(momentum, tracking):
    model = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4, momentum=momentum,
                         track_running_stats=tracking), nn.Tanh(), nn.Linear(4, 3)).double()
    model.eval()
    model[0].train()  # mixed modes must be restored
    modes = [m.training for m in model.modules()]
    batch = StepBatch(torch.randn(6, 4, dtype=torch.float64), torch.tensor([0, 1, 2, 0, 1, 2]))
    ref = deepcopy(model).train()
    classification_loss(ref, batch)  # exactly one reference running update
    initial = {k: b.clone() for k, b in model.named_buffers()}
    seen = []
    def hook(bn, inputs, output):
        seen.append((bn.training, bn.track_running_stats,
                     {k: v.clone() for k, v in bn.named_buffers()}))
        x = inputs[0]
        expected = (x-x.mean(0)) / torch.sqrt(x.var(0, unbiased=False)+bn.eps)
        expected = expected*bn.weight + bn.bias
        torch.testing.assert_close(output, expected)
    handle = model[1].register_forward_hook(hook)
    train_step(model, create_optimizer(model.parameters()), batch, classification_loss)
    handle.remove()
    assert seen[0][:2] == (True, tracking)
    assert seen[1][:2] == (True, False)
    for name, value in model.named_buffers():
        torch.testing.assert_close(value, dict(ref.named_buffers())[name], rtol=0, atol=0)
    if tracking:
        assert model[1].num_batches_tracked.item() == 1
        # Second pass starts from ORIGINAL buffers and never modifies them.
        for name, value in seen[1][2].items():
            assert torch.equal(value, initial["1."+name])
    assert [m.training for m in model.modules()] == modes
    assert model[1].track_running_stats == tracking


def test_same_materialized_batch_and_rng_replay_with_dropout():
    model = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.4), nn.Linear(4, 3))
    batch = StepBatch(torch.randn(6, 4), torch.tensor([0, 1, 2, 0, 1, 2]))
    initial_rng = capture_rng()
    masks, batches, randoms = [], [], []
    def hook(module, inputs, output):
        masks.append(output == 0)
    handle = model[1].register_forward_hook(hook)
    def loss(m, b):
        batches.append((id(b), b.inputs.data_ptr(), b.targets.data_ptr(), b.inputs.clone()))
        randoms.append((random.random(), np.random.random(), torch.rand(3)))
        return classification_loss(m, b)
    train_step(model, create_optimizer(model.parameters()), batch, loss)
    after = capture_rng()
    handle.remove()
    assert len(batches) == 2 and batches[0][:3] == batches[1][:3]
    assert torch.equal(batches[0][3], batches[1][3])
    assert torch.equal(masks[0], masks[1])
    assert randoms[0][:2] == randoms[1][:2]
    assert torch.equal(randoms[0][2], randoms[1][2])
    # Success consumes exactly one forward's RNG, not two.
    restore_rng(initial_rng)
    loss(model, batch)
    expected = capture_rng()
    assert torch.equal(after.cpu, expected.cpu)
    assert after.python == expected.python
    assert after.numpy[2:] == expected.numpy[2:]
    assert np.array_equal(after.numpy[1], expected.numpy[1])

