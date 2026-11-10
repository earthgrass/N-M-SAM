import torch
from torch import nn


def test_mask_only_is_dense_weight_projection_without_bn_or_survivor_changes():
    torch.manual_seed(31)
    model = nn.Sequential(nn.Conv2d(4, 3, 1, bias=False), nn.BatchNorm2d(3))
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    mask = torch.tensor([1, 0, 1, 0], dtype=torch.bool).reshape(1, 4, 1, 1).expand_as(model[0].weight)
    with torch.no_grad():
        model[0].weight.mul_(mask)
    assert torch.equal(model[0].weight[mask], before["0.weight"][mask])
    assert torch.count_nonzero(model[0].weight[~mask]) == 0
    for key in ("1.weight", "1.bias", "1.running_mean", "1.running_var", "1.num_batches_tracked"):
        assert torch.equal(model.state_dict()[key], before[key])
