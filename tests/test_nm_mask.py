from itertools import combinations
import pytest
import torch
from nmsam.sparsity.layout import MaskSpec, MaskLayout
from nmsam.sparsity.mask import build_magnitude_mask, enumerate_masks, validate_mask, mask_delta


@pytest.mark.parametrize("shape", [(3, 8), (2, 2, 2, 2), (4, 1, 2, 4)])
def test_linear_conv_layout_roundtrip_and_exact_order(shape):
    w = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float64).reshape(shape)
    layout = MaskLayout(shape)
    groups = layout.groups(w)
    assert torch.equal(groups[0, 0], w[0].flatten()[:4])
    assert torch.equal(groups[-1, -1], w[-1].flatten()[-4:])
    assert torch.equal(layout.restore(groups), w)
    mask = build_magnitude_mask(w)
    validate_mask(mask, layout)
    assert (layout.groups(mask).sum(-1) == 2).all()
    assert torch.equal(layout.groups(mask)[0, 0], torch.tensor([False, False, True, True]))


def test_noncontiguous_layout():
    w = torch.arange(24., dtype=torch.float64).reshape(8, 3).t()
    assert not w.is_contiguous()
    layout = MaskLayout(tuple(w.shape))
    assert torch.equal(layout.groups(w)[0, 0], w[0, :4])
    assert torch.equal(layout.restore(layout.groups(w)), w)
    validate_mask(build_magnitude_mask(w), layout)


def test_six_candidates_complete_legal_unique_deterministic():
    masks = enumerate_masks()
    assert masks.shape == (6, 4) and masks.dtype == torch.bool
    expected = list(combinations(range(4), 2))
    assert [tuple(row.nonzero().flatten().tolist()) for row in masks] == expected
    for candidate in masks:
        validate_mask(candidate.reshape(1, 4), MaskLayout((1, 4)))
    assert torch.equal(masks, enumerate_masks())


def test_tie_break_repeated_zero_and_signed_ties_do_not_mutate_weights():
    w = torch.tensor([[1., -1., 1., -1., 0., 0., 0., 0.],
                      [2., 2., 3., 3., 1., 4., 4., 2.]])
    before = w.clone()
    expected = torch.tensor([[1, 1, 0, 0, 1, 1, 0, 0],
                             [0, 0, 1, 1, 0, 1, 1, 0]], dtype=torch.bool)
    for _ in range(5):
        assert torch.equal(build_magnitude_mask(w), expected)
    assert torch.equal(w, before)


@pytest.mark.parametrize("shape", [(3, 5), (3, 1, 3, 3), (4,), (0, 4)])
def test_strict_reject_tail_and_nonweight_shapes(shape):
    with pytest.raises(ValueError):
        build_magnitude_mask(torch.zeros(shape))


@pytest.mark.parametrize("values", [[1., 1., 1., 0.], [1., 0., 0., 0.],
    [1., 1., 0.5, 0.], [1., 1., float("nan"), 0.]])
def test_invalid_masks_rejected(values):
    with pytest.raises(ValueError):
        validate_mask(torch.tensor([values]), MaskLayout((1, 4)))


@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_nonfinite_weights_rejected(value):
    with pytest.raises(ValueError):
        build_magnitude_mask(torch.tensor([[value, 0., 1., 2.]]))


@pytest.mark.parametrize("spec", [(0, 4), (5, 4), (True, 4), (2, 0)])
def test_bad_nm_spec(spec):
    with pytest.raises(ValueError):
        MaskSpec(*spec)


def test_q_and_all_six_candidates_effective_displacement():
    w = torch.tensor([[4., 3., 2., 1.]])
    old = enumerate_masks()[0].reshape(1, 4)
    for candidate in enumerate_masks():
        new = candidate.reshape(1, 4)
        q = mask_delta(w, old, new)
        assert torch.equal(w*old+q, w*new)
    new = enumerate_masks()[1].reshape(1, 4)
    assert torch.equal(mask_delta(w, old, new), torch.tensor([[0., -3., 2., 0.]]))


def test_builder_only_selected_tensor_other_layers_remain_dense():
    from torch import nn
    model = nn.Sequential(nn.Linear(8, 4), nn.BatchNorm1d(4), nn.Linear(4, 3))
    before = {k: v.detach().clone() for k, v in model.named_parameters()}
    masks = {"0.weight": build_magnitude_mask(model[0].weight)}
    assert set(masks) == {"0.weight"}
    for name, p in model.named_parameters():
        assert torch.equal(p, before[name])  # builder never changes dense masters
        effective = p * masks[name] if name in masks else p
        if name not in masks:
            assert torch.equal(effective, p)

