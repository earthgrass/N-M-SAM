import torch

from baseline_adapters.apbr_gpt import apbr_gpt_matrix, normalized_inverse_square
from baseline_adapters.sparsegpt import sparsegpt_matrix


def _case():
    torch.manual_seed(1031)
    w = torch.randn(3, 12, dtype=torch.float64)
    x = torch.randn(12, 40, dtype=torch.float64)
    h = 2 * x @ x.T / x.shape[1]
    s = torch.randn_like(w) * 0.2
    q, mean = normalized_inverse_square({"layer": s})
    return w, h, q["layer"], mean


def test_global_q_normalization_and_scale_invariance():
    w, _h, _q, _mean = _case()
    first, _ = normalized_inverse_square({"layer": w})
    second, _ = normalized_inverse_square({"layer": w + 3.0})
    torch.testing.assert_close(first["layer"], second["layer"], rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(first["layer"].mean(), torch.tensor(1.0, dtype=torch.float64))


def test_lambda_zero_is_exact_original_sparsegpt():
    w, h, q, _ = _case()
    original = sparsegpt_matrix(w, h, blocksize=8)
    result = apbr_gpt_matrix(w, h, q, lambda_value=0.0, blocksize=8)
    assert torch.equal(result.keep_mask, original.keep_mask)
    torch.testing.assert_close(result.weight, original.weight, rtol=0, atol=0)


def test_exact_and_shared_match_when_geometry_is_shared():
    w, h, q, _ = _case()
    shared_q = q.mean(0, keepdim=True).expand_as(q).clone()
    exact = apbr_gpt_matrix(w, h, shared_q, lambda_value=0.01, mode="exact_rowwise", blocksize=8)
    shared = apbr_gpt_matrix(w, h, shared_q, lambda_value=0.01, mode="shared_geometry", blocksize=8)
    assert torch.equal(exact.keep_mask, shared.keep_mask)
    torch.testing.assert_close(exact.weight, shared.weight, rtol=1e-12, atol=1e-12)


def test_nonconstant_geometry_changes_exact_q_across_rows_and_stays_finite():
    w, h, q, _ = _case()
    result = apbr_gpt_matrix(w, h, q, lambda_value=0.1, mode="exact_rowwise", blocksize=8)
    assert bool(torch.isfinite(result.weight).all())
    assert torch.count_nonzero(result.weight[~result.keep_mask]) == 0
