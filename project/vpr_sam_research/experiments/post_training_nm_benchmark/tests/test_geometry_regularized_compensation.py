import torch


def _solve(h, w, keep, log_radius, alpha, geometry):
    k = torch.where(keep)[0]
    d = torch.where(~keep)[0]
    hkk = h[k][:, k]
    rhs = h[k][:, d] @ w[d]
    scale = torch.diag(hkk).mean()
    if geometry:
        centered = log_radius[k] - log_radius[k].mean()
        penalty = torch.exp(-2 * centered)
    else:
        penalty = torch.ones_like(k, dtype=h.dtype)
    system = hkk + alpha * scale * torch.diag(penalty)
    delta_k = torch.linalg.solve(system, rhs)
    delta = torch.zeros_like(w)
    delta[d] = -w[d]
    delta[k] = delta_k
    return delta, system, rhs


def test_lambda_zero_matches_standard_fixed_mask_solution():
    torch.manual_seed(1)
    x = torch.randn(8, 20, dtype=torch.float64)
    h = x @ x.T + 0.1 * torch.eye(8, dtype=torch.float64)
    w = torch.randn(8, dtype=torch.float64)
    keep = torch.tensor([1, 0, 1, 0, 0, 1, 1, 0], dtype=torch.bool)
    s = torch.linspace(-1, 1, 8, dtype=torch.float64)
    iso, _, _ = _solve(h, w, keep, s, 0.0, False)
    geo, system, rhs = _solve(h, w, keep, s, 0.0, True)
    torch.testing.assert_close(iso, geo, rtol=0, atol=0)
    k = torch.where(keep)[0]
    torch.testing.assert_close(system @ geo[k], rhs)
    assert torch.equal(geo[~keep], -w[~keep])


def test_geometry_regularizer_changes_direction_beyond_isotropic_ridge():
    torch.manual_seed(2)
    x = torch.randn(8, 24, dtype=torch.float64)
    h = x @ x.T + 0.2 * torch.eye(8, dtype=torch.float64)
    w = torch.randn(8, dtype=torch.float64)
    keep = torch.tensor([1, 0, 1, 0, 0, 1, 1, 0], dtype=torch.bool)
    s = torch.tensor([-2.0, 0, 1.0, 0, 0, 0.5, -0.5, 0], dtype=torch.float64)
    iso, _, _ = _solve(h, w, keep, s, 0.1, False)
    geo, _, _ = _solve(h, w, keep, s, 0.1, True)
    assert not torch.allclose(iso[keep], geo[keep])
    assert torch.equal(iso[~keep], geo[~keep])
