# Mathematical and implementation audit

Status: `MATH_AUDIT_PASS` for the local CPU implementation.

## APBR path

The audited APBR geometry uses one global radius `rho=0.05`. For first gradient `g` and FP64 log-scale `s`, its log-domain implementation constructs a globally normalized perturbation proportional to `sign(g) exp(2s + log|g|)`. The normalization is global over every trainable coordinate, not per layer or per 2:4 group. The original energy feedback is `gamma(1-d*p_i)`, where `p_i` is each coordinate's share of global perturbation energy; the global log-volume constraint is preserved. Geometry tensors follow the model parameter mapping exactly and are stored in FP64.

## Joint training regularizer

For every output row and 2:4 group, all six deletion pairs are evaluated. `Q`, the activation Gram, and the detached first gradient are stop-gradient quantities. Model weights retain their gradient path. APBR probes retain a differentiable path to `s`, without differentiating through the first gradient or the Hessian inversion. The normalized six-candidate soft minimum is averaged by actual block count across layers.

The scale update remains the original APBR energy update plus the pre-registered pruning-only term:

`s_next = Pi(s + delta_energy_raw - eta_s * lambda_p * grad_s R_prune)`.

No CE gradient is silently applied to `s`. `lambda_p=0` degeneracy, Q stop-gradient, first-gradient stop-gradient, finite-loss behavior, projection, and calibration edge cases are covered by tests.

## Groupwise selector and compensation

The current implementation uses the exact two-coordinate constraint update described in `GROUPWISE_OBS_DERIVATION.md`. It does not select a group and then replay a sequential one-column approximation. Deleted weights and propagated probe coordinates are exactly zero; dense tails remain untouched; every complete group keeps exactly two coordinates.

## Evidence

The Phase-3 targeted suite passed 23 tests. These include the explicit SPD multi-coordinate reference, row-dependent supports, calibration failures and repeatability, formal schedule guards, atomic checkpoint roundtrip, Final-Test freeze guard, and an active/warm-up JointStepper transaction. Protected original APBR and SparseGPT sources were not modified.
