# Frozen Protocol

## Part A

For every frozen GPT mask, construct the counterfactual exactly as

`W_mask_only = M_GPT * W_dense`

using the matching Dense model and its original BN state. No scoring, compensation, training, or BN update is allowed. Each frozen endpoint receives one independent Final Test evaluation only after the checkpoint is published.

## Part D pilot

- Parent: APBR-SAM Seed 0 Dense Epoch 200.
- Mask: Seed-0 Radius-Guided SparseGPT final mask.
- Layer: `layer2.0.conv2`, selected before observing pilot outcomes.
- Calibration: the frozen 1,024 Train IDs selected with seed `20261027`.
- Independent diagnostic set: 1,024 disjoint Train IDs selected with seed `20261031`.
- Hessian damping: the frozen SparseGPT rule, `0.01 * mean(diag(H))`.
- Candidate dimensionless strengths: `alpha in {0, 1e-3, 1e-2, 1e-1}`; no Final Test is used.
- Absolute row-wise regularization: `lambda_row = alpha * mean(diag(H_KK,damped))`.
- Geometry normalization: `s_tilde_i = s_i - mean_{j in K}(s_j)` per output row and `R_tilde = diag(exp(s_tilde))`.
- Controls: no compensation, standard fixed-mask quadratic solve, isotropic ridge, APBR geometry ridge, and a fixed-seed within-row shuffled-geometry diagnostic.
- Stop after this single layer. No full-network geometry compensation is authorized by this protocol.
