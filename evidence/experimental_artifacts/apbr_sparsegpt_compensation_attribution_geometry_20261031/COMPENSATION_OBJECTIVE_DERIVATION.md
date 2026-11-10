# Fixed-Mask Compensation Objective

Let a single output row have dense weight `w`, local Gram/Hessian proxy `H`, retained coordinates `K`, and deleted coordinates `D`. The fixed support imposes

`Delta_D = -w_D`.

For the damped local quadratic objective

`1/2 Delta^T H_d Delta`, where `H_d = H + delta I`,

stationarity over the only free variables, `Delta_K`, gives

`H_d,KK Delta_K + H_d,KD Delta_D = 0`,

and therefore

`Delta_K = H_d,KK^{-1} H_d,KD w_D`.

The deleted coordinates are constraints, not optimization variables. They must not receive a ridge term or be reactivated.

## APBR geometry regularization

The checkpoint stores `s_i`, the log of the positive relative coordinate scale:

`a_i = r_i = exp(s_i)`.

It is a persistent relative geometry state. It is not an independently realized perturbation length for each coordinate; the actual training perturbation is globally normalized. To remove a common row-wise scale ambiguity, the pilot uses

`s_tilde_i = s_i - mean_{j in K}(s_j)` and `r_tilde_i = exp(s_tilde_i)`.

The geometry-regularized retained-coordinate objective is

`1/2 Delta^T H_d Delta + 1/2 lambda Delta_K^T R_tilde_K^{-2} Delta_K`,

which yields

`Delta_K = (H_d,KK + lambda R_tilde_K^{-2})^{-1} H_d,KD w_D`.

Because APBR is coordinate-wise, each output row generally has a different retained set and a different diagonal geometry penalty. A shared factorization across all output rows is therefore not generally available.

## Dimensionless strength and controls

The pilot defines

`lambda_row = alpha * mean(diag(H_d,KK))`.

Thus `alpha` is dimensionless and the regularizer follows the local Hessian scale. The isotropic control replaces `R_tilde_K^{-2}` with `I` at the same `lambda_row`. A fixed-seed shuffled control permutes the APBR diagonal within the same output row while preserving its distribution.

At `alpha=0`, standard, isotropic, geometry, and shuffled-geometry systems are identical. This is an exact implementation gate, not a performance expectation.

This solver is a row-wise fixed-support quadratic optimizer. It must not be called native SparseGPT, whose support selection and sequential OBS updates are interleaved.
