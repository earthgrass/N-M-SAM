# APBR Geometry Definition Audit

The checkpoint stores `s_i`, not a realized per-coordinate perturbation length. The positive coordinate scale is

`a_i = exp(s_i) > 0`.

Evidence is consistent across the paper Method, `Geometry.perturbations`, checkpoint schema, and Geometry Mask implementation:

- `s` is initialized at zero and saved as one tensor per trainable parameter tensor.
- Every parameter coordinate has an independent `s_i`; tensor shapes and parameter-name order are checkpointed and validated on restore.
- The dense perturbation uses `A = diag(exp(s))` inside a globally normalized perturbation. Therefore `a_i` is a relative anisotropic scale, not an independent absolute radius; the realized perturbation still obeys the shared global budget `rho`.
- The update preserves the global log-volume in exact arithmetic: `sum_i s_i` is invariant over all trainable coordinates. It does not impose per-layer, per-row, or per-block normalization.
- The original pruning score is `|w_i| exp(-s_i) = |w_i|/a_i`; larger `a_i` reduces the original score at fixed weight.
- Eligible Conv2d geometry tensors exactly match OIHW weights. Both are mapped to SparseGPT matrices by output row followed by contiguous `(input channel, kernel height, kernel width)` coordinates.
- BN affine, bias, first Conv2d, and final Linear have geometry states from training but are excluded from the 19-layer pruning set.

For Q-based APBR-GPT the inverse-square geometry is

`q_i = a_i^{-2} = exp(-2s_i)`.

Only one eligible-coordinate scalar normalization is allowed:

`q_normalized_i = q_i / mean_{all 19 eligible Conv2d coordinates}(q)`.

No layer-, row-, or block-wise recentering is used in this project.
