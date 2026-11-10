# APBR-GPT Method Derivation

The frozen Conv2d SparseGPT adapter accumulates

`H_raw = 2 X X^T / n_vectors`.

The implementation retains this exact normalization. Dead coordinates and the original relative damping are handled exactly as in the frozen adapter, producing

`H_base = H_raw + D_original`.

Let

`mu_H = trace(H_raw) / d`

and let `q_o` be the globally normalized inverse-square APBR geometry for output row `o`. APBR-GPT defines

`Q_o(lambda) = H_base + lambda * mu_H * diag(q_o)`.

For `lambda=0`, Q equals the original damped second-order matrix. The implementation directly dispatches to frozen Original SparseGPT at zero, giving an exact regression gate.

## Fixed-mask compensation

For retained coordinates K and deleted coordinates D, `Delta_D=-w_D`. The APBR-Q fixed-mask solution is

`Q_KK Delta_K = H_base,KD w_D`.

The right-hand side contains `H_base,KD`; the APBR term is diagonal and has no K-D cross term. Cholesky solves are required; no explicit inverse is used.

## Unified dynamic APBR-GPT

Unified QQ replaces the factor used by both dynamic N:M deletion scoring and sequential OBS error propagation with the factor derived from Q. All grouping, block traversal, dense-tail, exclusions, stable ties, and weight-error propagation remain frozen.

Because `q_o` differs by output row, Exact APBR-GPT requires one Q factorization per output row. Shared Geometry is a labelled approximation using `mean_o(q_o)` and one factorization per layer.

Radius-Guided SparseGPT remains a distinct method: it multiplies the dynamic deletion score by `a_i^{-1}` but continues to use H for OBS. It is not Q-based APBR-GPT.
