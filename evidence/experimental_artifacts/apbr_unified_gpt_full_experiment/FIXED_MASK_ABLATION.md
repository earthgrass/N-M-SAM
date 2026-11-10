# Fixed-Mask Geometry Compensation Pilot

Pilot: APBR Seed 0, `layer1.0.conv1`, Original SparseGPT Mask, 1,024 calibration Train images plus 1,024 disjoint Train diagnostics.

No compensation has diagnostic reconstruction MSE `1.4056e-3`. Standard fixed-mask quadratic compensation reduces it to `8.2865e-6` (about 170× lower).

Every positive geometry strength worsens reconstruction relative to the `lambda=0` standard solution. APBR geometry is also slightly worse than isotropic ridge at the same strength:

| alpha | Isotropic diagnostic MSE | APBR-Q diagnostic MSE | Shuffled-Q diagnostic MSE |
|---:|---:|---:|---:|
| 1e-4 | 8.2975e-6 | 8.2982e-6 | 8.2980e-6 |
| 1e-3 | 8.3950e-6 | 8.4013e-6 | 8.3991e-6 |
| 1e-2 | 9.2420e-6 | 9.2994e-6 | 9.2745e-6 |
| 1e-1 | 1.4525e-5 | 1.4949e-5 | 1.4724e-5 |
| 1 | 4.3108e-5 | 4.5164e-5 | 4.3905e-5 |

All Cholesky solves succeeded; maximum relative residual stayed below `3.55e-6`; deleted weights remained exactly zero. Thus APBR geometry changes the compensation direction, but the learned coordinate alignment does not outperform ordinary L2 shrinkage or shuffled geometry on this preregistered layer.
