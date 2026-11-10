# Per-layer diagnostics

The complete machine-readable table is `per_layer_diagnostics.csv`. It contains, for all 19 eligible Conv2d layers:

- strict 2:4 coordinate and deletion counts;
- static/native and magnitude/native Mask agreement;
- Mask-only, fixed-OBS and Native physical reconstruction MSE;
- survivor compensation RMS and recorded compensation time;
- dead-feature count, absolute damping and the recorded damped diagonal-ratio proxy;
- final Q and final `s` statistics;
- Q/Mask and magnitude/Mask retrospective associations;
- mean, maximum and final-Epoch perturbation-energy shares.

## Largest Native reconstruction errors

| Rank | Layer | Native MSE | Fixed OBS MSE | Mask-only MSE |
|---:|---|---:|---:|---:|
| 1 | `layer3.0.conv1.weight` | 2.8496e-4 | 4.2848e-4 | 2.9984e-3 |
| 2 | `layer2.0.conv1.weight` | 2.7810e-4 | 5.2157e-4 | 2.9705e-3 |
| 3 | `layer4.0.conv1.weight` | 2.3274e-4 | 3.1013e-4 | 1.6579e-3 |
| 4 | `layer4.1.conv1.weight` | 1.9778e-4 | 2.7640e-4 | 1.7425e-3 |
| 5 | `layer2.1.conv1.weight` | 1.5637e-4 | 2.4727e-4 | 1.8652e-3 |

The top five form 55.88% of the unnormalized sum of Native layer MSE. This is a localization statistic, not a causal attribution and not an end-to-end output MSE.

## Largest static/native support disagreements

| Layer | Coordinate agreement | Identical-group rate |
|---|---:|---:|
| `layer4.1.conv2.weight` | 66.83% | 39.37% |
| `layer1.0.conv1.weight` | 72.18% | 52.51% |
| `layer2.0.conv1.weight` | 74.52% | 52.61% |
| `layer1.0.conv2.weight` | 76.64% | 56.23% |
| `layer1.1.conv1.weight` | 77.96% | 59.03% |

`layer4.1.conv2.weight` has the highest final perturbation-energy share and the lowest static/native support agreement, but its Native reconstruction MSE is small. This is useful evidence against reducing the failure to a single “largest-energy layer caused the error” explanation.

## Condition information

Exact spectral condition numbers were not saved. The CSV exposes the recorded `damped_diagonal_ratio_proxy` and explicitly marks `spectral_condition_number` as `NOT AVAILABLE`. The largest proxy is 1,736.84 in `layer1.0.conv1.weight`; this proxy must not be reported as a true matrix condition number.

