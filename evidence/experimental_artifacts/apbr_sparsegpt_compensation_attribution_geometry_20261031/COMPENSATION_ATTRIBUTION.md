# SparseGPT Compensation Attribution

Status: **Mask-only checkpoints frozen; GPU Final Test pending.**

The comparison is defined as the accuracy and CE difference between two checkpoints sharing the same Dense parent and the same final mask:

- Mask-only: `M_GPT * W_dense`.
- Native: the saved SparseGPT/Radius-Guided endpoint containing its compensated survivor weights.

The resulting difference will be called the **native compensation-weight benefit under the fixed final mask**. It is not a full causal decomposition of dynamic SparseGPT, because the final mask itself was selected during an interleaved compensation trajectory.

Known native endpoints on APBR parents:

| Seed | Mask | Native Accuracy | Native CE |
|---:|---|---:|---:|
| 0 | Original SparseGPT | 76.99% | 0.85810 |
| 0 | Radius-Guided SparseGPT | 76.86% | 0.85793 |
| 1 | Original SparseGPT | 76.78% | 0.87169 |
| 1 | Radius-Guided SparseGPT | 76.81% | 0.87216 |

The two masks differ at 1.309% of eligible coordinates in Seed 0 and 1.260% in Seed 1. Retained-coordinate native update RMS is nearly unchanged: 0.0040867 versus 0.0040877 in Seed 0, and 0.0041141 versus 0.0041126 in Seed 1 (Original versus Guided).

Mask-only Accuracy/CE and compensation increments remain intentionally blank until a real GPU Final Test is available.
