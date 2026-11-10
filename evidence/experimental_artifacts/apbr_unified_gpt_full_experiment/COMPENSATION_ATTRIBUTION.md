# SparseGPT Compensation Attribution

The counterfactual is `W_mask-only = M_final * W_dense`, using the matching Dense parent and frozen final Mask with unchanged BN state. The difference to Native is called the **fixed-final-Mask native compensation-weight benefit**; it is not a complete causal decomposition of the interleaved SparseGPT trajectory.

| Parent | Seed | Mask | Mask-only Acc. | Native Acc. | Compensation gain |
|---|---:|---|---:|---:|---:|
| APBR-SAM | 0 | Original GPT | 48.66% | 76.99% | +28.33 pp |
| APBR-SAM | 0 | Radius-Guided | 48.42% | 76.86% | +28.44 pp |
| APBR-SAM | 1 | Original GPT | 39.13% | 76.78% | +37.65 pp |
| APBR-SAM | 1 | Radius-Guided | 39.50% | 76.81% | +37.31 pp |
| Dense-SAM | 0 | Original GPT | 47.40% | 77.51% | +30.11 pp |
| Dense-SAM | 1 | Original GPT | 46.00% | 77.56% | +31.56 pp |

For APBR parents, Original GPT compensation gain is `32.99 ± 6.59` points over two seeds; Radius-Guided gain is `32.88 ± 6.27` points. The magnitude and CE reductions show that the Native endpoint is dominated by compensated survivor weights rather than the final support projected onto the untouched Dense weights.

Radius-Guided versus Original Mask-only changes are inconsistent: -0.24 points in Seed 0 and +0.37 in Seed 1. Native changes are likewise small and inconsistent: -0.13 and +0.03 points. The learned radius changes about 1.3% of eligible mask coordinates but does not establish a repeatable benefit.
