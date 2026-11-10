# APBR Dense Weights + Original SparseGPT — Audit Report

Status: **SEEDS 0 AND 1 COMPLETE; SEED 2 BLOCKED**

## Results

| Seed | Dense-SAM + SparseGPT | APBR-SAM + SparseGPT | APBR - SAM | SAM drop | APBR drop |
|---:|---:|---:|---:|---:|---:|
| 0 | 77.51% | 76.99% | -0.52 pp | 1.33 pp | 1.69 pp |
| 1 | 77.56% | 76.78% | -0.78 pp | 1.41 pp | 2.30 pp |

Two-seed mean: Dense-SAM + SparseGPT **77.53%**, APBR-SAM + SparseGPT **76.89%**. Paired APBR-SAM minus Dense-SAM: **-0.65 +/- 0.18 pp** (sample SD).

## Algorithm audit

All four runs loaded only the dense `model` state. APBR stepper geometry, APBR score, prior masks, and fixed-mask OBS were not consumed. The adapter dynamically selected 2:4 deletion coordinates from the compensated current block and performed Cholesky/inverse-Hessian sequential OBS updates. Every run used the same 1,024 official Train IDs, FP32, damping 0.01, block size 128, 19 eligible Conv2d layers, 11,157,504 eligible positions, and 5,578,752 deletions. Deleted weights were exactly zero; all outputs were finite. No Recovery, BN-only update, gradient training, Test tuning, or automatic retry occurred.

The maximum-layer numerical gate passed all algorithm/numerical checks. Its legacy same-process free-memory criterion reported FAIL because the CUDA context retained memory; independent post-process observation showed zero compute processes and 0 MiB GPU allocation. Both records are preserved.

## Interpretation boundary

For Seeds 0 and 1, APBR-trained dense weights produced lower Original SparseGPT final accuracy and larger pruning drops than Dense-SAM weights. This is a two-seed partial result, not a complete three-seed or statistically significant conclusion. Seed 2 remains blocked by the missing exact APBR dense parent.
