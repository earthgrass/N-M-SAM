# Accuracy-only comparison draft — new baselines pending

Historical metrics are reused, not rerun. Percent, mean ± sample standard deviation; n=3 independent seeds.
Official Test participates in Best selection. Native compensated outputs will be distinguished from unmodified Raw Hard.

| Method | Dense Acc (%) | Native post-pruning / Raw Hard Acc (%) | Recovery Best Acc (%) |
|---|---:|---:|---:|
| Dense-SAM + Magnitude | 96.7867 ± 0.0764 | 86.3133 ± 2.1600 | 96.2467 ± 0.1301 |
| APBR training + Magnitude (ablation) | 96.7433 ± 0.0723 | 88.5633 ± 2.2491 | 96.3767 ± 0.1320 |
| APBR-SAM (Full VPR-L2 + Geometry-guided) | 96.7433 ± 0.0723 | 89.5433 ± 1.4521 | 96.4167 ± 0.1012 |
| Dense-SAM + OBC (controlled layout adapter) | existing same parents | PENDING GPU/calibration | PENDING |
| Dense-SAM + SNOWS | existing same parents | BLOCKED official source | BLOCKED |

Final, Loss, best epochs, Mask/parent hashes and per-seed paired confidence intervals remain in the accompanying historical CSV/statistics JSON.
Existing intervals are exploratory and based on only three independent seeds; no SOTA or stable superiority conclusion is drawn.
No external published accuracy is merged into this common-protocol table.
