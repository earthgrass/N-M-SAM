# Claim-to-evidence table

| ID | Claim | Classification | Evidence | Boundary |
|---|---|---|---|---|
| C1 | Transfer and local evidence are complete. | VERIFIED IMPLEMENTATION STATE | 351/351 files, 2.15 GB, both SHA256 audits PASS. | Does not validate scientific efficacy. |
| C2 | All endpoints share the same parent, Epoch, calibration IDs, config, eligible map and BN state. | VERIFIED IMPLEMENTATION STATE | `mask_compensation_endpoint_comparison.csv`; local checkpoint audit. | Pruning algorithms still differ. |
| C3 | Mask-only and Fixed OBS use exactly the same support. | VERIFIED IMPLEMENTATION STATE | Identical Mask SHA256 `5e5a...ea8c`; bitwise tensor equality. | Applies only to these two branches. |
| C4 | The +26.66 pp Mask-only→Fixed OBS gain is a same-support compensation gain. | SUPPORTED HYPOTHESIS | Same parent/support/BN/config, only fixed OBS changes survivors. | Single parent and single Seed. |
| C5 | Native GPT's +1.87 pp over Fixed OBS is purely better compensation. | REJECTED CLAIM | Native Mask differs; only 79.23% coordinate agreement. | Selection and dynamic path are confounded. |
| C6 | Native GPT reconstructs the frozen parent better layerwise. | OBSERVED MECHANISM BEHAVIOR | Native MSE lower in 19/19 layers; sum 0.002058 vs 0.003105. | Layer-MSE sum is not an end-to-end MSE. |
| C7 | Historical `s` tracks Q ranking. | VERIFIED MECHANISM BEHAVIOR | Per-layer Spearman median 0.99948. | Expected from EMA design; not proof of usefulness. |
| C8 | Energy concentrates in late layers. | OBSERVED MECHANISM BEHAVIOR | Final top-one 36.88%, top-three 72.38%; per-Epoch logs. | Coordinate-level concentration unavailable. |
| C9 | Energy/Mask alignment A is worse than random 0.5. | NOT AVAILABLE | Only per-layer energy was logged. | A requires per-coordinate energy; no value computed. |
| C10 | High Q coordinates are more often deleted by Native GPT. | OBSERVED ASSOCIATION | Top Q decile deletion 55.17%, bottom decile 44.93%. | Post-hoc; native scores use evolving weights. |
| C11 | Dense magnitude remains influential in Native support. | OBSERVED ASSOCIATION | Deleted mean absolute weight 0.00409 vs retained 0.00852. | Not a causal score decomposition. |
| C12 | Geometry drift causes training CE to fall. | INCONCLUSIVE | Correlation is high but both trend with Epoch. | No intervention or lagged causal design. |
| C13 | A software defect explains the negative result. | NOT SUPPORTED | Hash, BN, Mask legality, parent and numerical audits pass. | Undetected issues remain logically possible, but none are evidenced. |
| C14 | HC-SAM improves Native GPT over SAM/APBR. | NOT SUPPORTED | HC 76.59%; SAM 77.47/77.51; APBR 76.99. | Historical comparison is reference-only and one Seed. |
| C15 | HC-SAM should be expanded immediately to more Seeds. | NOT RECOMMENDED | No positive task signal; key alignment metric unavailable. | A bounded redesigned pilot may still be justified. |

