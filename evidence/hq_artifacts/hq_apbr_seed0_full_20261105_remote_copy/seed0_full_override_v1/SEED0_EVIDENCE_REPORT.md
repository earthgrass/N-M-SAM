# HQ-APBR / HC-SAM Seed0 exploratory result

Original GPU performance gate FAIL is retained. Execution used explicit research override. No recovery.

| Endpoint | Accuracy % | CE |
|---|---:|---:|
| dense | 78.51 | 0.829257 |
| final_q_mask_only | 48.06 | 2.084161 |
| final_q_fixed_obs | 74.72 | 0.960037 |
| sparsegpt_native | 76.59 | 0.883123 |

Dense-to-native drop: 1.92 percentage points.
Train seconds: 9489.210; calibration seconds: 3429.830.
Historical SAM/APBR results require protocol audit before a formal ranking.
