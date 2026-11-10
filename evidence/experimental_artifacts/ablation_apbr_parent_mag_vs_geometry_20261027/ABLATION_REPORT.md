# APBR-SAM Same-Parent Ablation: Magnitude vs Geometry

Audit status: **PASS**. Interpretation case: **B**.

Both branches use the same APBR-SAM Epoch-200 parent per seed, identical eligible layers and OIHW 2:4 grouping, and the same deterministic Recovery-20 schedule.

## Results

| Seed | Dense | Mag Raw | Geo Raw | Δ Raw (Geo−Mag) | Mag Rec-20 | Geo Rec-20 | Δ Rec (Geo−Mag) | Flip rate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 78.68% | 58.94% | 59.22% | +0.28 pp | 78.67% | 78.72% | +0.05 pp | 0.05% |
| 1 | 79.08% | 50.12% | 51.87% | +1.75 pp | 79.21% | 79.15% | -0.06 pp | 0.05% |
| 2 | 79.03% | 54.72% | 56.14% | +1.42 pp | 78.70% | 78.71% | +0.01 pp | 0.05% |

Raw paired difference: **+1.15 ± 0.77 pp** (sample SD).
Recovery-20 paired difference: **+0.00 ± 0.06 pp** (sample SD).
Raw CE difference (Geometry−Magnitude): **-0.0490 ± 0.0295**.
Recovery CE difference: **-0.0002 ± 0.0001**.
Mask flip rate: **0.05% ± 0.00 pp**.

## Compute cost

| Seed | Magnitude total | Recovery-20 | Peak allocated | Peak reserved |
|---:|---:|---:|---:|---:|
| 0 | 166.94 s | 165.70 s | 0.778 GiB | 1.010 GiB |
| 1 | 166.82 s | 165.54 s | 0.778 GiB | 1.010 GiB |
| 2 | 166.25 s | 164.96 s | 0.778 GiB | 1.010 GiB |

## Mechanism and limits

The frozen geometry proxy cost is reported as the ratio of summed deleted `|w| exp(-s)` under Geometry versus Magnitude. This is a deterministic mechanism diagnostic and partly tautological because Geometry directly optimizes that ranking; it is not independent evidence of test accuracy.

All 20 recovery train-order hashes match per seed. Crop/flip equivalence follows from the shared deterministic DataLoader generator and worker-seed implementation, although individual augmentation decisions were not serialized.

Seed 0 Geometry Recovery was originally executed on an RTX 4080 SUPER, whereas the new Magnitude branch used the RTX PRO 6000 Blackwell. Seeds 1 and 2 used Blackwell for both branches. The frozen software/data schedule is matched, but Seed 0's near-zero Recovery difference should not be treated as hardware-independent numerical equivalence.

Three paired seeds quantify repeatability but do not establish statistical significance.
