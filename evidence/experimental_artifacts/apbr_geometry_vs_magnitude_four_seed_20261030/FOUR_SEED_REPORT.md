# APBR-SAM Geometry vs Magnitude: Four-Seed Summary

All endpoints use the same APBR-SAM dense parent within each seed. Raw-Hard and fixed-mask Recovery-20 are reported separately.

| Seed | APBR Dense | Magnitude Raw | Geometry Raw | Magnitude Recovery-20 | Geometry Recovery-20 | Geo−Mag Raw (pp) | Geo−Mag Recovery (pp) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 78.68 | 58.94 | 59.22 | 78.67 | 78.72 | 0.28 | 0.05 |
| 1 | 79.08 | 50.12 | 51.87 | 79.21 | 79.15 | 1.75 | -0.06 |
| 2 | 79.03 | 54.72 | 56.14 | 78.70 | 78.71 | 1.42 | 0.01 |
| 3 | 78.61 | 62.55 | 63.04 | 78.87 | 78.78 | 0.49 | -0.09 |

## Mean ± sample SD

| Endpoint | Accuracy (%) or paired difference (pp) |
|---|---:|
| APBR Dense | 78.85 ± 0.24 |
| Magnitude Raw-Hard | 56.58 ± 5.37 |
| Geometry Raw-Hard | 57.57 ± 4.73 |
| Magnitude Recovery-20 | 78.86 ± 0.25 |
| Geometry Recovery-20 | 78.84 ± 0.21 |
| Geometry−Magnitude Raw | 0.98 ± 0.71 |
| Geometry−Magnitude Recovery | -0.02 ± 0.06 |

## Interpretation

Geometry is better than Magnitude at Raw-Hard in all four seeds. After Recovery-20, the two masks converge to nearly identical final accuracy, with differences changing sign across seeds.
Seed 3 independently satisfies the 19-layer, 11,157,504-position, 5,578,752-deletion strict 2:4 audit for both masks.
