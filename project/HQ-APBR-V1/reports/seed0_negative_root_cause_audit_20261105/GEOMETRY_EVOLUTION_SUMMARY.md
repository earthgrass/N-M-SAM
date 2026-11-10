# Geometry evolution summary

`geometry_evolution_summary.csv` contains one row per Epoch for Q distribution summaries, training CE, geometry drift and layer-energy concentration. `geometry_checkpoint_snapshots.csv` contains exact full-tensor `s` statistics at the six retained checkpoints.

| Epoch index after refresh | Mean(s) | Std(s) | Min(s) | Max(s) |
|---:|---:|---:|---:|---:|
| 0 | 8.61e-18 | 0.0411 | -0.1342 | 0.0860 |
| 9 | 9.99e-17 | 0.3056 | -1.3803 | 0.7143 |
| 49 | 1.50e-16 | 0.4948 | -1.9677 | 1.8970 |
| 99 | 1.03e-16 | 0.4960 | -1.8847 | 2.0014 |
| 149 | 1.52e-16 | 0.5135 | -1.8708 | 2.0726 |
| 199 | 2.01e-16 | 0.5610 | -1.8848 | 2.1074 |

The near-zero means verify the global centering invariant. Geometry spread rises rapidly during early training and continues to broaden more slowly afterward.

Across 200 Epochs, geometry-drift RMS and center training CE have Pearson correlation 0.858 and Spearman correlation 0.950. Both fall strongly with training progress, so the correlation should be read as a shared time trend rather than causal evidence.

The final historical geometry closely tracks current Q ordering: the median per-layer Spearman correlation between `log(Q)` and `s` is 0.99948 (range 0.96756–0.99995). The history mechanism is therefore active and numerically coherent.

Perturbation energy is not uniformly spread. Mean top-one and top-three eligible-layer shares of total energy are 28.42% and 63.49%; at Epoch 199 they are 36.88% and 72.38%. The final three dominant layers are `layer4.1.conv2.weight`, `layer4.0.conv2.weight`, and `layer3.1.conv2.weight`.

The following requested evidence is absent: per-coordinate perturbation energy, batchwise perturbation vectors, and ordinary-SAM perturbation vectors. Coordinate-level alignment A and direction cosine are therefore `NOT AVAILABLE`.

