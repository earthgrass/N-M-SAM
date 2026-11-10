# HC-SAM Seed 0 Negative Result Root-Cause Audit

## Audit scope and evidence state

This is a local, read-only audit of the completed Seed 0 run. It did not open CIFAR-100, repeat Test evaluation, run training, alter a checkpoint, or change a hyperparameter. The local transfer contains 351 files and 2,150,679,355 bytes; the independent transfer manifest and the original run manifest both pass SHA256 verification. The formal local checkpoint/Mask audit status is `COMPLETED_SEED0_LOCAL_AUDIT_PASS`.

The fixed parent is Epoch index 199 (200 completed Epochs, 78,200 optimizer steps), SHA256 `5bf3011c4fc15c999c6dc62f6a2d6b7a2ffebcd6931145367f58e8fbee5fdbdc`. The configuration SHA256 is `56ff197b0860dfd0f4a0768c5609031214e6405c51557009a9977582b38400e7`.

## Executive finding

No **VERIFIED IMPLEMENTATION ISSUE** was found in the frozen evidence. All four endpoints use the same dense parent, checkpoint epoch, calibration IDs, pruning configuration, eligible-layer map, and BN state. All three sparse endpoints cover 19 eligible Conv2d layers, 11,157,504 coordinates and 5,578,752 deletions under strict 2:4.

The negative result is best classified as **OBSERVED MECHANISM BEHAVIOR**, with two distinct components:

1. HC-SAM itself ends at 78.51%, 0.33 pp below the audited SAM Seed 0 reference (78.84%). This historical comparison is reference-only because the execution hardware and complete run identity differ.
2. Native SparseGPT loses 1.92 pp from the HC-SAM parent, versus 1.37 pp for the audited formal SAM reference and 1.69 pp for the APBR reference. Thus the HC parent does not show improved native GPT compensability in this Seed.

The static Final-Q support is especially weak. Fixed-mask OBS restores Mask-only accuracy from 48.06% to 74.72%, proving that compensation is effective, but Native SparseGPT reaches 76.59%. The remaining 1.87 pp is not a pure “compensation gain”: the two methods use different supports and selection paths.

## Endpoint identity and isolation

| Endpoint | Accuracy | CE | Mask identity | Interpretation |
|---|---:|---:|---|---|
| Dense | 78.51% | 0.829257 | none | Fixed Epoch-199 parent |
| Final-Q Mask-only | 48.06% | 2.084161 | `5e5a...ea8c` | Static Final-Q support, no compensation |
| Final-Q + Fixed-mask OBS | 74.72% | 0.960037 | `5e5a...ea8c` | Same support as Mask-only; sequential fixed-support OBS |
| Native SparseGPT | 76.59% | 0.883123 | `6d9c...18f0` | Dynamic support selection plus native compensation |

Mask-only and Fixed-mask OBS are bitwise support-identical. Their 26.66 pp accuracy difference can therefore be attributed to the fixed-mask OBS weight update under this frozen run. Native SparseGPT uses a different Mask: static/native coordinate agreement is 79.23%, keep-support Jaccard is 65.60%, and only 60.66% of complete 2:4 groups choose exactly the same pair. The two sparse endpoints cannot be compared as if only the compensation weights changed.

The BN-state SHA256 is identical for all four endpoints: `263e76f336191cf41addbf0c66688704b60a3b9ba3a1c9c7c4a04179df6c0444`. The calibration-ID SHA256 is `3b7da27d79762235c87f18571bc73224d6e6f2b3ba663b7d9a4a3de312f23157`. Relative damping is 0.01, block size is 128, and compute dtype is FP32 for the pruning paths.

One intentional semantic difference is relevant. Final-Q support selection uses the undoubled Gram with ordinary relative damping and no dead-coordinate replacement; both OBS paths use the historical `2H` convention and replace dead diagonals before damping. There are 369 dead input features across five layers. This is not an accidental implementation divergence: it is explicit in the frozen source and logs. It may contribute to support/compensation mismatch in those five layers, but it cannot explain the whole gap because Native reconstruction MSE is lower in all 19 layers, including layers without dead features.

## Geometry behavior

Per-Epoch Q summaries, geometry drift, and per-layer perturbation-energy shares are available for all 200 Epochs. Full `s` tensors are available only at Epoch indices 0, 9, 49, 99, 149 and 199.

The final geometry remains globally centered, while its spread grows substantially: population standard deviation rises from 0.0411 after the Epoch-0 refresh to 0.5610 after the Epoch-199 refresh; the final range is [-1.8848, 2.1074]. The unique Q-diagonal distribution moves from min/mean/max 8.33/149.89/681.05 at Epoch 0 to 29.24/3,474.31/85,906.95 at Epoch 199.

Final `log(Q)` and historical `s` are strongly aligned, as expected from the EMA construction: per-layer Spearman correlation has median 0.99948 and range [0.96756, 0.99995]. This verifies that the intended history tracker follows Q rankings. It does **not** show that those rankings align with the final GPT support or improve accuracy.

Geometry-drift RMS and center training CE have Pearson 0.858 and Spearman 0.950 correlation across Epochs. This means both are largest early and decay during training. It is an association between same-Epoch CE and the post-Epoch refresh, not evidence that drift causes CE changes.

Perturbation energy becomes highly concentrated. Averaged over training, the largest eligible layer receives 28.42% of total perturbation energy and the top three receive 63.49%. At the final Epoch, `layer4.1.conv2.weight` alone receives 36.88%; the top three late `conv2` layers receive 72.38%. The evidence supports “late-layer concentration occurred,” but not “late-layer concentration caused the accuracy deficit.” Per-coordinate concentration cannot be assessed from the saved logs.

The cosine between HC-SAM and ordinary SAM perturbation directions is **NOT AVAILABLE**: neither batch gradients nor perturbation vectors were retained. It cannot be reconstructed from the final geometry alone.

## Compensability–Mask alignment

The requested statistic

\[
A=\frac{\text{eligible perturbation energy on coordinates deleted by Native GPT}}
        {\text{all eligible perturbation energy}}
\]

is **NOT AVAILABLE**. Training stored one energy share per layer per Epoch, whereas every eligible layer is pruned at exactly 50%. Without within-layer coordinate energy, any coordinate-level value of A would be fabricated. The random legal 2:4 expectation remains 0.5, but this run cannot be compared with it.

Post-hoc mask diagnostics do show that Q and magnitude both relate to Native support. Within each layer, Native GPT deletes 55.17% of coordinates in the top Q decile and 44.93% in the bottom Q decile. Deleted coordinates have mean dense magnitude 0.00409 versus 0.00852 for retained coordinates. These are retrospective associations: Native GPT selected with evolving compensated weights, so they are not a causal score decomposition.

The static Final-Q Mask is very close to a dense-parent magnitude Mask: 95.28% coordinate agreement and 90.62% identical 2:4 groups. Its agreement with Native GPT is only 0.50 pp higher than the magnitude Mask's coordinate agreement (79.23% versus 78.73%). This supports the bounded hypothesis that the static Q score changed relatively few support decisions beyond magnitude, while Native GPT's sequential dynamics produced a materially different support.

## Reconstruction error localization

Summing the 19 layerwise physical reconstruction MSE diagnostics gives 0.019684 for Mask-only, 0.003105 for fixed OBS and 0.002058 for Native GPT. These sums are diagnostic aggregates rather than a network-output MSE because layers have different activation scales. Native MSE is lower than fixed-mask OBS in 19/19 layers.

The five largest Native layer MSEs are:

| Layer | Native MSE | Fixed-mask OBS MSE |
|---|---:|---:|
| `layer3.0.conv1.weight` | 2.8496e-4 | 4.2848e-4 |
| `layer2.0.conv1.weight` | 2.7810e-4 | 5.2157e-4 |
| `layer4.0.conv1.weight` | 2.3274e-4 | 3.1013e-4 |
| `layer4.1.conv1.weight` | 1.9778e-4 | 2.7640e-4 |
| `layer2.1.conv1.weight` | 1.5637e-4 | 2.4727e-4 |

These five account for 55.88% of the unnormalized sum of Native layer MSE. The exact spectral condition number was not logged and is **NOT AVAILABLE**. The report therefore preserves the recorded damped diagonal-ratio proxy rather than relabeling it as a spectral condition number.

Historical SAM/APBR artifacts provide final accuracy, CE, parent identity and pruning configuration, but no matching per-layer reconstruction traces were found. Layerwise error comparisons with those runs are **NOT AVAILABLE**.

## Classified conclusions

### VERIFIED IMPLEMENTATION ISSUE

None found. Parent, epoch, config, calibration IDs, BN state, eligible mapping, 2:4 counts, zero constraints, and checkpoint hashes pass.

### OBSERVED MECHANISM BEHAVIOR

- Static and Native Masks differ on 2,317,382 boolean coordinates; only 60.66% of groups are identical.
- Fixed OBS recovers 26.66 pp on the static Mask, yet Native remains 1.87 pp better.
- Native GPT has lower physical reconstruction MSE in every eligible layer.
- Final-Q support is 95.28% coordinate-identical to magnitude support.
- Perturbation energy is concentrated in a few late layers.

### SUPPORTED HYPOTHESIS

- On this frozen parent, the static Final-Q support/compensation path is less effective than Native GPT's dynamic selection/compensation path.
- HC-SAM Seed 0 does not improve the Native SparseGPT endpoint over the audited historical SAM/APBR references.
- Historical `s` successfully tracks Q rankings, but this tracking alone did not yield improved post-pruning accuracy.

### INCONCLUSIVE

- Whether coordinate-level energy/Mask misalignment caused the negative result.
- Whether late-layer energy concentration caused the lower accuracy.
- Whether HC-SAM is consistently worse across seeds.
- How much of the static/native gap is due to support rather than the dynamic execution path without a Native-Mask fixed replay artifact.

### NOT AVAILABLE

- Coordinate-level alignment A.
- HC-versus-SAM perturbation-direction cosine.
- Exact spectral Gram condition numbers.
- Same-protocol historical per-layer reconstruction diagnostics.

## Recommendation

Do not expand the present HC-SAM formulation directly to Seeds 1–2 or a larger dataset. The intended Q-history mechanism is implemented and active, but the one completed Seed gives no task-level advantage: Dense accuracy is slightly lower, Native pruning drop is larger, and the direct static Final-Q endpoint is substantially worse than Native GPT.

HC-SAM is still worth one bounded mechanism-repair study, rather than abandonment, if it is redesigned around the missing alignment question. Before any new 200-Epoch run, preregister coordinate-level energy logging, ordinary-SAM direction cosine, and a same-parent support isolation test in which magnitude, historical-s, Final-Q and Native masks all receive the same fixed-mask compensation. Continue only if those diagnostics show that the learned history changes support in a stable, compensable direction. Otherwise, retain this run as a well-audited negative result and stop the branch.

