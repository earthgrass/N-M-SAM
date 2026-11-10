# SAM Dense Seed 0: native SparseGPT vs singleton-OBS conditional greedy

## Purpose and frozen setup

This is a **local CPU Seed 0 exploratory test** of whether the earlier singleton-OBS conditional greedy plus joint group compensation helps when the Dense parent is original SAM instead of APBR. It does **not** use APBR geometry. Both branches start independently from the same official CIFAR-100 ResNet-18 SAM Epoch-200 Dense Final, SHA256 `3a0716555d33ec92efe8362e24abec9be27625d24ffb6dfb1d51fddb5b91bd13`. They share the same 1,024 historical Train-only calibration IDs and layer Hessians, FP32 Gram accumulation, FP64 factorization/OBS, relative damping `0.01`, block size `128`, and exact 2:4 convention across 19 eligible Conv2d layers. No training, BN update or Recovery was run.

The native branch selects both deletions at group entry with the original SparseGPT diagonal-factor score and compensates in original column order. The exploratory branch chooses the first deletion using \(w_j^2/(2R_{jj})\), where \(R=U_g^\top U_g\); it virtually deletes that coordinate, selects the second with the updated singleton OBS cost, then jointly compensates the chosen pair from the current unpruned group weights. **Both selection score and compensation unit differ** from native SparseGPT, so this is a complete-algorithm comparison, not a score-only ablation.

## Fixed Final Test and local cost

Both checkpoints and SHA256 identities were frozen before each branch's one fixed 10,000-image Test evaluation.

| Method | Test accuracy | CE | Sum of physical layer quadratic errors | Decomposition | Selection + compensation | Pruning wall |
|---|---:|---:|---:|---:|---:|---:|
| Native SparseGPT | 77.49% | 0.850231 | 1.099589 | 1.872 s | 3.254 s | 5.391 s |
| Singleton-OBS conditional greedy + joint OBS | **77.84%** | **0.828370** | **0.908167** | 1.765 s | 3.420 s | 5.488 s |

The exploratory algorithm exceeds the same-run native baseline by **0.35 percentage points** (35/10,000 Test images), with CE lower by `0.021861` and the frozen Dense-calibration layer quadratic error sum lower by **17.41%**. The historical GPU native SparseGPT result on this same SAM parent was 77.47%; the present local native replay is 77.49%. The error sum is a layerwise physical quadratic diagnostic, not global network output error.

In this one ordered CPU run, pruning wall time differs by **+0.096 s (+1.79%)** and the combined selection-plus-compensation stage by **+0.165 s (+5.09%)**. Both branches share a separate `73.382 s` CPU calibration phase. Existing timers **do not separate score time from compensation time**, so neither `+0.165 s` nor the wall difference can be called the compensation-only overhead. The local timing also cannot establish GPU cost or a stable runtime ratio.

## Work count and interpretation

For one 2:4 group and one output row, native GPT evaluates **four** scalar deletion scores. The singleton-OBS conditional selector evaluates four first-round scores and three conditioned second-round scores: **seven**, or **1.75 times as many candidate score evaluations**. It also constructs a shared \(4\times4\) group inverse and performs a virtual rank-one update. This `1.75` is a count of candidates, **not** a FLOP ratio or runtime multiple. Both methods share the expensive full Hessian decomposition. The joint compensator uses a two-coordinate local solve instead of native columnwise errors; its incremental time was not isolated.

The final Masks disagree at 2,065,890 / 11,157,504 coordinate decisions (18.52%), or 976,545 / 2,789,376 quartets (35.01%). Both remove exactly 5,578,752 weights. Independent read-only audit confirmed all 19 Masks are bool and strict 2:4, deleted weights are exact zero, all model tensors are finite, and BN/excluded tensors are bitwise unchanged. Checkpoint, Mask, parent, calibration, Test-data and source hashes matched the recorded values; no failure file appeared. The audit did not rerun Test.

For context, the same exploratory method on the **APBR Dense Seed 0** parent reached 77.72%, versus a local native GPT replay at 77.00%. The positive SAM-parent result shows the method can help without APBR geometry; it does **not** establish that the new score alone caused the gain, nor a reliable APBR-by-method interaction. These are one-seed exploratory results, not a new multi-seed formal comparison.
