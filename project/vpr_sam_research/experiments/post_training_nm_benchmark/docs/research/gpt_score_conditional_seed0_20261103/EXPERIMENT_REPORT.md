# Native SparseGPT score with conditional second choice: APBR Dense Seed 0

## Controlled change

This local CPU exploratory experiment tests whether **reselecting only the second deletion** improves native SparseGPT 2:4 pruning. At each current quartet:

1. Compute the original adapter's score \(c_j=w_j^2/U_{jj}^2\) for all four weights, and stably choose the lowest-scored first coordinate \(i\). Thus the first choice is exactly native SparseGPT's first-ranked coordinate **at the same current group state**.
2. Virtually delete \(i\) using the group's conditional inverse \(R=U_g^\top U_g\). Update the other three weights and their rank-one Schur-complement inverse \(R'\). Factor \(R'=U'^\top U'\), then apply the **same GPT score form** \((w'_j)^2/(U'_{jj})^2\) to choose the second coordinate. The virtual state is discarded.
3. Give the selected two-coordinate Mask to the unchanged, original-column-order **native sequential OBS** update. There is no joint pair compensation and no APBR geometry signal in this selector.

Both branches start from the same original APBR-VPR Seed 0 Epoch-200 Dense parent (SHA256 `4332079becdb58fb50f0e68b8bd0dcc251db0184a6050facb524817520a51d82`), use the same 1,024 historical CIFAR-100 Train calibration IDs, same per-layer FP32 Gram, FP64 OBS, `0.01` relative damping and block size `128`. The branches process all 19 eligible Conv2d layers independently. No recovery, BN update or parameter training occurs. Both checkpoints were frozen and hashed before either fixed 10,000-image Test evaluation.

## Results

| Method | Final Test accuracy | CE | Sum of physical layer quadratic errors | CPU pruning wall time |
|---|---:|---:|---:|---:|
| Native SparseGPT, same local run | 77.00% | 0.858189 | 1.047292 | 5.23 s |
| Native GPT score + conditional second choice | 77.01% | 0.855625 | 0.989429 | 9.21 s |

The accuracy difference is **+0.01 percentage points, one of 10,000 Test images**. CE decreases by `0.002564`; the sum of per-layer physical quadratic errors on the frozen Dense calibration Gram decreases by **5.52%**. This error sum is not the network's final output MSE, and the modest loss change does not establish a robust classification benefit. The historical GPU native GPT result on this same Dense parent was 76.99% / CE `0.858100`; the local native replay is within one Test image of it.

The conditional second choice differs from the initial native second-ranked coordinate in **310,878 / 2,789,376 groups (11.15%)**, measured at the conditional branch's current state. The final two branch Masks differ in **845,281 groups (30.30%)** because changed masks alter subsequent compensated weights. The final coordinate-level disagreement is 1,765,190 / 11,157,504 (15.82%). Each output deletes exactly 5,578,752 coordinates and preserves strict 2:4.

The conditional branch took longer in this ordered CPU run. Its four small conditional factorizations per group add work; the single-run timings do not establish GPU cost or a controlled speed ranking. This variant showed **no meaningful Test accuracy improvement** over native SparseGPT. It also did not reproduce the earlier 77.72% result from a different conditional-greedy design: that earlier design changed both the first score and the compensation unit to joint OBS. These are different methods, and the comparison does not isolate which of those changes caused the earlier improvement.

## Evidence and limits

- Eight targeted CPU tests passed: native first-score parity, stable tie behavior, diagonal-Hessian native replay, an independent correlated fixture with a changed second choice, independent Schur/Cholesky reference, block-size invariance, strict 2:4/dense-tail/zero checks, and invalid-input rejection.
- The run wrote 19 layer logs, two frozen checkpoints, `pre_test_freeze.json` and `results.json` to `artifacts/gpt_score_conditional_apbr_seed0_cpu_20261103/`. It recorded input parent, historical Raw, Test pickle, Mask, checkpoint and source SHA256 identities.
- Independent read-only artifact audit passed: all recorded hashes, layer sums and calibration IDs matched; Mask tensors are bool with exact zeros at deleted positions; all model tensors are finite; BN and excluded tensors are bitwise unchanged from the Dense parent; no failure file was present. The audit did not rerun Test.
- Results are **one Seed 0 CPU exploratory experiment**, not a new formal multi-seed result. No hyperparameter was selected using Test, and no GPU or Recovery run was started.
