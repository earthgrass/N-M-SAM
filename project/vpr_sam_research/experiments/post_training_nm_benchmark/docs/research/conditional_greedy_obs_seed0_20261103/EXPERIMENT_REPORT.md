# Conditional greedy 2:4 OBS: APBR Dense Seed 0 CPU pilot

## Question and algorithm

This is **not** the native SparseGPT 2:4 selector. Native SparseGPT scores the four current coordinates once, selects two for removal, and then propagates their OBS errors column by column ([paper, §3.3 and Algorithm 1](https://proceedings.mlr.press/v202/frantar23a/frantar23a.pdf)). The new selector first chooses one deletion by singleton OBS cost, virtually applies its OBS update to the three remaining coordinates, and chooses the second by the conditional singleton cost. It then applies one **joint two-coordinate OBS** update to the original current quartet. It never enumerates all six pairs during selection. This experiment changes the support selector **and** the compensation unit relative to native SparseGPT; it is therefore not a one-variable ablation against native SparseGPT.

For a group's conditional inverse curvature \(R\), current row weights \(w\), and first choice \(i\):

\[
i=\arg\min_k \frac{w_k^2}{2R_{kk}},\quad
w'_j=w_j-\frac{R_{ji}}{R_{ii}}w_i,\quad
R'_{jj}=R_{jj}-\frac{R_{ji}^2}{R_{ii}},\quad
j=\arg\min_{k\ne i}\frac{(w'_k)^2}{2R'_{kk}}.
\]

For any **fixed pair and fixed current state**, the first singleton cost plus the conditional second cost equals the exact two-coordinate quadratic OBS cost. The greedy first choice can nevertheless exclude the best pair. The six-pair comparator evaluates all pairs at the current group and selects the local minimum; it is not a global optimum over all 2:4 groups.

## Frozen comparison

- Parent: original APBR-VPR Seed 0 Epoch-200 Dense Final, SHA256 `4332079becdb58fb50f0e68b8bd0dcc251db0184a6050facb524817520a51d82`.
- Model: CIFAR-100 ResNet-18; 19 eligible internal Conv2d layers; 11,157,504 eligible coordinates. BN and excluded parameters remain identical to the parent.
- Calibration: the historical 1,024 fixed CIFAR-100 **Train** images; no Test access during calibration/pruning. Shared FP32 activation Gram per layer, FP64 damping/factorization/OBS; relative damping `0.01`, block size `128`.
- Both branches start independently from the same Dense parent and use the same Gram, grouping and joint compensation. No geometry scale is used by this pilot. No training, BN update or Recovery.
- Both checkpoints and their SHA256 hashes were frozen before each branch's one fixed evaluation on the 10,000-image Test set.
- The first run is retained as diagnostic evidence because the adapter source changed during execution (`artifacts/conditional_greedy_obs_apbr_seed0_cpu_20261103/SOURCE_STABILITY_NOTE.md`). The source-frozen `..._v2` run below is authoritative. The two runs produced identical checkpoint and Mask hashes.

## Seed 0 results

| Selector and compensation | Final Test accuracy | CE | Physical layer quadratic error sum | Pruning wall time, CPU |
|---|---:|---:|---:|---:|
| Conditional greedy + joint OBS | 77.72% | 0.829459 | 0.867344 | 9.37 s |
| Dynamic six-pair reference + joint OBS | 77.87% | 0.829583 | 0.859425 | 7.93 s |

Conditional greedy is **0.15 percentage points lower** in accuracy and **0.92% higher** in the sum of per-layer physical quadratic errors. Its CE is lower by about 0.000124, a small discordant change. The two final Masks differ in 593,975 of 2,789,376 quartets (21.29%), or 1,333,058 of 11,157,504 coordinate decisions (11.95%). Each branch deletes exactly 5,578,752 positions. The error sum uses the frozen Dense activation Gram and is not a global network output error. Once the two paths differ, later groups see different compensated weights; their per-group costs must not be interpreted as same-state regret.

The historical **native SparseGPT on this same APBR Dense parent** reached 76.99% in a separate environment. That contextual result is not a controlled speed or single-change comparison with this CPU pilot. The present pilot does **not** establish a contribution from APBR geometry, because its selector never reads the geometry state.

## Numerical and artifact audit

- Targeted CPU numerical tests: **7 passed**. They include the singleton-plus-conditional-cost identity, a case where greedy misses the six-pair optimum, an independent reduced-Hessian reference, block-size invariance, diagonal reduction and invalid-input rejection.
- Independent read-only artifact audit: both checkpoint and Mask hashes match the frozen records; 19 layers each have bool keep-Masks, exact 2:4, finite model tensors and exact zeros at deleted positions. All BN and excluded tensors are bitwise equal to the Dense parent. The historical APBR Raw checkpoint has the same parent hash and the calibration ID list matches; historical Raw SHA256 is `74ea702356b6cb3f10ad34985ca8de6757ccb80702cced3debe4ec5936b9b9b3`.
- No failure file was emitted. There was no Recovery or additional training. The Test pipeline was called only after both checkpoints were frozen; source hashes were checked before this call.
- Recorded source hashes cover the runner and two adapters. The supplemental `SHA256SUMS.txt` also covers the reused data/evaluation helper, model/config and Test pickle for provenance.

## Interpretation and next step

This single-Seed CPU result shows the conditional greedy selector is technically viable and materially changes the Mask. It did **not** beat the dynamic six-pair reference or show a speed advantage. The reported CPU times were measured in fixed branch order and are sensitive to cache and warm-up, so they should not be treated as a controlled timing ranking. The potential scaling benefit for larger groups is still hypothetical: 2:4 requires 7 sequential singleton candidate scores versus 6 pair costs; 4:8 would require 26 versus 70 pair candidates, and 8:16 would require 100 versus 12,870. Kernel efficiency and whole-model accuracy at those structures remain untested. No Seed 1/2 or GPU run was started.
