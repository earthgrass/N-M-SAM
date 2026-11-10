# Geo-OBS Seed 0 pilot (APBR-L2 first, then APBR-VPR)

## Scope and status

The APBR-L2 and original APBR-VPR epoch-200 Dense Final checkpoints were read without retraining. All new methods use the same official CIFAR-100 Train calibration indices (1,024 examples, seed 20261027), 19 eligible convolutional layers, 2:4 sparsity, 1% relative Hessian damping, and zero recovery epochs. Each new model was tested once on the official 10,000-image Test set after all masks and checkpoints for that parent were frozen. Historical checkpoints and results were not overwritten.

The first **static** L2 attempt completed all 19 layers but its CSV writer failed on heterogeneous fields. The failure record remains in the isolated output. A separate finalizer reconstructed the checkpoints from the 38 saved layer artifacts, verified masks and weights, and then ran the fixed Test evaluation. This static run is a reference, not the dynamic method specified later.

## Mathematical definition

For one output row, let the frozen APBR scale be \(A=\operatorname{diag}(e^{s_i})\), the physical weights be \(w\), and \(v=A^{-1}w\). With damped activation Hessian \(H_d\) and \(Q=H_d^{-1}\), the geometric local deletion cost for a candidate two-coordinate set \(D\) is

\[
C_D^{\mathrm{geo}}=\tfrac12 v_D^\top(Q_{DD})^{-1}v_D.
\]

This follows from the congruence \(Q_A=AQA\). The dynamic operator computes all six pair costs in each four-column group **after** compensation from prior groups. `Geo-Select + Std-Comp` uses \(v\) only to select the pair, then applies the OBS update to the physical \(w\). `Geo-Full` performs selection and compensation in \(v\) coordinates and maps the result back through \(A\). The standard dynamic Group-OBS control uses physical \(w\) throughout. Two shuffled controls permute the geometry within each 2:4 group before the same two geometric modes.

The sequential algorithm freezes all four coordinates of each processed group. For \(U=\operatorname{chol}(Q,\mathrm{upper})\), the next group's conditional inverse is \(U_{gg}^{\top}U_{gg}\); it equals the Schur complement in \(Q\) after conditioning on **all** preceding columns. It does **not** equal the reduced inverse obtained by leaving previously retained coordinates free. An independent 8-dimensional example measured a maximum difference of 0.2486 between those two quantities. Thus the operator is exact for its stated sequential frozen-group step, but is not the globally optimal simultaneous fixed-support solution.

## Numerical and resource gates

- Seven independent CPU numerical tests passed: explicit frozen-group reference, Schur identity, the six-coordinate counterexample, zero-geometry reduction, isotropic ranking, and blocksize invariance for all three modes. The earlier static selector tests also passed (17 tests).
- The real APBR-L2 `layer2.0.conv1` single-layer GPU gate passed. Wall times were 0.396 s for standard dynamic Group-OBS, 0.079 s for Geo-Select, and 0.075 s for Geo-Full, each below the preregistered 0.5 s ceiling. The first mode includes GPU warm-up and a slower initial decomposition, so these numbers are not a fair speed ranking.
- The APBR-VPR single-layer gate also passed. Both full-model runs completed 19 layers, with 5,578,752 exact zeros among 11,157,504 eligible weights, finite weights, and unchanged excluded parameters and BN buffers. The dynamic runs each peaked at 1.301 GiB CUDA allocated and 1.354 GiB reserved. Calibration took 0.683 s (L2) and 0.649 s (VPR). The per-method pruning and compensation wall time was approximately 4.85–5.17 s; checkpoint writing and Test evaluation are separate.
- Full-model calibration-set accuracy was 100% for all five new branches on both parents. This is training-side data and does not demonstrate generalization. The per-method calibration cross entropy values are stored in `calibration_task_metrics.json`.

## Final Test results

Accuracy is percent; CE is average cross entropy per test image. `Native SparseGPT` is a previously audited same-parent result, not a rerun. The original Dense-SAM + Native SparseGPT Seed 0 reference is 77.47% / 0.85026 CE on a **different** dense parent.

| Method | APBR-L2 Seed 0 Acc | APBR-L2 CE | APBR-VPR Seed 0 Acc | APBR-VPR CE |
| --- | ---: | ---: | ---: | ---: |
| Dense Final (unpruned) | 78.91 | 0.80720 | 78.68 | 0.80311 |
| Native SparseGPT (historical) | 77.12 | 0.86602 | 76.99 | 0.85810 |
| Static Group-OBS + fixed-mask OBS | 76.36 | 0.89723 | — | — |
| Static geometric six-pair selection + fixed-mask OBS | 76.45 | 0.89582 | — | — |
| Dynamic Group-OBS | 77.61 | 0.83629 | 77.86 | 0.82981 |
| **Dynamic Geo-Select + standard OBS** | **77.72** | **0.83548** | **77.83** | **0.82820** |
| Dynamic Geo-Full | 77.68 | 0.83590 | 77.86 | 0.82674 |
| Shuffled Geo-Select | 77.58 | 0.83640 | 77.80 | 0.83028 |
| Shuffled Geo-Full | 77.69 | 0.83708 | 77.76 | 0.83047 |

Within the **same APBR-L2 parent**, Geo-Select exceeds dynamic Group-OBS by 0.11 percentage points and its shuffled control by 0.14 points. Its mask differs from the standard dynamic Group-OBS in 4.1521% of 2:4 groups. Within the **same APBR-VPR parent**, Geo-Select trails dynamic Group-OBS by 0.03 points, while Geo-Full ties it at 77.86%; the Geo-Select mask differs in 4.0739% of groups. Relative to each parent’s historical Native SparseGPT, dynamic Group-OBS itself improves substantially (0.49 points L2; 0.87 points VPR). Hence the larger gain over Native SparseGPT cannot be credited to APBR geometry alone.

The L2 Geo-Select result exceeds the different-parent Dense-SAM + Native SparseGPT Seed 0 reference by 0.25 points. That comparison changes both the dense training method and the pruning method. It is a complete-pipeline comparison, **not** a controlled geometry ablation.

| Complete pipeline reference (Seed 0) | Test Accuracy | Test CE | Difference vs SAM + GPT |
| --- | ---: | ---: | ---: |
| Dense-SAM + Native SparseGPT (historical SOTA baseline) | 77.47% | 0.85026 | reference |
| APBR-L2 + dynamic Geo-Select + standard OBS | 77.72% | 0.83548 | +0.25 pp |
| Original APBR-VPR + dynamic Geo-Select + standard OBS | 77.83% | 0.82820 | +0.36 pp |

This compact table compares complete pipelines with different dense parents. The same-parent rows above isolate the incremental effect of geometric selection.

## Interpretation

The dynamic groupwise OBS decision is the dominant positive signal in this pilot. Geometry supplies a small additional selection signal on APBR-L2, but the accuracy gain does not replicate on the original APBR-VPR parent. Shuffled controls and cross entropy are consistent with a possible coordinate-specific effect, yet the differences are small and this is a single-seed exploratory experiment. It does not establish that APBR geometry is generally superior, nor that Geo-Full is preferable to standard physical OBS compensation. No recovery training was performed. Each dynamic branch was fixed before its Test evaluation, but the dynamic protocol was specified after the earlier static pilot result was visible; these results must remain exploratory, not presented as a preregistered confirmation.

## Evidence

The remote authoritative outputs are under `/root/autodl-tmp/apbr_l2_seed0_20261102/artifacts/` in `geo_obs_fixed_l2_seed0_v1`, `geo_dynamic_fixed_l2_seed0_v1`, and `geo_dynamic_vpr_seed0_v1`. The complete local copy is `experiments/post_training_nm_benchmark/artifacts/geo_obs_seed0_20261102_remote_copy/`. Each directory contains the parent/source/calibration identity, 19-layer evidence, bool masks, model checkpoints, fixed Test metrics, and SHA256 manifests. All original manifest entries passed server-side and local hash verification (49 static files and 111 files for each dynamic parent). Two additional files per dynamic parent were verified through supplementary manifests, for 275 verified file entries in total. The three remote manifest SHA256 values match their local copies. The single-layer gate JSON files are also copied locally. The original Native SparseGPT evidence remains in its prior audit directory. `comparison_evidence_v2.json` contains machine-readable same-parent and cross-parent deltas.
