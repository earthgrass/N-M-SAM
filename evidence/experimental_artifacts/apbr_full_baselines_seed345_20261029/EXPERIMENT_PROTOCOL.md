# APBR-SAM CIFAR-100 six-seed extension protocol

## Scope

Seeds 3, 4, and 5 extend the already audited Seeds 0, 1, and 2. Algorithms and hyperparameters remain frozen. Blockwise APBR, OBS combinations, learning-step changes, endpoint selection, and Test-driven decisions are out of scope.

The original formal YAML is a read-only reference because it binds the earlier GPU UUID and Seeds 0/1/2. The eventual Seed 3/4/5 extension YAML must preserve all data, model, optimizer, SAM/APBR, calibration, pruning, and recovery fields exactly; only the verified new GPU UUID, seed list, output root, and audit metadata may differ.

## Per-seed serial matrix

1. Train Dense-SAM for 200 epochs from the seed-specific random initialization.
2. Train APBR-SAM for 200 epochs from the corresponding seed-specific initialization/random streams as supported by the frozen implementation.
3. From Dense-SAM Final, run Magnitude, Wanda, SparseGPT, and ROSE.
4. From APBR-SAM Final, run Geometry and Magnitude masks with the same parent weights, BN state, eligible layers, grouping, tie rule, and recovery data order.
5. Save each immediate endpoint. SparseGPT and ROSE are labelled native compensated, not mask-only Raw Hard.
6. Run fixed-mask Recovery-20 from each method's native saved endpoint.
7. Audit masks and checkpoint parent hashes before Final Test.
8. Evaluate Dense Finals, all six immediate endpoints, and all six Recovery-20 Finals exactly once.

## Frozen training and pruning contract

- Dataset: official CIFAR-100, 50,000 Train / 10,000 Final Test, no validation partition.
- Dense budget: 200 epochs; fixed Epoch-200 checkpoint.
- Recovery budget: 20 epochs; fixed Epoch-20 checkpoint; no best-epoch selection.
- SAM/APBR optimizer, radius, gamma, feedback, scheduler, augmentations, and random-stream implementation remain unchanged.
- Structured sparsity: exact 2:4 in 19 internal Conv2d layers using the frozen OIHW output-row-contiguous layout.
- Expected audit totals: 11,157,504 eligible positions and 5,578,752 deletions.
- Excluded: stem Conv2d, classifier, BN affine parameters, and bias; incomplete tails remain dense.
- Test data is unavailable to training, calibration, pruning, recovery, tuning, and selection.

## APBR diagnostic-only additions

Diagnostics are observational and cannot affect a checkpoint, mask, seed, or hyperparameter:

- geometry-state mean, sample spread, extrema, and displacement from initialization;
- per-epoch update RMS and maximum absolute update;
- finite-value and scale-range checks;
- Geometry-versus-Magnitude Spearman rank correlation;
- per-layer overlap, flip counts/rates, and changed-position distribution;
- paired Raw and Recovery-20 accuracy/CE differences.

If required per-epoch geometry information is not already serialized by the frozen trainer, instrumentation must be reviewed to prove it is read-only and does not alter random-number consumption or optimizer behavior before deployment.

## Failure policy

- Single-GPU serial execution.
- No automatic retry, seed replacement, epoch reduction, method skipping, parameter adjustment, or CPU substitution.
- Any source/config/hash drift, Test access, non-finite value, mask violation, OOM, failed Cholesky, parent mismatch, or incomplete final artifact stops the affected queue and preserves evidence.
- Existing Seed 0/1/2 artifacts are read-only inputs to the final six-seed analysis and are never regenerated.

## Final analysis

Report six-seed mean and sample standard deviation for both immediate/native and Recovery-20 endpoints. Report within-seed Geometry-minus-Magnitude differences on APBR parents, and APBR-Dense-minus-SAM-Dense differences. Missing or failed values remain missing/failed and are never imputed.

## Current execution state

`BLOCKED_PRELAUNCH`: the last known RTX 5090 endpoint is offline. CPU regression, CUDA smoke, queue deployment, and formal training have not been authorized by a successful environment gate.
