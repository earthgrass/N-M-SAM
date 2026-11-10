# APBR Dense Weights + Original SparseGPT — Frozen Protocol

Status: **FROZEN, NOT STARTED (PREFLIGHT BLOCKED)**

## Question

Compare Original SparseGPT applied independently to Dense-SAM and APBR-SAM Epoch-200 dense weights for paired Seeds 0, 1, and 2.

## Frozen SparseGPT semantics

- Use `baseline_adapters/sparsegpt/adapter.py` and its dynamic SparseGPT mask selection.
- Conv2d inputs use the audited `grouped_patch_matrices` im2col mapping and OIHW row flattening.
- Hessian is accumulated from each model's own activations as `2 X X^T / T`.
- Structured target is 2:4 (`keep_n=2`, `m=4`) with dense incomplete tails.
- FP32 compute, block size 128, relative damping 0.01.
- Cholesky, inverse-Hessian factor, sequential block updates, and error propagation are retained.
- Eligible layers use the existing formal 19-layer manifest: 11,157,504 positions and 5,578,752 deletions.

Source identities:

- SparseGPT adapter SHA256: `4b1462b1f09e6e3789406cb506fc4c47b47cfd159fa3d0e4fc013f45927a3f36`
- Formal postprocess SHA256: `b530ddf6c44e4e7468e3c5cd446b849651114b5c00707025d8d064c57a1592db`
- Method endpoint config SHA256: `76d439a1c3b6a1463cf90ff80d5307e19036d2b6be93d79626dd7b8dcf993836`
- Base training config SHA256: `e4be5df7cf39b1ebd2730999cd6fd42458a5bcc0779e621894b1c588616879d6`

## Calibration

- Official CIFAR-100 Train only.
- 1,024 images, no augmentation, batch size 128, one pass.
- Selection seed 20261027 using NumPy PCG64 permutation of `[0, 50000)` and the first 1,024 IDs.
- The paired Dense-SAM/APBR run for a seed uses identical image IDs but builds independent activations and Hessians.
- Test data is inaccessible until all six sparse endpoints, masks, configs, and hashes are frozen.

## APBR isolation

Only `checkpoint['model']` (including BN buffers) enters model inference. `checkpoint['stepper']`, geometry/log-scales, APBR scores, prior APBR masks, and all fixed-mask OBS code are excluded.

## Prohibited operations

No dense retraining, Recovery, BN-only update, fine-tuning, Test-driven tuning, damping retry, calibration-count reduction, external mask, hybrid score, or substitution of another seed/checkpoint.

## Execution gate

The six-run GPU stage requires all six exact dense parents. It is currently blocked because APBR-SAM Seed 2 Epoch-200 Dense Final is missing. No GPU smoke or formal pruning run has been started.
