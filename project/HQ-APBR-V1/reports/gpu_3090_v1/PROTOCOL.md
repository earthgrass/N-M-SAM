# HQ-APBR V1 first GPU exploration protocol

The user authorized GPU gate, at most three temporary smoke Epochs, then one Seed0 200-Epoch exploratory training and four fixed final evaluations. Hardware changed to RTX3090 24GiB. This deployment extends baseline commit e6705fc without replacing its source files or audit evidence. All output lives in a new remote run directory.

The configuration is configs/hq_apbr_v1_seed0_gpu_v1.json. EMA eta0.1 is a prospectively fixed exploratory choice, with no Test selection. Geometry is zero for Epoch index0, then changes only after each completed Epoch. Gram is FP32 XXT divided by patch count, relative damping0.01, inverse diagonal by Cholesky and triangular solve. Geometry and global perturbation normalization are FP64, model FP32, AMP/TF32 disabled. No APBR energy-feedback update is introduced. Noneligible trainable parameters have s=0 and participate in the shared global L2 radius0.05.

Calibration repeats the same fixed1024 Train images without augmentation for each of19 eligible layers. Only one layer's Gram is retained at a time; batch unfold temporaries are released per hook. Matrices are freed after each layer's inverse diagonal is extracted. This trades repeated model forwards for bounded memory; the complete measured wall time includes the repeated forwards. Historical native SparseGPT H=2XXT/n is explicitly restored for the two audited OBS adapters during final pruning. Its dead-feature replacement is recorded separately from HQ's ordinary damping without diagonal replacement.

Numerical gates use small independent CPU FP64 references, actual-layer input subsets (<=32features, <=256patches), and the real model. Gram relative error tolerance5e-5, inverse diagonal relative error5e-4; standalone32D FP32/FP64 inverse diagonal thresholds5e-5/1e-10. Perturbation norm relative tolerance2e-5, actual FP32 parameter addition radius tolerance1e-3. Resume must match model/optimizer/scheduler/geometry/RNG bitwise on the same GPU. Source/config/environment identity is checked on explicit resume. No automatic failure retry.

Resource caps fixed before GPU testing: peak allocated20GiB, reserved22GiB. Every-Epoch calibration must consume at most30% of measured training Epoch time across the three smoke Epochs. Exceeding any cap stops the pipeline; there is no adaptive-frequency fallback. The original every-Epoch V1 is retained. Smoke and formal training use separate output directories; formal Seed0 restarts from the frozen seed initialization after gate PASS.

The formal final parent is zero-based Epoch199, after200 completed Epochs and78200 optimizer steps. Training has no Test loader. All four endpoint files (Dense, static Final-Q Mask-only, Final-Q plus fixed-mask OBS, native dynamic SparseGPT) must be frozen with SHA256 before the single final evaluation of each. Recovery epochs0. Historical results are references until full protocol compatibility is audited.

Local CPU integration gate:25 tests passed (cpu_junit.xml). GPU performance and accuracy remain pending until actual remote evidence exists.

Entry: `python scripts/run_gpu_exploration.py --config configs/hq_apbr_v1_seed0_gpu_v1.json --archive <official-archive> --output <new-run-directory>`.

Explicit recovery, only after investigating an interruption: same command with `--resume <same-run/training/last.pt>`. Completed runs cannot be resumed through training; final evaluation cannot overwrite or automatically repeat.
