# Frozen Experiment Protocol

Experiment ID: `fixed_mask_second_order_repair_cifar100_20261028`

- Dataset: official CIFAR-100, 50,000 Train / 10,000 Test.
- Model: frozen 100-class CIFAR-ResNet18.
- Parents: each seed's APBR-SAM fixed Epoch-200 Final checkpoint and its matching geometry state.
- Eligible set: the existing formal 19 internal Conv2d layers; stem, classifier, BN, and bias remain excluded.
- Sparsity: strict 2:4 by contiguous OIHW output-row groups; stable tie handling; bool keep-mask; dense tail.
- Conditions: A Geometry Raw, B Geometry + fixed-mask second-order compensation, C same-parent Magnitude Raw, D Magnitude + the identical compensation algorithm.
- Calibration: 1,024 fixed CIFAR-100 training images, no augmentation, selection seed 20261027, identical order for B and D within a seed.
- Compensation: FP32; relative damping 0.01; block size 128; no AMP, TF32, or compile; no dynamic mask selection.
- Training: none. Recovery epochs: 0. BN-only calibration: none.
- Evaluation: one fixed Final Test evaluation per frozen endpoint after implementation/configuration gates pass. Test is never used for calibration, tuning, mask selection, or retry decisions.
- Execution: CPU numerical/interface tests, one-layer GPU smoke on a real parent, then full-model jobs serially by seed and mask source. No automatic retry or parameter change after a failure.

Seed 0 Geometry + compensation is reused only because its parent, mask, calibration, algorithm, configuration, output checkpoint, and fixed evaluation evidence are all available and hash-verified. Seed 2 remains blocked until its exact dense parent is recovered.
