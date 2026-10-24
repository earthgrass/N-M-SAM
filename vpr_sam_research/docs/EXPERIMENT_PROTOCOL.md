# Completed experiments and reporting protocol

V1: SAM and Global VPR, seeds 0/1/2; VPR-L2: only seed 0.
CIFAR-ResNet18, official CIFAR10, 45000 train/5000 validation, 200 epochs,
batch128, SGD .1/.9/.0005, no Nesterov, cosine eta_min0, crop4/flip/normalize,
Cutout0. rho .05; Global gamma 5.482145661475941e-8, FP64 geometry, FP32 model.
Deterministic RTX5090 execution, no AMP/TF32/compile. Same-seed initialization,
split, augmentation and sample order are paired and hashed.

Pruning: dense final checkpoint, magnitude 2:4, exclude_first_last, tail=dense;
Guided uses the identical VPR-L2 dense final with a different frozen score.
Raw Hard preserves original BN. BN-only is separate, train-only 4096 IDs,
32×128 unaugmented batches, reset BN stats, cumulative average, no weight step.

Recovery: original unrecalibrated Raw Hard, fixed mask, fresh SGD .005/.9/.0005,
batch128, cosine eta_min0, 20 epochs/7040 steps, paired sample order.
Final remains the primary frozen result. Best validation accuracy over epochs
1–20 (ties earliest) is post-hoc exploratory; minimum loss is a separate rule.
No Test evaluation or Test-driven model/mask selection is included.

Matched random is fixed-checkpoint no-training evaluation, not additional
training seeds. See mechanism_study protocols and complete distributions.
No blockwise formal training or gamma search has been performed.
