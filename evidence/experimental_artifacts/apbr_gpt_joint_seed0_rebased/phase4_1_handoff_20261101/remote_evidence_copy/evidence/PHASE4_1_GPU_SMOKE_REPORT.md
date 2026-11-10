# Phase 4.1 — APBR Gradient Handoff Repair and GPU Revalidation

Final states:

- `APBR_HANDOFF_FIXED`
- `APBR_PARITY_PASS`
- `JOINT_CPU_GATE_PASS`
- `GPU_SMOKE_PASS`

## Repair

The new independent `GradientHandoff` adapter temporarily exposes `torch.autograd.grad` results through the unchanged legacy `parameter.grad` interface. It covers every APBR-managed parameter, preserves `None`, refuses a dirty destination, clones detached values to prevent aliasing, forbids nested use, and clears every destination in `finally`. The APBR perturbation, global rho, global normalization, energy feedback and volume projection were not modified.

`JointStepper` state version was raised to 2 so a pre-repair temporary smoke state cannot be resumed as the repaired method. No formal checkpoint existed.

## CPU parity and regression

The real parity test used the common rebased ResNet-18 initialization and an official CIFAR-100 training batch. With `lambda_p=0` and `eta_s=0`, Original APBR and repaired JointStepper were bitwise identical for first gradients, perturbations, energy-feedback candidate, restored pre-optimizer weights, final optimizer gradients, final parameters and final FP64 geometry. Maximum absolute error was 0. The global perturbation/ellipsoid norm was 0.05.

With the registered `lambda_p=1.3526042410542212` and `eta_s=7036.300421582199`, the real CPU Joint gate observed a nonzero APBR perturbation, radius 0.05, original energy update norm 0.00181984, pruning-scale update norm 0.00379792, exact weight restoration, one geometry commit, zero zero-skips, and a valid global volume constraint. The complete selected regression suite passed 91 tests with zero failures.

## GPU single-batch gate

On the RTX 5090:

- Center CE: 4.6863546371.
- Adversarial CE: 5.3391218185.
- `eps_norm`: 0.049999999996.
- Ellipsoid norm: 0.049999999996.
- Original energy update norm: 0.0016406597.
- Pruning-scale update norm: 0.0038661372.
- Combined scale update norm: 0.0047653021.
- Weight restoration maximum error: 0.
- Single-batch time: 0.54189 s.
- Peak allocated/reserved: 1,418,749,440 / 1,432,354,816 bytes.

All mandatory handoff checks passed before later smoke stages were allowed to run.

## Hessian and Groupwise OBS

The full 19-layer Hessian refresh completed in 0.23103 s. Every Cholesky decomposition passed. Peak allocated/reserved memory was 3,471,265,792 / 4,634,705,920 bytes.

The full 19-layer Dynamic Groupwise OBS completed in 45.70260 s. All outputs were finite. The mask audit confirmed 19 layers, 11,157,504 eligible coordinates, 5,578,752 deleted coordinates, and exact 2:4 compliance. Peak allocated/reserved memory was 2,668,421,120 / 4,691,329,024 bytes.

## Temporary one-epoch performance gate

Each branch processed all 50,000 official training samples in 391 batches from the identical initialization:

| Branch | Epoch time | Mean batch | P95 batch | Peak allocated | Peak reserved | Geometry delta L2 |
|---|---:|---:|---:|---:|---:|---:|
| Dense-SAM | 25.6431 s | 0.06558 s | 0.07055 s | 2.5546 GB | 2.8144 GB | 0 |
| APBR-SAM | 43.1758 s | 0.11042 s | 0.11443 s | 2.8312 GB | 3.1793 GB | 0.90793 |
| Joint APBR-GPT | 99.9052 s | 0.25551 s | 0.27802 s | 3.4476 GB | 3.5610 GB | 0.86437 |

The synchronized Joint/APBR epoch-time ratio is `2.313918`, below the fixed `3.0` stop boundary. The per-branch `nvidia-smi` utilization samples are retained but are not used for the decision because vGPU utilization telemetry was inconsistent with synchronized CUDA wall time.

## Stop state

No formal 200-epoch branch, Final Test, Recovery, or hyperparameter selection was started. Smoke products remain isolated under Phase 4.1 and cannot be used as formal cohort checkpoints. The GPU was idle after completion. Work stops here pending user review.
