# Frozen APBR-GPT Execution Protocol

## Single-layer gate

- Parent: APBR-SAM Seed 0 Dense Epoch 200.
- Layer: `layer1.0.conv1`, an eligible internal 3x3 Conv2d selected before results.
- Mask for fixed-support ablation: Original SparseGPT final Mask.
- Calibration: frozen 1,024 Train IDs; independent disjoint Train diagnostic IDs are used for out-of-calibration reconstruction error.
- Lambda candidates: `{0, 1e-4, 1e-3, 1e-2, 1e-1, 1}`.
- No validation partition exists in the frozen 50k/10k protocol. Lambda cannot be selected using Final Test. The Pilot reports all candidates and may freeze a value only from preregistered Train-only numerical/reconstruction criteria.
- Compare fixed Mask: no compensation, standard quadratic, isotropic ridge, APBR-Q, and deterministic shuffled APBR-Q.
- Compare dynamic selection/OBS: Original H, Exact Row-wise Q, and Shared Geometry Q.
- Record positive definiteness, condition estimate, residual, mask difference, reconstruction error, compensation norm, runtime, and CUDA peak allocated/reserved memory.

## Stop rule

No full-network HH/HQ/QH/QQ task may start unless the single-layer gate passes numerical correctness and its Exact Row-wise cost is judged feasible. If Exact cost is unacceptable, preserve the single-layer result and do not relabel Shared Geometry as Exact.

No Recovery-20, BN-only update, Dense retraining, APBR geometry update, or Final-Test hyperparameter selection is allowed.
