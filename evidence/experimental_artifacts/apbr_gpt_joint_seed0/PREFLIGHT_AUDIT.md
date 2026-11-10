# APBR-GPT Joint Seed 0 — Preflight Audit

Status: `BLOCKED_BEFORE_GPU_GATE`

## Repository and protected state

- Branch: `research/post-training-nm-benchmark`
- HEAD: `55fe3e300dbbd09c1178a6316afebc9ecf367947`
- The worktree already contains extensive untracked research artifacts. They
  were preserved; no reset, clean, checkout, or overwrite was performed.
- Original APBR, SparseGPT, and benchmark implementations were not modified.

## Frozen protocol recovered

- CIFAR-100: official 50,000 train / 10,000 test archive.
- Validation size: zero. The existing protocol uses fixed epoch 200 rather
  than validation selection. A validation split must not be invented for this
  experiment.
- Model: `ssam2022_cifar_resnet18_c100_v1`.
- Seed: 0; random initialization; 200 total epochs; 5 epoch APBR warm-up.
- APBR: global perturbation, rho 0.05, FP64 persistent log-scales, original
  global energy feedback and zero-volume constraint.
- SparseGPT: activation reconstruction Gram (not the CE Hessian), 1% relative
  damping, FP32 formal computation, block size 128, dynamic 2:4 OBS order.
- Eligible coordinates: the frozen 19 internal Conv2d layers.

## Asset identity

- APBR Seed 0 epoch-200 checkpoint SHA256:
  `4332079becdb58fb50f0e68b8bd0dcc251db0184a6050facb524817520a51d82`.
- The checkpoint records initial model SHA256:
  `87aa9dd7492a89adf9e3ab78ca4a3227f30bc8de53b18a3bd73cd61ab0bd759d`.
- Current model, APBR step, APBR geometry, and formal config source hashes
  match the hashes recorded by the historical checkpoint.
- A local reconstruction does not reproduce the initial tensor hash because
  the local PyTorch runtime differs from the formal runtime. This is not
  accepted as either pass or failure; reconstruction must be repeated in the
  target PyTorch 2.8/CUDA environment before training.

## New isolated implementation

The new module `baseline_adapters/apbr_gpt_joint` implements:

- exact six-pair `0.5 w_D^T (Q_DD)^-1 w_D` costs;
- K-probe APBR symmetric costs with coefficient `1/(2K)`;
- differentiable normalized six-candidate soft-min;
- dynamic group selection inside SparseGPT's sequential OBS update;
- synchronized probe propagation through the same linear OBS operator;
- fixed-seed within-block shuffled-probe control.

It does not modify the frozen SparseGPT compensator.

Targeted CPU result: `6 passed` in `tests/test_apbr_gpt_joint.py`.

## Blocking conditions

1. The previously authorized RTX 6000D SSH endpoint is offline. GPU identity,
   initial-model replay, CUDA smoke, one-epoch timing, and formal training
   cannot be executed.
2. The supplied specification does not define the numerical calibration rule
   for `eta_s`, nor the allowed safety interval for the one-time calibrated
   `lambda_p`. Choosing these silently would change the proposed method.

No Final Test was accessed by the new implementation.
