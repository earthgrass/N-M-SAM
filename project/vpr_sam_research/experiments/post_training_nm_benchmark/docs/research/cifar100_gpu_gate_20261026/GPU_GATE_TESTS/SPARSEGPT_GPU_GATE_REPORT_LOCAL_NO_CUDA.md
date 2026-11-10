# SparseGPT Single-Layer GPU Gate

## Current status

- `SPARSEGPT_GPU_NUMERICAL_PASS = false` (not run; no CUDA device)
- `SPARSEGPT_GPU_RESOURCE_PASS = false` (not run; no CUDA device)
- Gate status: `BLOCKED_NO_CUDA`
- `FULL_EXPERIMENT_NOT_RUN = true`
- Prerequisites: `DATA_IDENTITY_PASS = true`, `CIFAR100_MODEL_IDENTITY_PASS = true`

These false values mean **not accepted**, not a measured failure of SparseGPT. CPU results are not presented as GPU acceptance.

## Environment evidence

- Python: `3.12.10`
- PyTorch: `2.13.0+cpu`
- PyTorch CUDA build: `null`
- `torch.cuda.is_available()`: `false`
- CUDA device count: `0`
- `nvidia-smi`: command not found
- GPU model, total VRAM, free VRAM, peak allocated/reserved memory and CUDA timings: unavailable because no CUDA device exists.

Machine-readable evidence is saved in `GPU_GATE_TESTS/environment.json` and `GPU_GATE_RESULTS.json`. Repository evidence records root, branch `research/post-training-nm-benchmark`, HEAD `9ff061a009e19e619a8a8ae3f5b08b494c83b8d0`, and the complete porcelain status without modifying or cleaning the worktree.

## Frozen executable gate

Entry point: `scripts/run_sparsegpt_single_layer_gpu_gate.py`

The gate is fixed to:

- randomly initialized 100-class model, seed `20261026`;
- layer `layer4.0.conv2`, weight `[512,512,3,3]`, `K=4608`;
- 32 fixed verified CIFAR-100 training images, no augmentation;
- train-only mean/std normalization from the verified archive;
- Hessian `H = 2XXᵀ / n` in CUDA FP32;
- `2:4`, block size 128, relative damping `0.01`, and no automatic retry or damping tuning;
- dynamic sequential mask selection and OBS error compensation;
- one model layer only—no full-model Hessian cache or pruning.

The existing SparseGPT adapter now accepts an explicit `compute_dtype`; its default remains FP64 for compatibility with all previous CPU numerical tests. The GPU gate explicitly selects FP32 and records decomposition and compensation timings without changing the algorithm.

## Pre-registered acceptance checks

If executed unchanged on a CUDA host, the script will:

1. independently verify the captured Conv2d patch order and `[4608,512]` patch matrix;
2. build the `[4608,4608]` Hessian and compare a fixed 64-dimensional submatrix with CPU FP64 (`relative Frobenius error ≤ 5e-5`);
3. apply exactly one relative damping value (`0.01`) and run Cholesky, inverse and compensation factor construction;
4. run dynamic column-wise 2:4 selection and compensated writeback to original OIHW coordinates;
5. require exactly 2 retained positions in every complete quartet, a boolean keep-mask, a controlled dense-tail probe, and finite Hessian/weights;
6. record Hessian time, decomposition time, compensation time, total time, peak allocated/reserved bytes, free VRAM before and after, OOM/Cholesky failures, traceback, and post-release memory;
7. stop on OOM, factorization failure or numerical failure without shrinking the layer, changing damping, or skipping `K=4608`.

## Why no peak-memory or timing number is reported

No CUDA kernel ran. Consequently, reporting GPU memory or timing would be fabricated. `GPU_GATE_RESULTS.json` contains no surrogate CPU measurements.

## Entry condition for a single-seed full-model technical trial

Not met. It requires all four statuses to be true: data identity, final model identity, SparseGPT GPU numerical gate, and SparseGPT GPU resource gate. The current machine cannot establish the last two.

Formal CIFAR-100 Dense-SAM/APBR-SAM training, full-model pruning, recovery, three-seed queues, accuracy tables and SOTA claims remain outside this task and require separate explicit approval.

