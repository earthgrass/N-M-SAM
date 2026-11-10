# SparseGPT K=4608 Remote GPU Gate Report

## Final status

- `DATA_IDENTITY_PASS = true`
- `CIFAR100_MODEL_IDENTITY_PASS = true`
- `SPARSEGPT_GPU_NUMERICAL_PASS = true`
- `SPARSEGPT_GPU_RESOURCE_PASS = false`
- Overall gate status: `FAIL` (fail-closed on the pre-registered in-process memory-release check)
- `FULL_EXPERIMENT_NOT_RUN = true`

No retry, damping change, layer reduction, K reduction, or algorithm change was performed.

## Environment

- GPU: NVIDIA GeForce RTX 4080 SUPER
- GPU UUID: GPU-15f4db77-e52f-c08d-e296-eb814b27dcef
- Driver: 595.71.05
- Advertised driver CUDA: 13.2
- VRAM: 32760 MiB
- Python: 3.12.3
- PyTorch: 2.8.0+cu128
- PyTorch CUDA build: 12.8
- Compute capability: 8.9
- CUDA matrix smoke: true

## Data and model identity

The remote archive independently matched 169001437 bytes, MD5 `eb9058c3a382ffc7106e4002c42a8d85`, and SHA256 `85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7`. All frozen content fields matched the local audit.

The model architecture hash matched: `95367b739c42cca14cb51d1a776e5dbdbe71f9ed4a1afbd2c5c0b1bb994827ea`. The remote initialized-state hash is `b93ca52940b59a84b1a3e1cf2bcc3131b22e99230e9888315d7d1850b556ad31`, differing from the local CPU/PyTorch-version hash. This PyTorch-default initialization drift is recorded; the gate result also records the tested layer's before/after hashes. No accuracy claim is made.

## Numerical gate

- Layer: `layer4.0.conv2`, weight `[512,512,3,3]`, K=4608
- Patch matrix: `[4608,512]`; direct patch-order reference exact: true
- Hessian: `[4608,4608]`, CUDA FP32
- FP64-reference relative Frobenius error: `2.24351138185e-07` (limit `5e-5`)
- Reference maximum absolute error: `3.29778475014e-09`
- Relative damping: 0.01; applied damping: `9.90509506664e-05`
- Complete 2:4 groups valid: true
- Dense-tail probe: true
- Bool keep-mask: true
- All outputs finite: true
- OOM: false
- Cholesky failure: false

Therefore `SPARSEGPT_GPU_NUMERICAL_PASS = true`.

## Timing

- Hessian build: `0.012786` s
- Decomposition: `0.075125` s
- Compensation: `1.370275` s
- Observed decomposition + compensation call: `1.539017` s
- Hessian + prune total: `1.551802` s

## Resource gate

- Peak allocated: 519058944 bytes (495.01 MiB)
- Peak reserved: 614465536 bytes (586.00 MiB)
- Free VRAM before: 33526841344 bytes
- Free VRAM after in-process cleanup: 33365360640 bytes
- In-process free-memory shortfall: 154.00 MiB
- Frozen tolerance: 64 MiB
- Memory-release check: false

After the Python process exited, `nvidia-smi` returned to 1 MiB used / 32230 MiB free with no running GPU process. This is consistent with process-scoped CUDA/context workspace retention, but the frozen in-process resource criterion still fails. It was not relaxed and the gate was not rerun.

## Decision

The next single-seed full-model technical trial is **not authorized by this gate result**, because the resource status is false even though the numerical status passed. No training, full-model pruning, recovery, seed queue, or accuracy evaluation was run.
