# Phase 4 GPU Smoke Report

Authoritative status: `GPU_SMOKE_BLOCKED`.

## Deployment and environment

The only uploaded code package was `apbr_gpt_joint_seed0_server_bundle_v3.tar.gz`, SHA256 `d8b98256ff17fca186c62d0468871cb7fcbf00351ac487ed7baa1d97038ea884`. The archive hash matched before extraction. All 77 payload files passed an independent Python SHA256 verification after extraction. GNU `sha256sum -c` could not consume the Windows CRLF manifest because it treated `\r` as part of each filename; this is a portability defect in the verification script, not a payload hash mismatch.

The official CIFAR-100 archive separately passed byte-size, MD5, SHA256, and tar integrity checks before extraction. No Test pickle was opened.

- GPU: NVIDIA GeForce RTX 5090, 31.36 GiB, compute capability 12.0.
- Driver: 595.71.05.
- PyTorch: 2.8.0+cu128; CUDA build 12.8; cuDNN 91002.
- Python: 3.12.3.
- TF32 disabled; matmul precision `highest`; deterministic algorithms enabled by the smoke runner.
- GPU was initially free of compute processes; approximately 50 GiB workspace storage was available.
- Runtime dependencies were present. `pytest` was absent and was not installed because the Bundle did not freeze its version; the standalone smoke did not require it.

## Passed technical stages

The full 19-layer Hessian refresh completed in 0.2350 s. All Cholesky operations passed and all factors were finite. Peak CUDA memory was 3,471,265,792 bytes allocated and 4,634,705,920 bytes reserved.

The summed recorded Gram time was 0.00349 s and summed factor/Cholesky time was 0.12144 s. The maximum 2x2 candidate condition number was 14,027.05; `layer4.0.conv2` had the slowest factorization.

The complete 19-layer Dynamic Groupwise OBS pass completed in 45.6688 s. Every layer completed without OOM, NaN/Inf, or Cholesky failure. The final audit found:

- 19 eligible layers.
- 11,157,504 eligible coordinates.
- 5,578,752 deleted coordinates.
- Exact 2:4 compliance.
- Deleted weights exactly zero and all compensated weights finite.

The slowest OBS layer was `layer4.0.conv2` at 6.9684 s. Across the complete smoke, sampled GPU utilization averaged 49.06% and peaked at 83%; sampled device memory use peaked at 5,344 MiB.

## Training performance measurements

The same 50,000 official training samples and 391 batches were processed for each branch:

| Branch | Epoch time | Mean batch | Peak allocated | Peak reserved |
|---|---:|---:|---:|---:|
| Dense-SAM | 25.9828 s | 0.06645 s | 2.5546 GB | 2.8144 GB |
| APBR-SAM | 42.7600 s | 0.10936 s | 2.8312 GB | 3.1793 GB |
| Joint driver | 82.5485 s | 0.21112 s | 3.3543 GB | 3.4645 GB |

The observed Joint/APBR wall-clock ratio was 1.9305, below the 3.0 boundary. This ratio is retained as a diagnostic only and is not accepted as the full-method performance gate because the Joint path failed the APBR perturbation semantic check below.

## Blocking defect

The single-batch record showed:

- `eps_norm = 0.0`.
- Center CE = 4.6863546371.
- Adversarial CE = 4.6863546371 exactly.
- Parameter update L2 = 1.9412425952.
- Geometry update L2 = 0.0038661372.

`JointStepper` obtains first gradients with `torch.autograd.grad`, which returns tensors without populating each model parameter's `.grad`. It then calls the unchanged original APBR `Geometry.perturbations`, whose interface reads `p.grad`. Consequently, Geometry sees all gradients as absent, returns a zero APBR perturbation, and skips the original APBR energy feedback. The nonzero geometry change comes from the pruning-scale gradient, so it does not repair the missing APBR component.

The separate pruning probe generator used `loss.backward()` and therefore produced genuine nonzero APBR probes; this is why the Hessian/OBS and mask stages remain valid. The Joint training timing, however, measured CE plus pruning regularization without the required APBR adversarial perturbation and cannot certify the full method.

## Stop decision

No 200-epoch training, Final Test, Recovery, or hyperparameter selection was started. The smoke driver’s initial automatic `GPU_SMOKE_PASS` classification considered only the 3x ratio; `phase4_gate_status.json` is the authoritative override to `GPU_SMOKE_BLOCKED`. No code or algorithm was changed on the server and no automatic rerun was attempted. A narrowly scoped fix and a fresh smoke require user review.
