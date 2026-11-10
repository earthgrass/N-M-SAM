# Preflight Audit — APBR Dense Weights + Original SparseGPT

Date: 2026-10-30 (Asia/Shanghai)

Overall status: **BLOCKED — FORMAL GPU EXECUTION NOT STARTED**

## Repository

- Git root: `C:/Users/earthgrass73/Desktop/NM稀疏/APBR-post-training-benchmark`
- Branch: `research/post-training-nm-benchmark`
- HEAD: `55fe3e300dbbd09c1178a6316afebc9ecf367947`
- Worktree contains pre-existing untracked research adaptations and evidence. Nothing was reset, cleaned, overwritten, or committed.

## Checkpoint gate

Five of six required dense parents are hash-verified. All five are Epoch 200 / 78,200 steps, contain 122 model-state tensors including 60 BN running buffers, and have identical tensor-key/shape structure. APBR model weights are separable from the checkpoint `stepper` geometry; only the `model` state would enter SparseGPT.

Blocking item:

- **APBR-SAM Seed 2 Epoch-200 Dense Final is absent.**
- Formal audit records its expected SHA256 as `6b55caf932d846e16a13f7c2510e216db790bcd6738941288491c2e1d4b24a22`.
- Existing Seed 2 APBR Raw-Hard, Recovery, mask, or compensated artifacts are not valid substitutes because they have lost or changed dense coordinates.
- Seed 3 is not substituted because the registered comparison requires Seeds 0, 1, and 2.

The Seed 2 Dense-SAM rolling `last.pt` is valid: its SHA256 exactly equals the published fixed-final hash in both `checkpoint_publication.json` and `status.json`. A separately copied 7.2 MB `final_epoch_200.pt` is truncated and is not used.

## SparseGPT audit

The project has a frozen Conv2d-capable SparseGPT adapter that preserves dynamic N:M scoring and sequential OBS compensation. The formal configuration is explicit: 1,024 Train calibration images, FP32, block size 128, damping 0.01, 2:4, and no Test access. Prior maximum-layer GPU evidence exists, but this new experiment still requires its own smoke after the checkpoint gate passes.

This path is distinct from `fixedmask_obs`: it does not accept an external mask and selects deletion coordinates dynamically from the compensated current block.

## New GPU host observation

- GPU: NVIDIA GeForce RTX 5090, 32,607 MiB reported by `nvidia-smi`.
- Driver: 580.105.08; runtime reports CUDA 13.0.
- Python: 3.12.3; PyTorch: 2.8.0+cu128; `torch.cuda.is_available() == True`.
- Compute capability: 12.0; PyTorch-visible memory: 32,109 MiB.
- No CUDA compute process was listed at preflight; data disk had approximately 50 GB free.

The host remains unused because the mandatory checkpoint gate failed.

## Required user action

Provide the exact APBR-SAM Seed 2 Epoch-200 Dense checkpoint whose SHA256 is `6b55caf932d846e16a13f7c2510e216db790bcd6738941288491c2e1d4b24a22`, or restore access to the storage containing it. Once supplied, the next allowed steps are: verify its hash/identity, run CPU numerical tests, upload the minimal package, run the single-layer GPU smoke, then execute the six serial Original SparseGPT endpoints.
