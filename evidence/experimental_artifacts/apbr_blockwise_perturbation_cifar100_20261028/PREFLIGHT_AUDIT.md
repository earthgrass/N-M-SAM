# APBR Blockwise Perturbation Preflight Audit

Date: 2026-10-28 (Asia/Shanghai)  
Status: **CPU_PRECHECK_PASS; GPU_SMOKE_PENDING**

## Repository and protection

- Repository root: `C:/Users/earthgrass73/Desktop/NM稀疏/APBR-post-training-benchmark`
- Branch: `research/post-training-nm-benchmark`
- HEAD: `55fe3e300dbbd09c1178a6316afebc9ecf367947`
- Pre-existing untracked research files were preserved. No reset, clean, forced checkout, or overwrite was performed.
- The original APBR sources remain unchanged:
  - `vpr_sam_research/src/vpr_sam/optim/step.py`: SHA256 `3c022f2815d577ef3d94ffeed40cb93c06e2359a2734c857a06f1a0c1f738ac2`
  - `vpr_sam_research/src/vpr_sam/optim/geometry.py`: SHA256 `8e0e6d42bcf34a7a8c3b79dc4b76081e9d8c0d6313b183ce0f59109bd39058b5`
- The new ideas are implemented only in the independent `blockwise_apbr/` package, with an independent config, runner, output directory, and checkpoint identity.

## Frozen original APBR implementation

The existing implementation constructs, for each trainable coordinate,

`epsilon = rho * A^2 g / ||A g||_2`,

using a stable log-domain implementation. It enforces the ellipsoid boundary `||A^{-1} epsilon||_2 = rho`; it does **not** enforce `||epsilon||_2 = rho` after geometry becomes non-identity. All 62 trainable parameter tensors participate in the original dense perturbation and global feedback. The original feedback is `Delta s_i = gamma (1 - d p_i)` with `d = 11,220,132`, and its global sum is zero.

The formal Seed 0 log confirms the distinction: actual perturbation L2 ranges from approximately 0.04983--0.05000 in Epoch 1 and 0.04879--0.04923 in Epoch 200, while the ellipsoid norm remains the frozen `rho=0.05` boundary. This observation is metadata from the existing run and was not used to tune the new methods.

## Exact correspondence used by the independent ablations

The prompt requires the real implementation to remain authoritative when its formula differs from the symbolic description. The independent ablations therefore freeze the following correspondence:

1. **Global perturbation + block feedback (Scheme 4).** The applied perturbation is exactly the frozen original APBR perturbation, element for element. The invariant is exact perturbation identity and the original ellipsoid radius. Feedback normalization changes from one global group to every complete 2:4 tuple plus one residual-coordinate group.
2. **Equal block energy + block feedback (Scheme 1).** First compute the exact original APBR perturbation. Leave every residual coordinate unchanged. Redistribute only the eligible-block energy so all nondegenerate complete blocks have equal L2 energy. Preserve that step's original actual global L2 norm, rather than silently adding a second global normalization. This isolates block-energy redistribution from total-strength changes.
3. **Residual feedback.** The non-2:4 coordinates form one residual group and use the same APBR feedback formula normalized within that group. This is the only coherent normalization-scope mapping that simultaneously retains the APBR functional form, makes each group zero-sum, and preserves the original global log-volume invariant without recentering.
4. **Degenerate blocks.** Threshold `1e-30`; fail closed, record the failure, and do not retry or change the threshold.

No original APBR source or formal artifact is modified by this correspondence.

## Frozen common protocol

- Official CIFAR-100: all 50,000 Train samples; no validation split; Test never constructed during training.
- CIFAR-ResNet18, 100 classes, random initialization per seed.
- Seeds 0, 1, 2; each new method trains from scratch for exactly 200 epochs / 78,200 steps.
- Batch 128; SGD 0.1, momentum 0.9, weight decay 0.0005, no Nesterov; cosine schedule.
- Crop padding 4 and horizontal flip 0.5; deterministic seed/epoch DataLoader generator.
- `rho=0.05`, `gamma=5.482145661475941e-8`, FP64 geometry, FP32 model, no AMP/TF32/compile.
- No Recovery, BN-only calibration, OBS, SparseGPT, or test-driven selection.
- The formal 19-layer OIHW contiguous 2:4 scope is used only to define feedback blocks during training and later to generate the unchanged APBR Raw Hard mask.

## CPU gates

- New source compiles.
- Targeted blockwise numerical tests: 3 passed.
- Verified: Scheme 4 perturbation identity; Scheme 1 equal block norms; original-step L2 preservation; residual perturbation identity; separate block/residual zero-sum feedback; finite results.
- Existing APBR files show no Git diff and retain the SHA256 values above.

## New GPU host

- NVIDIA GeForce RTX 5090, UUID `GPU-41209259-8e3d-d00d-f79d-47e47e538979`.
- 32,607 MiB total / approximately 32,110 MiB free at preflight.
- Driver 580.76.05; PyTorch 2.8.0+cu128; Python 3.12.3; compute capability 12.0.
- No CUDA compute process was present.

Formal launch remains blocked until both schemes pass a real one-batch GPU training smoke and original-source hashes are reverified on the server.
