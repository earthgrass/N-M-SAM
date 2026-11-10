# Fixed-Mask Second-Order Repair: Preflight Audit

Date: 2026-10-28 (Asia/Shanghai)  
Status: **PREFLIGHT PARTIAL PASS; SEED 2 PARENT BLOCKED**

## Repository and protection state

- Repository root: `C:/Users/earthgrass73/Desktop/NM稀疏/APBR-post-training-benchmark`
- Branch: `research/post-training-nm-benchmark`
- HEAD: `55fe3e300dbbd09c1178a6316afebc9ecf367947`
- The working tree contains pre-existing untracked research code and evidence. No reset, clean, checkout, or overwrite was performed.
- Protected historical implementations (`benchmark/model.py`, `benchmark/nm.py`, and `benchmark/obc.py`) remain untouched.
- This experiment uses a separate artifact directory and will not overwrite the formal three-seed matrix.

## Reused compensation implementation

The audited implementation is `baseline_adapters/fixedmask_obs/adapter.py` (SHA256 `d7cff3b33665811bad242ed8139b03dbbff493a665f63074b87aec266d3b88c3`). It is a **fixed-mask SparseGPT-ordered OBS-style compensation**, not native SparseGPT:

- Conv2d inputs are converted with the existing grouped unfold mapping, consistent with OIHW output rows.
- Per-group statistics use `H = 2 X X^T / n`.
- Relative damping is `0.01 * mean(diag(H))` in the frozen full-model protocol.
- The implementation uses Cholesky factorization, an inverse Hessian, and the upper Cholesky factor used by SparseGPT-style sequential error propagation.
- The caller supplies an immutable bool keep-mask; the implementation contains no score computation or dynamic support selection.
- Deleted coordinates are written to exact zero, while retained coordinates may receive compensation updates.
- Existing targeted tests cover external-mask immutability, exact mask-only projection, an independently solvable positive-definite instance, local reconstruction improvement, and Conv2d unfold/Hessian consistency.

Known scope: this is a sequential/blockwise approximation whose outcome can depend on coordinate order. It must not be described as an exact globally constrained OBS solve or as native SparseGPT.

## Historical Seed 0 result

The historical Seed 0 APBR Geometry + fixed-mask OBS result is fully evidenced and matches the frozen configuration requested here (1024 official training images, selection seed 20261027, FP32, damping 0.01, block size 128, no AMP/TF32/compile, no recovery, no BN calibration):

- APBR Dense parent: `4332079becdb58fb50f0e68b8bd0dcc251db0184a6050facb524817520a51d82`
- Historical APBR Raw: `74ea702356b6cb3f10ad34985ca8de6757ccb80702cced3debe4ec5936b9b9b3`
- Geometry mask: `79d9c410c67e931101cf46112e732c68133a9935e4f99a41503a0de8ee7b40f4`
- Compensated checkpoint: `4907a607f7a45ff3a9cc9a5c0bf66b2cfdc26260e2f019b3967a7095e22ae86f`
- Final fixed-endpoint evaluation: accuracy 0.7637, cross-entropy 0.90397955.
- All 19 masks matched the formal Raw artifact; 11,157,504 eligible positions and 5,578,752 deleted positions; deleted values were zero, retained values changed, BN buffers were unchanged, and all values were finite.

Therefore Seed 0 condition B is eligible for evidence-based reuse. Its historical test result will not be used to tune any new run.

## Parent and control availability

| Seed | APBR Epoch-200 Dense parent | Geometry Raw reference | Same-parent Magnitude Raw evidence | Preflight state |
|---:|---|---|---|---|
| 0 | present; SHA256 `4332079...a51d82` | present; SHA256 `74ea702...b9b9b3` | present and previously audited | ready |
| 1 | present; SHA256 `6571e30...6e10b` | present; SHA256 `63acc8d...f2113` | present and previously audited | ready |
| 2 | **missing locally**; expected SHA256 `6b55caf...b24a22` | full formal Raw artifact also missing locally | mask/evaluation evidence present | **blocked** |

The former Blackwell host that held Seed 2 is no longer reachable. A sparse Raw or Recovery checkpoint cannot substitute for the original dense parent because the pruned coordinates' original values are required for the deletion displacement and compensation propagation. Seed 2 will not be fabricated or reconstructed from an invalid starting point.

## New server environment

- Host: newly supplied SeeTacloud instance.
- GPU: NVIDIA GeForce RTX 5090, UUID `GPU-e4222bb2-c234-ec5b-d776-bd6cf8372984`.
- Driver: 595.71.05.
- Memory: 32,607 MiB total; approximately 32,110 MiB free at preflight.
- Python: 3.12.3.
- PyTorch: 2.8.0+cu128; CUDA available; compute capability 12.0.
- No CUDA compute process was present at preflight. Formal execution remains gated on a fresh occupancy check.

## Gate decision

CPU targeted tests and a one-layer real-checkpoint GPU smoke must pass before any full-model run. Seeds 0 and 1 may then run/reuse under the frozen protocol. The three-seed experiment cannot be marked complete until the exact Seed 2 dense parent (`6b55caf932d846e16a13f7c2510e216db790bcd6738941288491c2e1d4b24a22`) and its matching geometry state are recovered.
