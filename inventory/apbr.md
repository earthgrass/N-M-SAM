# APBR archival identity map — 2026-11-10

This is a read-only inventory. No training, pruning, evaluation, manuscript, source, or historical evidence was changed. Machine-readable file selection and SHA256 are in `apbr.json`.

## Branch boundaries

1. **Implemented APBR and APBR-L2**: original source tree, frozen configs, small run evidence, tests, and result provenance. Original `vpr` and `fixed_l2` are separate training identities.
2. **Latest manuscript**: preserve the unchanged `paper/narrative_revision_20261106` Git HEAD `5ea3a042f44a8bdbf4e842b745536fa026b6d612`, including its latest PDF and revision records. Its no-τ method is a paper definition, not a renamed historical adapter.

If one archive branch contains both, keep these identities in separate directories and link them in the archive README. The latest manuscript tables must not be used as a machine-verifiable algorithm-to-run manifest.

## Implemented algorithms

| Identity | Training perturbation or support rule | Status |
|---|---|---|
| Original APBR / VPR | `epsilon = rho A^2 g / ||A g||` | Implemented; ellipsoid constraint |
| APBR-L2 | `epsilon = rho A^2 g / ||A^2 g||` | Implemented; actual global L2 radius |
| Physical conditional OBS | First physical singleton OBS minimum; virtually condition; second conditional physical minimum; joint physical update | Implemented and evaluated |
| Near-optimal APBR conditional OBS | At **both** choices, form physical-cost window `<= (1+tau) min`; geometry picks within window; `tau=.01` | Implemented and evaluated |
| Latest paper APBR-first | First directly `argmin |w| exp(-s)`; second purely conditional physical OBS; joint physical update; no τ | Paper definition; no independently verified matching implementation/run in the audited evidence |

Setting the historical near-optimal adapter to `tau=0` returns physical selection, not the latest paper APBR-first method. The active curvature spans the remaining suffix of the weight row, not an isolated 4x4 matrix.

## Preserve table provenance as two records

The author explicitly supplied `APBR-SAM 77.63 ± 0.14` and `SAM + conditional-greedy OBS 77.43 ± 0.11` after reviewing the paper. Preserve those manuscript values unchanged. The author-update record explicitly states that this edit supplied no new independent checkpoint/mask/config/evaluation identity.

The independently audited older APBR-VPR near-optimal `tau=.01` runs are Seed 0 `77.83%` and Seed 1 `77.43%`; their existing comparison JSON reports mean `77.63%`, sample SD `0.2828427125` percentage points. This is a distinct run-level record. Do not silently equate it with the author's later uncertainty or the no-τ paper rule.

Read in this order:

1. `review/AUTHOR_VALUE_UPDATE_20261106.md` (latest author-supplied table update).
2. `TAU_ORIGIN_AUDIT.md` (run/code evidence).
3. `ALGORITHM_VERSION_MAPPING.md` and `RESULT_PROVENANCE_CHECK.md` (historical audit before author update).
4. `sections/method.tex`, `sections/experiments.tex` and latest PDF (current manuscript).

Earlier `PENDING` wording in provenance files and `paper_state.json` predates the table update; do not interpret it as the current rendered table. Preserve it as audit history and explain its chronology in the archive index.

## Restore dependencies

`apbr.json::source_files` traces local Python imports from the relevant training/pruning/audit entry points without executing them. It also includes the entire original `src/` training package, the common CIFAR-100 model/Conv2d/2:4 helpers, SparseGPT, fixed-mask OBS, both conditional adapters, explicit configs, and targeted tests.

The archival source repository HEAD is `55fe3e300dbbd09c1178a6316afebc9ecf367947`, but newer adapters and runners are untracked and the formal training script is modified. Therefore source-byte hashes, not that HEAD alone, identify these snapshots.

Hardcoded local data and checkpoint paths occur in runner modules. Retain the original bytes and document future path remapping; importing some runners also imports other script utilities. Checkpoint parent/mask hashes and the official CIFAR archive hash must be checked before any future experiment. No archived queue should auto-start.

The dataset and checkpoint payloads remain local assets; copy a path/hash manifest into Git, not those blobs. Do not upload remote helper credentials or full chat transcripts. Do not use old `paper/APBR-SAM_geometry_first_20261106_source.zip` as the latest manuscript.
