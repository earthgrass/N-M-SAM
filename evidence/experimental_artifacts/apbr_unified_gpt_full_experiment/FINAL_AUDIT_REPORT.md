# Final Audit Report

Status: **SEED 0 SHARED-Q COMPLETED; SEED 1/2 PAUSED BY GATE**.

Completed:

- Seed 0/1 compensation attribution and Original/Radius-Guided mask-only counterfactuals.
- Exact Row-wise Q and Shared-Q single-layer numerical gates, including lambda=0 regression.
- Seed 0 full-network Shared-Q fixed-mask comparisons: no compensation, H compensation, APBR-Q, isotropic ridge, and shuffled APBR-Q.
- Seed 0 unified cross-ablation: HH, HQ, QH, and QQ over the registered lambda grid.
- Nineteen-layer q_shared distribution, Exact-to-Shared aggregation error, q/H correlation, reconstruction, compensation RMS, runtime, and VRAM diagnostics.
- All 33 endpoints saved from the same frozen Dense parent with separate checkpoints and bool 2:4 masks.
- Local post-transfer audit: 33/33 masks contain 19 layers, 11,157,504 eligible coordinates, exactly 5,578,752 deletions, strict 2:4 groups, and dense tails; 33/33 checkpoints have finite weights, zero deleted coordinates, matching embedded/external masks, the expected parent SHA256, and `test_accessed=false`.

Integrity and isolation:

- Parent APBR checkpoint identity and full SHA256 are retained in machine-readable endpoint evidence.
- `test_accessed=false` in the authoritative completed status and every endpoint record.
- No Recovery, BN-only update, gradient training, Dense modification, mask retuning, or Final-Test lambda selection.
- The first failed output-directory attempt, caused by a writeback missing `torch.no_grad`, remains preserved; attempt 2 is the authoritative completed run.
- Targeted adapter tests: 5 passed.
- Remote runtime did not contain `pytest`; dependencies were intentionally not changed. The same targeted suite passed locally in the project environment.
- The authoritative remote manifest contains 72 files; post-transfer verification found 0 missing files and 0 SHA256 mismatches.

Gate decision:

- Shared APBR-Q did not beat Isotropic Ridge on task-level train-side diagnostic metrics for any positive lambda.
- Its layer reconstruction advantage was neither broad nor stable.
- q_shared retained some coordinate variation but was close to isotropic and strongly correlated with the Hessian diagonal.
- A small QH CE reduction at lambda 1e-2/1e-1 is recorded as exploratory, not sufficient to override the preregistered gate.

Therefore no Seed 1/2 full-network queue and no Final Test were launched. This preserves the negative result without using Test data to rescue or select the method.
