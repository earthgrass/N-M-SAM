# Blockwise APBR Seed-0 Protocol Amendment V2

Date: 2026-10-28 (Asia/Shanghai)

The original Seed-0 Scheme-4 run failed after completing Epoch 14.  Exact
checkpoint replay located the first failure at Epoch 15 batch index 69, before
global step 5,543.  `layer1.0.conv1.weight` contained 128 complete 2:4 blocks
whose four center-gradient and perturbation coordinates were exactly zero.
The original implementation attempted to calculate relative block energy as
`0 / 0` and deliberately failed closed.

This was a missing mathematical boundary case, not a CUDA, OOM, non-finite
model, or data error.  V2 freezes the following non-arbitrary rule:

- A zero-direction block keeps its original zero perturbation.
- It receives the neutral geometry update `delta_s = 0` because no relative
  energy signal exists.
- Scheme 4 remains byte-identical to the frozen original APBR perturbation.
- Scheme 1 equalizes the preserved eligible energy only across blocks with a
  defined nonzero direction; zero-direction blocks remain zero.
- Residual coordinates and every active block retain their original formulas.
- No epsilon, random direction, clipping, hyperparameter search, Recovery, or
  second-order compensation is introduced.

Evidence:

- Original failure retained under the original `formal/` directory.
- Diagnostic replay recorded in `diagnostics/scheme4_epoch14_zero_energy.json`.
- The repaired implementation replayed the complete next epoch (391 batches),
  reaching 5,865 steps; recorded in
  `diagnostics/scheme4_epoch14_repair_replay.json`.
- Local targeted tests: 5 passed.
- Real RTX 5090 single-batch smoke: PASS for both V2 schemes.

V2 uses a new experiment identity, configuration, output root, and Seed-0-only
queue.  It restarts both schemes from their fixed initial state.  The failed
Epoch-14 checkpoint is not used as a formal training parent.
