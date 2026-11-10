# Audit Report

Current state: **BLOCKED ON GPU AVAILABILITY; NO SCIENTIFIC CONCLUSION CLAIMED.**

Completed:

- Source/geometry audit establishing `r_i = exp(s_i)` and the relative-scale interpretation.
- Independent implementations for Mask-only construction and row-wise geometry-regularized fixed-mask compensation.
- 22 targeted tests.
- Six frozen Mask-only checkpoints with exact 2:4 structure, unchanged BN state, no compensation, no training, and no Test access.
- Original-versus-Guided mask overlap and retained-weight compensation RMS diagnostics.
- Frozen single-layer pilot protocol and dimensionless regularization schedule.

Not completed:

- Final Test of Mask-only checkpoints.
- Calibration/independent-Train output reconstruction attribution.
- CUDA single-layer solver residual, runtime, and memory profile.
- Any full-network geometry compensation (explicitly unauthorized at this stage).

Reason: the registered RTX 5090 host stopped accepting connections; local PyTorch is CPU-only. Existing results were not reinterpreted as completing these gates.

Decision: there is not yet sufficient evidence to start a full Seed 0/1 geometry-compensation experiment.
