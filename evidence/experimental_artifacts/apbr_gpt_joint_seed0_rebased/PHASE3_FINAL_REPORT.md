# APBR-GPT Joint Optimization — Phase 3 final report

## Final states

| Gate | State |
|---|---|
| Historical Seed 0 initialization | Not recovered |
| Replacement cohort | `REBASED_COHORT_READY` |
| Mathematical audit | `MATH_AUDIT_PASS` |
| Real train-only calibration | `REAL_CALIBRATION_PASS` |
| CPU orchestration | `CPU_ORCHESTRATION_PASS` |
| Deployment package | `SERVER_BUNDLE_V3_READY` |

## Asset decision

The historical target initialization hash `87aa9dd7...759d` was not found among 66 loadable model-state records. A new, isolated `apbr_gpt_joint_seed0_rebased` cohort was therefore frozen. Its common checkpoint SHA256 is `4317514e...03c`, model tensor SHA256 is `6c16e81a...2ddc`, and initial FP64 geometry SHA256 is `20771c3a...ae09`. Dense-SAM, APBR-SAM, and Joint APBR-GPT all reference the same initial model. All three must be newly trained; old Seed 0 outputs cannot enter rebased paired statistics.

## Calibration

Calibration used the real frozen ResNet-18, 128 fixed samples from the official CIFAR-100 training pickle, real activation Grams for all 19 eligible layers, and the rebased FP64 geometry. It did not construct an optimizer or mutate parameters, BN buffers, geometry, or RNG state.

- `lambda_p = 1.3526042410542212`; achieved eligible-weight gradient ratio `0.05`.
- `eta_s = 7036.300421582199`; achieved projected scale-update ratio `0.05`.
- Energy update norm: `0.001514993748684254`.
- Independent pruning update norm: `0.0000757496874342127`.
- Update cosine similarity: `0.07361620777287145`.
- Combined projected update norm: `0.0015224455689145488`.

Attempts 3 and 4 produced byte-identical calibration identity, lambda, and eta files. All 19 Hessian non-timing fields were identical; only wall-clock timing varied. Earlier RNG-audit failures remain preserved and are not treated as passes.

## Mathematics and implementation

The APBR global perturbation, energy feedback, global log-volume constraint, and FP64 parameter mapping were checked against the original source. The joint regularizer evaluates all six deletion pairs per output row and group, stops gradients through Q, activation Gram, and the first gradient, and retains weight and epsilon-to-scale paths as specified.

Dynamic Groupwise OBS now applies the exact two-coordinate constrained update, propagates every APBR probe through the same linear map, zeros the selected pair, and freezes the complete four-column group. Explicit SPD references cover row-dependent supports. It is deliberately not described as Original SparseGPT bitwise replay.

## CPU orchestration and cost

The CPU smoke completed a real ResNet-18 forward/backward and one optimizer iteration, a real-activation Hessian refresh, 19-layer small-network Groupwise OBS orchestration, atomic checkpoint save/restore, and a full-scope mask audit. The full audit counted 19 layers, 11,157,504 eligible coordinates, and 5,578,752 deletions with strict 2:4 compliance.

The formal runner fixes 200 epochs, five warm-up epochs, and Hessian refreshes every ten epochs thereafter. It stores each Hessian cache as a hash-checked sidecar so resume does not silently rebuild a different cache. Final evaluation is guarded and separate from training/pruning.

There are 2,789,376 2:4 groups and 16,736,256 candidate evaluations per batch. The engineering estimate remains below the mandatory 3x APBR stop threshold, with approximately 127.7 MiB for the two principal FP32 candidate-cost tensors; true GPU wall time and peak memory remain a required smoke measurement. Deterministic CPU refresh took 27.00 s in attempt 3 and 23.90 s in attempt 4 across all layers.

## Tests and bundle

Eighty-nine targeted and original APBR regression tests passed with zero failures. The v3 bundle SHA manifest contains 77 files. The tarball is 41,463,585 bytes with SHA256 `d8b98256ff17fca186c62d0468871cb7fcbf00351ac487ed7baa1d97038ea884`. Pre-archive and independent post-extraction verification both reported zero failures. The first extraction attempt hit the Windows path-length limit; the same archive then passed in an independent short temporary directory.

## Remaining blockers and stop state

No server was contacted, no CUDA operation was performed, formal training did not start, and Final Test was not accessed. The next authorized stage must upload the v3 bundle, run its integrity and CPU checks, perform the one-GPU smoke, and measure the 3x/runtime-memory gate. Only then may the three rebased branches be trained. This Phase 3 stops here.
