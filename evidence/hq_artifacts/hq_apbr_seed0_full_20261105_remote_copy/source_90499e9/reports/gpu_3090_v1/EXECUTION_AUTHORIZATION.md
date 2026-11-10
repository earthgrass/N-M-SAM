# Seed0 explicit research override

On2026-11-05, after reviewing the measured GPU gate, the user explicitly authorized a fresh Seed0 200-Epoch exploratory run. This supersedes the earlier pause recorded in USER_DECISION.json; that earlier record and all failed-gate evidence are preserved.

Original GPU numerical gate: PASS. Original GPU performance gate: FAIL (controller label GPU_SMOKE_BLOCKED). Measured calibration/train ratio0.30552562057541494 exceeds unchanged limit0.30. Execution decision: **EXPLICIT RESEARCH OVERRIDE APPROVED**. The failure is never rewritten as PASS.

Frozen training JSON SHA256 remains `56ff197b0860dfd0f4a0768c5609031214e6405c51557009a9977582b38400e7`. eta0.1, rho0.05, every-Epoch exact Q refresh, optimizer, data,200Epoch budget and no recovery are unchanged. All numerical checks and20/22GiB memory caps remain active.

New orchestration lives in scripts/run_authorized_seed0.py. It checks the original gate file SHA256, the approval file, the unchanged training config, smoke completion, resume PASS, and the same GPU/software. It saves an initial checkpoint, verifies two independently constructed Seed0 initial models match bitwise, and checks the training driver's actual initial model hash. Smoke weights are not loaded. The existing Epoch driver only adds the optional initialization-hash assertion; its training computations and hyperparameters are unchanged.

Outputs use a new `seed0_full_override_v1` directory. The earlier `run_v1` gate/smoke output remains intact. This is a single exploratory Seed0, with final Epoch index199 after78200 steps. Dense/Mask-only/fixed-mask OBS/native GPT checkpoints are frozen before Test. No automatic retry, other seeds, recovery, or formula adjustment is authorized.
