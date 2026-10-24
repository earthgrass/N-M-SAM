# Dense-SAM-SSAM-Config-FSAM-BN v1.0

Independent seed0 anchor, approved 2026-10-24. Standard two-gradient SAM/SGD,
not FriendlySAM or sparse SAM. The exact user specification is in reference/APPROVED_PROTOCOL.md.
No runtime imports from historical VPR projects. No pruning, recovery, extra seed or search is launched.

The model's BasicBlock/ResNet class text is copied from SSAM2022 commit
76974fd0255cf2f61201e19566cb64ceb429ecdd, models/resnet.py. Registry/unused models removed.
Official class-file SHA is in reference/model_source.json. The frozen seed0 split is copied byte-for-byte,
not regenerated; its original fingerprint describes historical preprocessing, so new preprocessing and
config identity are separately recorded instead of pretending the old whole data protocol is unchanged.

Training: 200epoch/70400steps, LR .05, WD .001, rho .1, batch128 with tail,
epoch-end CosineAnnealingLR, post-normalize Cutout16/p.5, exact specified mean/std.
BN uses batch statistics twice, but restores running_mean/var/count after the second backward.
It is a controlled BN rule inspired by F-SAM, not a claim of identical author helper behavior.
45k/5k held-out validation, deterministic keyed RNG, retained tail and Best Validation selection
differ from the full50k and Best Test behavior of the SSAM code. 96.83 is a reference, never a gate.

```bash
python -m pytest -q tests --basetemp=artifacts/tests-cpu
ANCHOR_TEST_DEVICE=cuda:0 python -m pytest -q tests --basetemp=artifacts/tests-cuda
python precheck.py
# Only after all gates PASS and a verified SOURCE_FREEZE.json exists:
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8 python -u train.py
# Explicit same-run resume, never automatic failure restart:
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8 python -u train.py --resume
```

SOURCE_FREEZE.json contains the committed source SHA map and Git commit. Data is an independently
verified copy at data/cifar10, download disabled. The existing torch2.8/CUDA12.8 interpreter is reused;
there is no import of that environment's historical project modules. No reinstall of old CUDA.

All run products are under artifacts/dense_sam_anchor_20261024/seed0, including exact config,
resolved config, parameter shapes, initialization hash, split IDs/class counts, epoch/batch orders,
gradient diagnostics, latest telemetry, CSV/JSONL metrics, stdout, failure evidence and source-bound
schema2 checkpoint payloads. RNG and optimizer/scheduler states are included. Best selection is
strict Accuracy improvement, earliest tie. Loss at Best Accuracy is explicitly distinguished from
the separate minimum Validation Loss. last.pt and epoch_200.pt retain complete state.
Checkpoint publication validates the temporary AND published paths and writes SHA256 receipts.

Final Test is evaluated once only after exactly200epochs, using best_val.pt selected exclusively
by Validation. An attempt marker prevents accidental repeated automatic evaluation after a crash.
Failures stop the process; no automatic restart, parameter adjustment or follow-on task.
training_summary.md is produced only after the complete trajectory and final Test audit.

The post-training default parent is best_val.pt, as newly authorized. This supersedes earlier
from-scratch restrictions only for future explicitly authorized downstream branches; none are run here.
