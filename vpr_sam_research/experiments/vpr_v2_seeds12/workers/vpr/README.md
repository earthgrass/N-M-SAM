# VPR-v2 on the 96.77% SAM protocol

Read `PROTOCOL_FREEZE.md` and machine-readable `CONFIG_DIFFERENCES.json`.
This isolated experiment reuses model/data/scheduler and dual-BN SAM code
from the completed SSAM50k anchor, plus a byte-identical copy of the
previously audited fixed_l2 geometry. Parent history and checkpoints remain
read-only. It imports no historical project Python modules.

Formal execution on the approved server uses the existing Python3.12 /
PyTorch2.8+cu128 environment. Controller invocation from this directory:

```sh
nohup /absolute/path/to/approved/python -u launch.py > artifacts/controller.log 2>&1 < /dev/null &
```

The controller runs mandatory CPU/CUDA tests and real-data smoke, then one
seed0 VPR-L2 dense trajectory, dense integrity audit, three pruning/BN-only
branches and three20epoch recoveries, followed by checkpoint integrity
audits and reports. It stops on any failure and never restarts automatically.

Artifacts: `artifacts/vpr_v2_unified50k/seed0/` for dense Best/Last and full
training/LR/geometry logs; `artifacts/comparison/{sam_magnitude,vpr_magnitude,vpr_guided}/`
for Raw/BN-only and every recovery epoch, Best/Last, metrics and provenance.
Final table, curves and SHA256 inventory are under `artifacts/comparison/`.

All endpoints use official Test; Best Test is explicitly selected on that
set. There is no independent unbiased held-out performance estimate and
only one seed. No extra seed or parameter search is authorized.

CPU check: `python -m pytest -q tests --basetemp=artifacts/pytest-cpu`.
CUDA check: same command with `ANCHOR_TEST_DEVICE=cuda:0` set explicitly.
Source manifest must match the clean committed snapshot before either
formal execution or postprocessing; artifacts are excluded from source hashes.
