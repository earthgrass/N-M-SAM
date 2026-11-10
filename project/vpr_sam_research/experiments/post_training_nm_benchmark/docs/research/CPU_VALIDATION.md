# CPU validation and engineering audit

Only synthetic inputs and read-only historical checkpoint loads were executed.
No dataset accuracy measurement, formal pruning, dense training, recovery
experiment, SSH contact or GPU task occurred. Current local runtime is
Python3.12.10 / torch2.13.0+cpu / torchvision0.28.0+cpu. This differs from the
historical GPU stack and is not a GPU acceptance result.

Reproducible command:

```text
python -m pytest -q tests --basetemp=artifacts/pytest-cpu-final --junitxml=artifacts/cpu_tests.xml
```

The current JUnit file and machine-readable counts are under `evidence/`.
Coverage includes canonical grouping, deterministic ties/dense tails,
conditional quadratic compensation optimum, exact native Conv-H cast/order,
Conv-to-Linear equivalence, nonfinite rejection, explicit Mask when kept weights
are zero, excluded BN/first/last parameters, frozen teacher, healthy layer
boundary resume, separate BN-only copy, artifact digest/identity tampering,
train-ID validation and pending calibration/permission gates. The recovery
integration test interrupts after epoch1 and verifies exact final model,
momentum, learning rates, sample order, metrics and scheduler after resume
against uninterrupted synthetic3epoch training.

The first OBC test run exposed a mistake in the **test reference**: it multiplied
the FP32 matrix product before casting, whereas native code casts to FP64 first.
The reference was corrected to the exact native operation order; no native
formula, tolerance or safety gate was changed. The next suite passed.

Two diagnostic tests deliberately reproduce native SNOWS/CAP failures. Their
PASS means evidence is reproduced, not that those algorithms are validated.
The final-preflight report is intentionally BLOCKED: calibration pending and
GPU execution not authorized. No gate is bypassed.

New artifacts use `schema=2`, `kind=post_training_nm_baseline`, explicit healthy
state/model/Mask hashes and parent identities. This is a separate new artifact
kind, not a claim that the old experiment loader can consume new baseline files.
Original schema2 files are validated read-only with their existing digest
rules; no historical checkpoint is rewritten or retroactively re-identified.
Checkpoint writes fsync a temporary file, reopen/validate, then atomically
publish. Healthy recovery carries optimizer/scheduler/RNG/history and masks;
source/config/environment/parent changes reject resume.

The formal OBC CLI is preparation code: it requires CUDA, a clean source commit,
approved device, free space, reviewed calibration and explicit permission.
Full-size operation, small CUDA acceptance, ongoing resource monitor integration
and end-to-end numerical/result checks remain future gates. No working SNOWS
runner is claimed. Candidate core-source fixes require a research decision.

Independent package model/data files are copies of the verified architecture
and transforms; no runtime import resolves through the historical project's
Python module path. Historical weights and logs are explicit read-only inputs.
All new output paths are inside this isolated experiment's artifacts directory.
