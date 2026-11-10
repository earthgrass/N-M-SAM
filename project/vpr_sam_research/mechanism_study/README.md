# Fixed-checkpoint mechanism study

No optimizer or training is constructed. Two preserved ensembles each contain
32 PCG64 Mask controls (seeds 10000–10031; these are **not training seeds**).
PROTOCOL.json is the original coarse 20-quantile-bin registration;
PROTOCOL_FINE.json is the pre-evaluation fine log-gap registration.
Do not choose between or filter ensembles by validation performance.

Use private frozen V2 project commit f742733 and V1 project d4c58d6, their
original dataset/checkpoints, the previously audited masks.pt and reference IDs.
Do not add this script/config to their src/scripts/configs trees: that would
change historical source_hashes. Keep diagnostics in a separate directory.
Use the historical pinned PyTorch 2.8.0+cu128 environment and explicit paths:

```bash
export PYTHONPATH="$V2_FROZEN_PROJECT/src"
export PYTHONDONTWRITEBYTECODE=1
"$AUDITED_PYTHON" /path/to/archive/mechanism_study/run.py \
  --project "$V2_FROZEN_PROJECT" --v1-project "$V1_FROZEN_PROJECT" \
  --protocol /path/to/archive/mechanism_study/PROTOCOL.json \
  --output /new/private/coarse
# Sequentially, after coarse completes and the approved GPU is idle:
"$AUDITED_PYTHON" /path/to/archive/mechanism_study/run.py \
  --project "$V2_FROZEN_PROJECT" --v1-project "$V1_FROZEN_PROJECT" \
  --protocol /path/to/archive/mechanism_study/PROTOCOL_FINE.json \
  --output /new/private/fine
"$AUDITED_PYTHON" /path/to/archive/mechanism_study/audit_controls.py \
  --project "$V2_FROZEN_PROJECT" --folders /new/private/coarse /new/private/fine
```

Do not automatically rerun these completed evaluations. Commands document
reproduction only; require original private weights and dataset authorization.
Each run refuses existing output directories and concurrent GPU activity.
All masks are frozen before any new validation reading. Strict source checks
and exact historical Magnitude/Guided replay stop on mismatch.

Public CLI replaces the executed adapter's private V1 directory literal with
required --v1-project. Computation and matching are unchanged. Executed source
SHA256 remains in each PROTOCOL_FREEZE.json; that exact adapter is retained
privately, not rewritten. Public source SHA is recorded separately. Protocol
files are unmodified and their byte hashes remain verifiable.

Results contain all constructions, hashes, overlaps, per-layer diagnostics,
individual validation rows and independent CPU reconstruction, without large
checkpoints, datasets, server addresses, credentials or device UUIDs.
