# APBR-SAM post-training N:M benchmark preparation

Independent experiment package; historical artifacts are read-only explicit
inputs. APBR-SAM is the user-confirmed new name for Full VPR-L2 + Geometry-guided.
Historical `fixed_l2`/`vpr_guided` identities remain unchanged. Dense models are
never retrained here. No GPU server was contacted during this preparation.

* [Literature audit](docs/research/post_training_nm_sota_audit.md)
* [Pre-execution acceptance](docs/research/PRE_EXECUTION_ACCEPTANCE.md)
* [SNOWS controlled repair investigation](docs/research/SNOWS_CONTROLLED_REPAIR.md)
* [Approved OBC32-image freeze](docs/research/OBC_SMOKE_FREEZE.md)
* [Approved SNOWS acceptance fix](docs/research/SNOWS_STEP_ACCEPTANCE.md)
* [Source-chain grouping table](docs/research/NM_GROUPING_ACCEPTANCE.md)
* [Existing input audit](docs/research/existing_checkpoint_audit.md)
* [N:M contract](docs/research/nm_constraint_audit.md)
* [Compatibility](docs/research/obc_snows_compatibility.md)
* [GPU execution runbook](docs/research/post_training_nm_gpu_runbook.md)
* [CPU results](docs/research/CPU_VALIDATION.md)
* [Accuracy-only draft](docs/research/accuracy_table.md)

## Local CPU verification

From this experiment directory, using a separate virtual environment:

```powershell
python scripts/fetch_upstream.py --repo OBC
python scripts/fetch_upstream.py --repo SNOWS
python scripts/fetch_upstream.py --repo CAP
python -m pytest -q tests --basetemp=artifacts/pytest-local
python -m benchmark.probes
python -m benchmark.preflight
```

The final command intentionally exits2 with calibration and execution pending.
This is the correct fail-closed outcome, not a formal readiness PASS. CAP has
a colon-containing recipe filename unsupported on Windows: use a Linux checkout
or Git blob for that file; do not rename the official recipe or modify its code.
Only pinned inspected official definitions are loaded; no package import from
another historical experiment occurs at runtime. Do not redistribute unlicensed
official files; preserve each external repository's notices.

Audit15 historical Best checkpoints without rewriting them:

```powershell
python -m benchmark.audit_inputs --input-root "PATH_TO_delivery_20261025_1021" --output artifacts/stage1_inventory.json
```

## Status and artifact layout

`benchmark/`: independent exact model/data, legal Mask, native OBC interface,
layer checkpointing, BN copy, common recovery, provenance and gates.
`configs/`: copied historical protocol and BN IDs, official source lock, verified
input identities, new pending config. `tests/`: synthetic correctness and
resume tests; no CIFAR accuracy experiments. `docs/research/evidence/`: compact
audit evidence. `artifacts/`: ignored external clones, test logs, future outputs.

Future OBC GPU runner: `python -m benchmark.run_obc --help`.
Formal calibration budget/indices/augmentation are **not decided**. The separate
OBC32-image technical smoke is now approved/frozen; see obc_cuda_smoke.yaml and
OBC_SMOKE_FREEZE.md. This is not a formal budget or automatic GPU launch.
Current formal runner only
supports explicitly approved no-augmentation calibration; another augmentation
policy needs its own implementation and tests. GPU run permission is separately
false. SNOWS/CAP are blocked, MOONSHOT/UniPTS literature-only. All formal pruning,
BN evaluations, recovery metrics and GPU costs remain pending.

After a reviewed configuration commit and CUDA checks, the OBC runner preserves
native compensated pre-BN weights, evaluates a separate common BN model, then
uses fresh20epoch fixed-Mask SGD from the pre-BN parent. All20epochs/Best/Last,
Mask and parent hashes are retained. Maximum official Test Accuracy chooses
Best (earliest tie), so it is explicitly **not** an independent unbiased Test
estimate. No OBC optional native BN branch is automatically enabled.
