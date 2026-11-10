# GPU execution manifest — PENDING, do not launch now

## Current smoke-only device rebinding — resource clearance still BLOCKED

The user approved only changing the OBC technical smoke binding from
GPU-fc8784a7-85a2-2bcd-d2e2-f36ef021616c to
GPU-36255fcb-5df2-3c02-2935-79778f083920. Active smoke config SHA256 is now
`7eb57f4c63ce58aa2c03ea6fdc92cd07fcc15a75c3c7bbd802f3fa639e6fafc9`.
Every other YAML byte is preserved. Formal global expected_uuid, historical
protocols and earlier8cc41ac/0c0637audit records retain their old device identity.
The actual smoke runner already reads c['gpu_uuid']; no runner/algorithm or
resource threshold change was required. Its new-device/old-device rejection
and config-hash behavior were checked using its actual AST condition offline.

Current server samples show100% GPU utilization,0MiB used memory and no visible
compute PID; nvidia-smi's internal sample window shows99–100%. The container
does not provide independent host-side visibility; absence of visible PIDs is
not proof of exclusive idle use. No process was killed, GPU reset, driver change
or workload test attempted. Final status must remain **BLOCKED_RESOURCE** until
idle/exclusivity is verified.8GiB VRAM/4GiB disk thresholds are unchanged.

The runner's existing checks enforce UUID/model, free memory/disk and the visible
compute-process list. They do not certify host exclusivity when utilization
and process visibility disagree. The independent server-resource audit is
therefore mandatory; passing offline readiness alone does not authorize launch.

The post-commit receipt in artifacts/gpu_binding_20261025_* records the new full
commit and config SHA. NEXT_COMMAND_NOT_EXECUTED.txt contains the updated exact
command with those literals and a BLOCKED_RESOURCE header. Do not execute it.
Both the original server8cc41ac checkout/config and the new versioned checkout
are retained; data, checkpoints and previous logs are shared read-only. The
user must review resource clearance again before any smoke invocation.

## Approved32-image OBC smoke — current stage (supersedes earlier proposals)

User approved32official training IDs, seed0/batch8/noaug/1round, parent Normalize,
layer1.0.conv1 only, FP32/no AMP/TF32/compile,2fixed-Mask SGD steps and independent
BN copy. Config: `configs/obc_cuda_smoke.yaml`. Frozen IDs and file/canonical
hashes: `configs/obc_smoke_train_ids.json`; `configs/obc_smoke_freeze.json`
binds the parent checkpoint, actual parent data.py/config source hashes,
preprocessing AST, five training-file SHA256 values and each ID's file/row.
The frozen file is loaded at runtime and never resampled. Parent checkpoint
SHA remains63c2415476802ee30029d445bd2b1fed0992ec39ab0c89a852cf14b1c9683897.

The dataset provenance is verified offline against the original checkpoint-bound
manifest. Raw dataset files are on the shutdown server; actual bytes/order are
strictly rechecked before future CUDA pruning. This is READY **for** a technical
test when the clean-commit readiness receipt passes, not CUDA PASS. The original
formal calibration/execution budgets remain unapproved. Nothing runs this turn.

After final commit, run this offline command using the actual local delivery root:

```bash
python -m benchmark.smoke_readiness \
  --verify-input-root "$INPUT_DELIVERY_ROOT" \
  --expected-commit "$FROZEN_COMMIT" \
  --expected-config-sha256 "$SMOKE_CONFIG_SHA256" \
  --output artifacts/approved_smoke_stepfix_20261025/FINAL_READINESS.json
```

After that receipt reports READY_FOR_GPU_SMOKE, the user can start the server
for **OBC only**. Transfer the clean frozen project, three small approved config/
index/proof JSON/YAML files (preserving bytes), and existing seed0 Dense-SAM Best
at its relative delivery path. Use already-existing official CIFAR10 data and
the approved CUDA environment; do not install requirements-cpu.txt over it.
SNOWS has no approved full-GPU command and is not queued alongside OBC.

```bash
mkdir -p artifacts/logs
set -o noclobber
nohup python -u -m benchmark.obc_smoke \
  --config configs/obc_cuda_smoke.yaml \
  --expected-commit "$FROZEN_COMMIT" \
  --expected-config-sha256 "$SMOKE_CONFIG_SHA256" \
  --input-root "$INPUT_DELIVERY_ROOT" --data-root "$CIFAR10_ROOT" \
  --output "artifacts/cuda_smoke/obc_seed0_${RUN_ID}" \
  > "artifacts/logs/obc_smoke_${RUN_ID}.log" 2>&1 < /dev/null &
```

This command is prepared, not executed. Variables must come from the final
freeze/READY receipt. Reserve8GiB free VRAM and4GiB disk (planning values;
no GPU measurements yet). GPU UUID/free-memory/competing-task/disk gates remain
runtime checks. Full logs, resource samples, three separate native/BN/recovery
files and failures are preserved. Optimizer reload now checks exact state and
groups, not only finiteness; each optimizer step checks masked momentum zeroes.
No20epoch recovery, full-model pruning, Test evaluation or next seed follows.

SNOWS minimal rejected-step bug is now fixed in an additional isolated patch.
Armijo batch/gradient mismatch and complete CIFAR18 feature/Mask bridge remain
blocked; see SNOWS_STEP_ACCEPTANCE.md. The historical sections below record
previous unapproved proposals and later formal plans; they are not new approvals.

## Latest stage: separate technical smoke, not the formal runner

OBC status is **BLOCKED_CALIBRATION_APPROVAL**, with config review also pending.
SNOWS status is **BLOCKED_SOURCE**, despite isolated repaired-kernel CPU tests.
The original OBC adapter/BN/model/data files still match the prior accepted
source hashes; new smoke orchestration and an independent SNOWS diagnostic
patch are separately recorded in `evidence/smoke_readiness_20261025/`.

Do not start the server now. First approve the OBC smoke calibration/config,
freeze exact training IDs and code/config hashes locally, and obtain
READY_FOR_GPU_SMOKE from the offline readiness command. SNOWS can remain
blocked independently; it does not prevent an approved OBC-only technical run.
This stage launches nothing even after an offline gate passes.

### Calibration suggestions — NOT approved, not active defaults

| Scheme | Samples / passes / batch / preprocessing | Meaning and cost |
|---|---|---|
| OBC native reference |1024 ImageNet IDs,10 loader rounds, batch128 default; native RandomResizedCrop224/Flip/Normalize; noaug defaults1round with resize/center-crop |10240 augmented image visits; same IDs across rounds. Not a ready-made CIFAR recipe. H storage/greedy inverse cost independent of N; repeated forward cost grows with rounds. |
| OBC proposed technical smoke |32 distinct CIFAR train IDs, seed0,1round, batch8, no augmentation, parent's exact ToTensor+SSAM Normalize |4 calibration forwards for one eligible576-column layer, separate BN copy,2SGD steps. This is a proposal only: active config fields remain null. |
| OBC proposed controlled formal budget |1024 train IDs,1round, batch32, no augmentation, parent Normalize |32 calibration forwards per layer; memory bounded by batch/layer. Less view diversity than native10round augmentation; cannot call it native configuration replication. |
| SNOWS native references |prune_NM parser ngrads128; README N:M example3000 for ResNet50/CIFAR100/1:4, batch128, k_step30 |ngrads is cached image count: new batch1 shuffled loader -> xdata/nsamples -> xdata2. Cache is reused for Newton/HVP passes. It is not optimizer step count;128 and3000 are not equivalent configurations. |
| Proposed future SNOWS controlled budget |same approved training IDs/noaug/parent Normalize as OBC if a common budget is chosen |For example1024 is only a candidate common count, not selected. This enlarges128 or reduces3000, changes native calibration/normalization, and does not equalize HVP/Newton compute. k_step/Newton/CG/batching still require review after solver/model blockers are resolved. |

No production sample-index file has been generated by this preparation. Smoke and formal
budgets require separate approval; accepting32-image technical smoke would
not approve1024-image formal pruning. Existing4096 BN-only IDs remain separate.
All preprocessing/augmentation/seed/round choices above are recommendations.

### OBC exact future smoke command

Config: `configs/obc_cuda_smoke.yaml`. It has review.approved=false and an
entirely pending calibration block. The formal config's execution.approved
remains false. Checkpoint: existing standard Dense-SAM seed0 best_test.pt,
SHA256 `63c2415476802ee30029d445bd2b1fed0992ec39ab0c89a852cf14b1c9683897`;
relative path `best_checkpoints/seed_0/dense_sam/best_test.pt` under input root.

```bash
# Run readiness locally after approval/freeze; no server access or job launch.
python -m benchmark.smoke_readiness \
  --verify-input-root "$INPUT_DELIVERY_ROOT" \
  --expected-commit "$FROZEN_COMMIT" \
  --expected-config-sha256 "$APPROVED_SMOKE_CONFIG_SHA256" \
  --output artifacts/reviewed_smoke_readiness.json

# Only after READY_FOR_GPU_SMOKE, explicit later authorization and server available:
mkdir -p artifacts/logs
set -o noclobber
nohup python -u -m benchmark.obc_smoke \
  --config configs/obc_cuda_smoke.yaml \
  --expected-commit "$FROZEN_COMMIT" \
  --expected-config-sha256 "$APPROVED_SMOKE_CONFIG_SHA256" \
  --input-root "$INPUT_DELIVERY_ROOT" --data-root "$CIFAR10_ROOT" \
  --output "artifacts/cuda_smoke/obc_seed0_${RUN_ID}" \
  > "artifacts/logs/obc_smoke_${RUN_ID}.log" 2>&1 < /dev/null &
```

Use unique RUN_ID and retain failures; no automatic resume/retry of a failed
technical smoke. It prunes only layer1.0.conv1, leaving the rest dense;
therefore it is not a full-model2:4 result. No Test accuracy is measured.
It hashes/loads the original parent, verifies official data provenance and
GPU UUID/free memory/no competing task, runs official compensation, validates
the partial Mask and finite quadratic diagnostics, saves native_partial.pt,
then independent bn_only_smoke.pt and2-step recovery_smoke.pt. Recovery starts
before BN calibration. Reload validates weights/Mask/optimizer; resources.jsonl
records allocation/peak/disk. SMOKE_REPORT.json records time, output bytes and
CUDA_TECHNICAL_SMOKE_PASS, which is not a formal20epoch acceptance receipt.
FAILURE.json preserves an exception; the process stops. No next GPU job starts.

Reserve **8GiB free VRAM and4GiB disk for this bounded smoke**. The selected
576-column H is≈2.53MiB FP64 and parallel2 inverse matrices≈2.53MiB FP32;
model copies, full-model activations/SGD and CUDA workspace dominate. Planning
reservation, not measured GPU usage. Checkpoints total roughly180–250MB plus
temporary writes/logs;4GiB leaves headroom. Full-size OBC still reserves≥12GiB
free VRAM as a conservative pilot planning value and≥22GiB disk; actual peak
and timing remain unknown. Maximum4608-column inverse batch32≈2.53GiB alone.

For SNOWS, no trustworthy full-smoke VRAM estimate or executable launch is
available while the full solver is blocked. Raw xdata+xdata2 at CIFAR32 FP32
is≈3MiB for128images or70.3MiB for3000; caching both64x32x32 feature tensors
would be≈64MiB or1.46GiB, before autograd/HVP/hooks. A common sample count
does not equalize memory or runtime. Do not reserve server time to debug these
offline source/config decisions.

The following older sections describe a later formal pilot/queue only. They
are **not** the current smoke command or authorization.

Config: `configs/post_training_nm_benchmark.yaml`, version
`preexecution-acceptance-20261025`. Exact existing Dense-SAM paths, SHA256,
producer commits and epochs are listed for seeds0/1/2 in
`configs/existing_inputs.json` (role=standard_dense_sam); paths are relative to
the supplied delivery root. Source and current config digests are in
`evidence/acceptance_state.json`. That is a preparation snapshot, not approval.

The server is shut down. No SSH connection or GPU operation was performed.
Calibration and execution permission are independent pending gates. The exact
OBC CLI is prepared but **has not been CUDA smoke-tested**. No SNOWS/CAP queue
is supplied as if their source had passed. No existing E0/E1 reruns occur.

## Prerequisites and freeze

1. Decide calibration samples, train IDs, SHA256, augmentation and rounds.
   No common number is selected. Current OBC loader supports explicit no-aug
   only; other choices require an implementation review before freeze.
2. Prepare an isolated Linux project checkout of this branch. Install the
   historically paired environment if feasible: Python3.12, torch2.8.0+cu128,
   torchvision0.23.0+cu128; record actual NumPy/PyYAML/driver versions. A different
   software environment is a declared comparison limitation and needs review,
   not an unrecorded substitution. CPU test environment is documented separately.
3. Transfer only the three existing **Dense-SAM** Best inputs, retaining the
   relative `best_checkpoints/seed_N/dense_sam/best_test.pt` paths and SHA256.
   APBR results are reused as evidence, never mixed into OBC's input.
4. Use existing official CIFAR10 files, download=False. Recheck MD5, full
   per-file SHA256 manifest and counts50000/10000 against each dense parent.
5. Confirm RTX5090 logicalcuda:0, approved UUID
   GPU-fc8784a7-85a2-2bcd-d2e2-f36ef021616c, no existing compute task, driver/CUDA
   compatibility, at least22GiB free at start and continuing free-space monitor.
   If a new server has a different UUID, stop for approval rather than silently
   accepting it. Do not delete historical outputs to satisfy capacity.
6. Fetch exact OBC pin; rerun CPU suite plus small CUDA version of adapter,
   weight projection, BN-copy and recovery-resume tests. Audit actual memory
   and deterministic numeric behavior. CPU PASS is insufficient.
   A reviewed CUDA smoke receipt must include status=PASS, source_sha256
   (from source_identity) and config_object_sha256 (object_sha of final config).
   Set its explicit path in execution.cuda_smoke_receipt during configuration
   freeze; the formal runner rejects missing/stale receipts. No such receipt
   exists or is manufactured by these CPU checks.
7. Freeze reviewed config, indices and scripts in a new explicit clean commit;
   only then set execution approval as part of that reviewed config. Save source
   digest/config SHA256/environment/data identities. Do not edit during runs.

## Exact future commands (templates, not executed)

Run inside the isolated experiment directory. Explicit paths are supplied at
execution, so no historical Python module path is imported.

```bash
python scripts/fetch_upstream.py --repo OBC
python -m pytest -q tests --basetemp=artifacts/pytest-server
python -m benchmark.preflight --gpu --output artifacts/GPU_PREFLIGHT.json
mkdir -p artifacts/logs
nohup python -u -m benchmark.run_obc \
  --config configs/post_training_nm_benchmark.yaml \
  --expected-config-sha256 "$APPROVED_CONFIG_SHA256" --expected-commit "$FROZEN_COMMIT" \
  --seed 0 --input-root "$INPUT_DELIVERY_ROOT" --data-root "$CIFAR10_ROOT" \
  --output artifacts/obc_seed0 \
  > artifacts/logs/obc_seed0.log 2>&1 < /dev/null &
```

`run_obc` itself rechecks all gates; do not run the launch lines if preflight
fails. Current config returns failure for calibration and permission. Seed0
is the one-seed technical pilot: full native pruning, common BN-only copy,
20epoch fixed-Mask SGD, all artifact hashes, accuracy, runtime/memory and
integrity review before authorizing seeds1/2. Never run seeds concurrently.

The reviewer supplies `pilot_review.json` with approved=true, source_sha256
and config_sha256 matching the pilot identity. This is an explicit human review
record, not automatically generated merely because accuracy looks plausible.

```bash
python -u -m benchmark.run_obc --seed 1 \
  --config configs/post_training_nm_benchmark.yaml \
  --expected-config-sha256 "$APPROVED_CONFIG_SHA256" --expected-commit "$FROZEN_COMMIT" \
  --input-root "$INPUT_DELIVERY_ROOT" --data-root "$CIFAR10_ROOT" \
  --output artifacts/obc_seed1 --pilot-review artifacts/pilot_review.json
# Only after seed1 finishes without any failure:
python -u -m benchmark.run_obc --seed 2 \
  --config configs/post_training_nm_benchmark.yaml \
  --expected-config-sha256 "$APPROVED_CONFIG_SHA256" --expected-commit "$FROZEN_COMMIT" \
  --input-root "$INPUT_DELIVERY_ROOT" --data-root "$CIFAR10_ROOT" \
  --output artifacts/obc_seed2 --pilot-review artifacts/pilot_review.json
```

A failure stops that process with traceback/evidence. Do not automatically
retry, change precision or numerical settings. After identifying a safe external
interruption, explicitly resume the same run using the same command + `--resume`.

Exact seed0 resume command:

```bash
nohup python -u -m benchmark.run_obc --seed 0 --resume \
  --config configs/post_training_nm_benchmark.yaml \
  --expected-config-sha256 "$APPROVED_CONFIG_SHA256" --expected-commit "$FROZEN_COMMIT" \
  --input-root "$INPUT_DELIVERY_ROOT" --data-root "$CIFAR10_ROOT" \
  --output artifacts/obc_seed0 \
  > artifacts/logs/obc_seed0_resume.log 2>&1 < /dev/null &
```

Never auto-retry the original log filename; preserve failures and resumed logs.
OBC restores its last healthy completed layer; interrupted in-layer work is
recomputed. Recovery restores healthy epoch boundary, optimizer, scheduler,
Mask, RNG and history. Changing source/config/environment/parent rejects resume.

## Resource budget and output accounting

No measured OBC GPU timing exists. Reserve an exclusive pilot slot rather than
promise an overnight completion. Planning-only slot: up to4hours for native
pruning and20epoch recovery, then review measured runtime before booking the
remaining two seeds; this is **not** an empirical completion estimate or a
timeout/early-stop rule. SNOWS time is UNKNOWN while source is blocked.

Largest H:4608² FP64≈162MiB;32 parallel inverse matrices FP32≈2.53GiB,
plus model copies, Hessian inversion/workspace and batches. Measure actual peak.
Three new OBC recoveries save22files/seed at roughly90MB checkpoint with SGD
state, about6GB plus native boundary/final and BN outputs; budget22GiB includes
headroom for6recoveries if a second candidate becomes runnable. Do not create
a second full queue based on this planning estimate. Disk growth needs active
monitoring on the eventual server; disk/OOM/nonfinite failures remain explicit.

Outputs per seed: identity.json, native/native_boundary.pt,
native/native_complete.pt (pre-BN compensated), common_bn.pt,
post_pruning_metrics.json, recovery/epoch_001..020.pt, best_test.pt, last.pt,
epochs.json, checkpoint_hashes.json, completion.json and unique failure logs.
Native original output is never replaced by its BN-only copy. Best uses Test
accuracy, earliest tie; Final and Loss remain available internally.

## Explicit remaining CUDA tasks

Small CUDA smoke tests, runtime/disk monitoring integration, full-size memory
validation, OBC pilot and reviewed multi-seed queue are pending. SNOWS/CAP need
separate algorithm/source decisions. This document does not claim that blocked
methods have a complete executable GPU implementation.
