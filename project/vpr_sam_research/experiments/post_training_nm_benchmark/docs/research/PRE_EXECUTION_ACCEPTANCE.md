# Pre-execution audit and acceptance — 2026-10-25

## Latest smoke device rebinding and independent resource audit

User authorized the technical smoke UUID change only. New smoke UUID is
GPU-36255fcb-5df2-3c02-2935-79778f083920; configuration SHA256 is
7eb57f4c63ce58aa2c03ea6fdc92cd07fcc15a75c3c7bbd802f3fa639e6fafc9.
Old8cc41ac/0c0637configuration and audit evidence are preserved. The new
device/config tests prove all other smoke bytes unchanged, actual runner
identity acceptance/rejection, old config hash rejection and unchanged sample
index.10targeted offline tests passed; no full old suite was repeated.

Six2second-spaced server observations all report100% GPU utilization,0%memory
utilization,0MiB used,32110MiB free and an empty compute-app list. pmon also
finds no process. Driver's71sample window reports99–100% utilization. PID1is
container bash, /.dockerenv exists, and direct visible-proc FD inspection finds
no NVIDIA device holder. Accounting is disabled. Current container visibility
cannot rule out host/other-user activity or establish a metric fault; no cause
is asserted. No host namespace entered, process killed or GPU work launched.

Conclusion: **BLOCKED_RESOURCE**, independent of successful UUID rebinding.
Resource thresholds, algorithm, SNOWS patches, parent checkpoint and index are
unchanged. See OBC_GPU_REBIND_20261025.md and its evidence packet. The updated
start command is recorded after commit but remains explicitly not authorized.

## Latest authorization: approved OBC freeze and SNOWS acceptance repair

Continued actual clean HEAD4002ab3; no branch rebuild/reset. OBC32train images,
seed0/batch8/noaug/1round/parent Normalize are now explicitly approved and frozen.
See configs/obc_cuda_smoke.yaml, obc_smoke_train_ids.json and obc_smoke_freeze.json.
Parent checkpoint SHA/data-manifest/config/source bindings were read-only
verified; actual Images class AST matches parent data.py. IDs have canonical
and byte SHA256 plus per-file/row identities; no Test image participates.

The smoke runner checks those identities, dataset order, FP32/no AMP/TF32,
one eligible layer, finite compensation/diagnostics, independent BN, zero masked
weights/momentum at each of2SGD steps, exact optimizer state reload and immutable
native output. GPU/resource/error gates remain implemented but not CUDA-tested.
After a clean final commit, the offline FINAL_READINESS.json in
artifacts/approved_smoke_stepfix_20261025 is the authoritative launch-readiness
receipt. It is generated after commit and is not a CUDA acceptance or permission
to start a task automatically. Full OBC calibration/execution remain unapproved.

SNOWS's separately authorized acceptance patch now prevents a rejected W_new
from being reapplied, returns compatible2item results with explicit status,
and records status at both active callers. Nine step tests and two actual-model
interface probes pass after preserving/fixing initial test-harness failures.
Armijo batch/gradient data and the previous HVP/CG remain unchanged. Complete
CIFAR18 representation/Mask equivalence still fails acceptance: BLOCKED_SOURCE.
See SNOWS_STEP_ACCEPTANCE.md and the additional independent patch.

Existing29+18test evidence was reused; only new/affected paths ran. This stage
has9OBC tests (5new/4affected) and11SNOWS tests, with7targeted harness retries
retained separately. No old full suite, historical experiment or server task ran.
Code/config/source hashes and source before/after patch hashes are recorded in
the current phase's evidence and freeze report. All old evidence remains intact.

## Latest continuation: GPU smoke preparation

Continued the existing branch/HEAD551c891 without rebuilding it. Before any
new edit, every prior accepted source digest matched: no stale-source changes
were found. Original OBC adapter, model/data, compensation and BN components
were reused without rerunning the previous full suite or historical experiments.

Current status: OBC=BLOCKED_CALIBRATION_APPROVAL (config review pending);
SNOWS=BLOCKED_SOURCE. OBC has a separate bounded partial-layer CUDA smoke
runner/config and CPU-tested micro-recovery/resource-monitor/gates. Both smoke
and formal calibration fields remain unapproved; no new sample IDs are frozen.

Fresh official SNOWS Git refs still point to the lock. An independently labelled
diagnostic patch fixes HVP operator decomposition and CG edge cases and makes
the module parse with an explicit unused-placeholder failure. It does not
constitute full optimizer acceptance.12new SNOWS cases now pass; initial3
dependency/test-interface failures and their3case retry are retained.6new OBC
smoke/gate/monitor cases pass. Old complete suites were not repeated. See
[controlled patch report](SNOWS_CONTROLLED_REPAIR.md) and
[smoke runbook, proposals and resource budgets](post_training_nm_gpu_runbook.md).

Further SNOWS solver changes stopped: exact original statements show a
rejected Armijo step can be reapplied after reset. Full CIFAR18 representation
bridge/explicit-zero Mask semantics also remain unresolved. No change to
line-search behavior, objective, regularization or model was silently made.
An earlier dense-target lifetime suspicion was withdrawn after full-control-flow
inspection and is not counted as a defect.

Current source/config/patch digests, changed-file list relative to551c891,
new JUnit files, official-ref check and exact solver-fragment evidence are
stored separately under `evidence/smoke_readiness_20261025/`. Earlier evidence
and its hashes below remain a historical snapshot. No GPU/SSH/formal task ran.

Static checks: Python sources compile. Git whitespace checking passes for
source/config/docs; the byte-preserved upstream patch and raw failed-test XML
contain original trailing spaces in diff context/traceback text. Those artifact
warnings are retained rather than editing original evidence. No numerical
tolerance or algorithm safety gate was relaxed.

## Previous acceptance snapshot (551c891)

CPU preparation accepted within the stated synthetic/inspection scope. **Formal execution remains BLOCKED.**
Server not contacted; no formal pruning, recovery queue or dense retraining. Historical files untouched.

## Actual state inspected

Branch `research/post-training-nm-benchmark`, parent HEAD `d0025b48664091444cc1ad363253010c336b23e6`. Preparation changes were staged/uncommitted when inspected; this report captures pre-commit source hashes, not a formal-run freeze.
Read existing upstream lock/config, OBC adapter/pipeline, common BN/evaluation/recovery and their tests, initial15-case JUnit and runtime metadata, all15 checkpoint inventory/geometry/mask records, copied source protocol/BN IDs, historical per-seed CSV/statistics, original Mask/postprocess call chain, and pinned native pruning/scorer code.
No redundant full checkpoint reload or rerun of the historical results was needed.

## Tests and evidence

Prior full suite:15PASS. This request added14 distinct cases and reran only affected paths: targeted18PASS, final source/group checks2PASS, closed-CUDA-gate1PASS. Combined evidence covers29distinct tests; repeated assertions are not independent experiments.
Raw JUnit files, exact test runtime, full tracebacks and hashes are in evidence/. Earlier tests were inspected first; the old numerical HVP/CG counterexample was not rerun merely to recreate its report.
New coverage: native permutation/inverse/quadratic correspondence, differing feasible Mask, noncontiguous3x3 weights, explicit native ties, exact BN cumulative mean/unbiased variance/counter and unrelated buffers, every SGD-step fixed Mask/zero weight/zero momentum, kept-weight update, every epoch reload, earliest accuracy tie, Python/NumPy/CPU RNG restoration, and independent missing calibration/config/grouping/CUDA gates.

## OBC and grouping acceptance

[Full source-chain comparison table and proof](NM_GROUPING_ACCEPTANCE.md). Both exact-N legality and conditional quadratic compensation passed CPU tests. Original Conv Unfold H goes through native Linear identity coordinates; original score/compensation/inverse code is unchanged. This intentionally adapts the grouping constraint and is not a strict reproduction of published native-convention numbers.
Native OBC removes earliest tied score; historical Magnitude/Guided keeps earliest tied score. This difference is explicit, not silently normalized. GPU memory, full-size numerical behavior and performance are unverified.

## Candidate blockers

SNOWS pin750497fe: parse failure before dependency/model loading; isolated unchanged function tests show default chunk HVP error1 on an analytic quadratic and zero-RHS CG nonfinite. CAP pinf4cbe696: actual N:M scorer AST call reproduces unexpected n keyword; direct CAPNMHandle uses self.m before assignment. Complete tracebacks, inputs, commits and actual local versions are in acceptance_tracebacks.json.
These are locked-source/integration blockers, not claims that the published algorithms are invalid. An independent model bridge cannot fix parsing or constructors. CAP remains conditional; full SparseML execution was not validated. No patch was applied. Official corrected revision or separately reviewed controlled patch is a future decision.

## BN / recovery acceptance and limits

BN-only copies are independent. Only running_mean/running_var/num_batches_tracked change; original model, weights and unrelated sentinel buffer stay equal. Eval changes no model state.
Synthetic interruption/restoration matches model parameters and buffers, optimizer momentum, scheduler/LR, Python/NumPy/CPU RNG, sample-order hashes, epoch/global steps, Best metadata and Mask/config identity. Every SGD step preserves pruned zeroes and fixed Mask; all epoch files load. Best=maximum Test Accuracy earliest tie; Last=latest healthy epoch, Final=epoch20 when complete.
Resume is at epoch boundaries: batch within next epoch is0, not an arbitrary mid-batch continuation. Sample order derives from restored RNG and the same RandomSampler/loader. Actual multiworker/CUDA continuation remains pending; do not describe this as universal whole-training determinism.

## Calibration and execution gates

Status=PENDING_APPROVAL. Calibration dataset is explicitly train-only. Sample count/IDs/hash/batch/forward passes/rounds/optimization epochs/seed/augmentation/preprocessing remain null. Method-native1024x10 OBC and SNOWS128 parser/3000 example are informational, not approved choices. Common BN4096 remains separate.
Missing calibration fields, unaccepted N:M layout/scope, wrong input or reviewed config SHA, wrong frozen Git commit, missing CUDA receipt, unavailable/wrong/busy GPU and insufficient disk reject execution. execution.approved=false, cuda_smoke_receipt=null. No formal task starts this round even if future prerequisites become available.

## Future commands and pending work

[Single-seed pilot, serial seeds1/2, exact resume commands, inputs/SHA256, logs/outputs, estimates and failure policy](post_training_nm_gpu_runbook.md). Config is configs/post_training_nm_benchmark.yaml.
Current **unapproved** config SHA256: `0bd425ceba5dcf6f16a2a92ce426f925bfe10221ecbd7759345d68ff1e347d31`.
Commands require a later reviewed configuration hash and frozen commit; current placeholders are intentional because calibration is not approved. GPU smoke receipt must bind source/config and pass before formal pilot. Resource monitoring integration, full-size pilot, and any SNOWS/CAP fixes remain unimplemented/unaccepted GPU work. No three-seed GPU queue is authorized or running.
