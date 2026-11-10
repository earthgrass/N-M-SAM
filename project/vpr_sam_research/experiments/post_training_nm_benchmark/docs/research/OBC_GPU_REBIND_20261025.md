# OBC smoke device rebinding and independent resource audit

**Final resource state: BLOCKED_RESOURCE. No smoke was invoked.**

User approved only the smoke GPU binding change. Previous8cc41ac config is preserved as previous_obc_cuda_smoke.yaml and in Git. Formal/history bindings stay unchanged.
Old UUID: `GPU-fc8784a7-85a2-2bcd-d2e2-f36ef021616c`; new UUID: `GPU-36255fcb-5df2-3c02-2935-79778f083920`.
Old config SHA256: `0c06373b1dc50da13241188c6e1ebf7b8d6e6adcfab9e798c6ed3487dcaa14d8`.
New config SHA256: `7eb57f4c63ce58aa2c03ea6fdc92cd07fcc15a75c3c7bbd802f3fa639e6fafc9`.
Source/config/patch digest: `ef79c95c8817cb43684f3f40d4cf1963a41d60b7436b7866dea5e0d3fc473d96`. The new full Git commit is recorded by the clean post-commit receipt, avoiding a self-referential commit ID.

## Modified files and reasons

* configs/obc_cuda_smoke.yaml: exactly one UUID byte replacement. All other fields/bytes match8cc41ac.
* configs/gpu_execution_manifest.json: add smoke-specific expected UUID and clarify independent resource clearance; global formal expected UUID is retained.
* tests/test_smoke_uuid_rebinding.py: actual runner AST positive/negative identity checks, old/new SHA gates, no parameter/index drift.
* PRE_EXECUTION_ACCEPTANCE.md and post_training_nm_gpu_runbook.md: record resource blocker and current binding; earlier history is not bulk-replaced.
* This report and evidence packet: old YAML, raw server command outputs, bounded samples, CPU JUnit/environment and component hashes.
Runner code, OBC mathematics, coordinate mapping, thresholds8GiB/4GiB,32IDs,batch8,1round, target layer, FP32 flags,2SGD steps, parent weights and SNOWS patches remain unchanged.

## Offline tests

10targeted tests PASS:8new device/config tests and2affected existing freeze/readiness tests. Old complete suites are not rerun. The actual runner identity expression is evaluated with fake properties entirely offline; no CUDA call or smoke entry is executed.

## Independent server observations

Six observations from20:43:17 to20:43:28 Asia/Shanghai all show100%GPU utilization,0%memory utilization,0MiB used,32110MiB free, about120.6–120.8W,42C. nvidia-smi internal71samples over14seconds report99–100%. Compute-app list/pmon contain no PID; direct current-container /proc FD scan finds no NVIDIA holder. Accounting is disabled. /.dockerenv and container PID1bash confirm the visibility boundary.
Only /dev/nvidia1 is mapped into this container as logical GPU0; the actual UUID is matched explicitly. This is not an automatic GPU switch. No host namespace escalation, process kill, reset, driver change, GPU workload or stress test was attempted.
No evidence identifies the cause. Empty container-visible PID lists cannot rule out host/other-user activity or establish a driver reporting fault. Exclusive idle is therefore unconfirmed. Ask the platform/host administrator to verify this exact GPU; do not assume it is free.

## Freeze, deployment and command

After committing the source/config/evidence, the new version is deployed into a separate checkout; original8cc41acserver project/config remain intact. Exact new commit/config hash and the updated command are in FINAL_DEVICE_RESOURCE_PREFLIGHT.json and NEXT_COMMAND_NOT_EXECUTED.txt under this phase artifacts directory.
The command has a BLOCKED_RESOURCE header and must not be run until a later independent resource review passes and the user authorizes execution. Successful rebinding does not clear GPU utilization.
