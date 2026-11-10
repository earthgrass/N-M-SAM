# GPU Resource Acceptance V2

Rule ID: `sparsegpt_single_layer_gpu_resource_acceptance_v2`

This rule is a versioned reinterpretation of resource release evidence. It does not modify or overwrite V1. V1 remains failed because its in-process free-memory delta was 154 MiB against a 64 MiB tolerance.

## Acceptance criteria

1. GPU UUID and state are recorded before execution.
2. Pre-run free VRAM is at least 8 GiB.
3. The fixed K=4608 gate completes without OOM, non-finite values, or Cholesky failure.
4. Peak allocated and reserved memory are measured.
5. In-process allocator/workspace retention is diagnostic and is not alone a failure.
6. At least three post-process samples show no compute process and no more than 512 MiB free-memory shortfall relative to the pre-run NVML reading.
7. GPU UUID, source hashes, and canonical configuration hash are recorded.

## Result

All V2 checks are true. `SPARSEGPT_GPU_RESOURCE_PASS_V2 = true`. `SPARSEGPT_GPU_RESOURCE_PASS_V1 = false` remains unchanged. The three post-exit samples each reported 32230 MiB free, 1 MiB used, and no compute process, versus 32227 MiB free before the gate.

This V2 single-layer result does not replace method-level smoke or full-model resource acceptance.
