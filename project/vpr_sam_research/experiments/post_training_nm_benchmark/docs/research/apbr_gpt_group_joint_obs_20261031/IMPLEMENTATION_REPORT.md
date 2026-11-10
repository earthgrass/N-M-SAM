# Group-level Joint OBS Implementation Report

## Implementation outcome

An isolated CPU-validated implementation now exists at `baseline_adapters/group_joint_obs/`. No frozen Original SparseGPT, fixed-mask OBS, APBR training, model, N:M contract, or historical experiment artifact was modified.

The module implements only:

1. the frozen damped OBS system and exact inverse Hessian Q;
2. all six pairwise Static Local Group-OBS costs per output-row 2:4 Block;
3. Group-OBS-only selection;
4. APBR near-optimal selection at caller-provided tau;
5. deterministic within-block shuffled-geometry controls;
6. strict bool 2:4 mask construction and diagnostics.

It intentionally does not implement a second compensation loop. New masks must be passed to the existing `fixedmask_obs_matrix`, preserving one audited SparseGPT-ordered external-mask compensation path.

## Added files

- `baseline_adapters/group_joint_obs/__init__.py`
- `baseline_adapters/group_joint_obs/adapter.py`
- `tests/test_group_joint_obs.py`
- `scripts/run_group_joint_obs_cpu_validation.py`
- this experiment directory's validation evidence and reports

## Frozen definitions

- Pair order: `(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)`.
- Main tau: 1%; sensitivity: 0.5% and 2%.
- Shuffled seeds: 20261031, 20261032, 20261033.
- Q_DD condition-number ceiling: `1e8`.
- Numerical tie tolerance: `64 * eps * max_D(abs(C_D))` within each Block; zero for an all-zero Block.
- Exact ties resolve to the first pair in frozen order.
- Dense tails remain retained.

## Q and Cholesky handling

`prepare_obs_system` reproduces dead-column handling and relative damping before a lower Cholesky of the damped Hessian. It retains the actual inverse `Q = cholesky_inverse(L)` and separately builds the upper SparseGPT factor `U = chol(Q, upper=True)`. Selection indexes the real Q. It never indexes U as if it were Q.

The implementation rejects asymmetric/nonfinite Hessians, failed Cholesky, non-SPD Q_DD blocks, nonfinite condition numbers, and Q_DD condition numbers above the frozen ceiling. It never retries with larger damping.

## Selection and compensation separation

`select_group_mask` zeroes dead input columns exactly as SparseGPT does, computes static local costs from the original dense row, and returns an immutable bool keep-mask plus complete candidate diagnostics. APBR geometry only chooses inside the GPT near-optimal candidate set. Shuffling permutes log-scales independently inside each output-row Block while preserving the Block's geometry multiset.

The compensation contract remains external. CPU regression confirms that replaying a native final SparseGPT mask through `fixedmask_obs_matrix` is bitwise identical in FP32 and FP64 for the controlled system.

## Tests and evidence

- Initial new-module suite: 12 passed.
- Final affected suite after FP32 replay and strict zero-minimum tests: 34 passed.
- Final deterministic evidence runner: `CPU_NUMERICAL_GATE_PASS`.
- No server contacted; no CUDA, dataset, checkpoint, Test data, full model, or Seed queue used.

## Remaining gates

The implementation is ready only for the separately authorized GPU Pilot preflight. Before any real-layer conclusion, the small and large layers must independently verify real activation Hessians, native-mask replay identity, Q_DD numerical condition, memory, runtime, and reconstruction metrics. No full-network Seed 0 or multi-seed execution is authorized by this CPU result.

