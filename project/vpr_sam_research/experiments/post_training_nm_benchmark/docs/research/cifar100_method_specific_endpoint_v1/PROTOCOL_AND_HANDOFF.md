# CIFAR-100 method-specific endpoint reporting protocol

## Protocol identity

- Comparison protocol: `cifar100_method_specific_endpoint_v1`
- Base training protocol: `cifar100_50k10k_final_v1`
- Base config SHA256: `e4be5df7cf39b1ebd2730999cd6fd42458a5bcc0779e621894b1c588616879d6`
- Data/model/optimizer/order/preprocessing/N:M scope remain unchanged.
- Test is used only for independent evaluation of fixed endpoint artifacts.

## Execution versus reporting

The active queue continues the previously frozen execution strategy: fixed-mask Recovery-20 for all five pruning methods and 12 endpoint evaluations per seed. No running training or controller is restarted, replaced, or shortened.

The reporting overlay selects each method's registered comparison endpoint:

| Method | Parent | Native pruning semantics | Primary endpoint |
|---|---|---|---|
| Magnitude | Dense-SAM epoch 200 | Stable magnitude score, bool 2:4 mask, zero | Raw Hard |
| Wanda | Dense-SAM epoch 200 | Activation-aware score, bool 2:4 mask, zero | Raw Hard |
| SparseGPT | Dense-SAM epoch 200 | Dynamic N:M selection plus sequential OBS compensation | Native compensated Raw Hard |
| ROSE | Dense-SAM epoch 200 | ROSE reorder, SparseGPT compensation, inverse mapping | Native compensated Raw Hard |
| APBR-SAM | APBR-SAM epoch 200 | Geometry score, bool 2:4 mask, zero, fixed-mask recovery | Recovery Final epoch 20 |

APBR Raw Hard is retained as a diagnostic endpoint. BN-only artifacts are retained for audit and are not main endpoints. Magnitude/Wanda/SparseGPT/ROSE Recovery-20 results remain supplemental even though the active queue executes them for all seeds. Main and supplemental results are never mixed in one ranking.

## Source-call semantics

- The versioned endpoint runner calls the existing frozen `legacy.prune` implementation. It does not reimplement scoring, reordering, or compensation.
- Magnitude uses `benchmark.nm.magnitude_mask` followed by in-place masking.
- Wanda's adapter validates that retained weights are unchanged and only masked weights are zeroed.
- SparseGPT's adapter performs damped Hessian factorization, dynamic per-group column selection, and sequential OBS error propagation before writing compensated weights back to the layer.
- ROSE constructs a column permutation from the registered loss rule, calls the SparseGPT compensation core in reordered coordinates, then writes weights and masks back through the inverse permutation.
- The Raw-Hard checkpoint is saved immediately after the native prune call.
- BN-only calibration runs on `copy.deepcopy(model)`, so it cannot overwrite Raw Hard or the APBR recovery parent.
- Only APBR-SAM creates an optimizer and runs 20 epochs. Its recovery payload binds `parent.sha256` to the exact APBR Raw-Hard checkpoint.

## Remote state inspected before change

At `2026-10-27T10:23:56+08:00`:

- GPU: NVIDIA GeForce RTX 4080 SUPER, UUID `GPU-15f4db77-e52f-c08d-e296-eb814b27dcef`.
- Old controller PID: `26844`, phase `seed1_dense`.
- Seed 1 Dense-SAM PID: `29401`, epoch 59/200, 23,069 steps, `test_accessed=false`.
- Seed 2 had not started.
- Remote branch: `research/post-training-nm-benchmark`; remote Git HEAD remained `9ff061a009e19e619a8a8ae3f5b08b494c83b8d0`; deployed research files were untracked from that remote checkout and therefore are identified by SHA256 rather than remote HEAD alone.
- The old controller had imported its queue plan into memory. Editing its script on disk would not alter that live plan. Its training child used `start_new_session=True`, allowing the controller to be retired without signaling the healthy training process.

## Live-controller decision

No controller handoff was performed. The alternative method-endpoint controller was never started, and no method-endpoint controller PID file exists. The active `run_cifar100_50k_full_queue.py` process remains responsible for Seed 1 and Seed 2 and continues the full Recovery-20 plan loaded in memory.

The reporting overlay does not change failure or retry behavior. No failed method is retried automatically.

## Result generation

`summarize_cifar100_method_endpoints.py` creates:

- `METHOD_ENDPOINT_MAIN.csv`: per-seed method-specific endpoints, Test loss/count/hash, and delta versus the corresponding parent.
- `METHOD_ENDPOINT_SUPPLEMENTAL_RECOVERY20.csv`: Recovery-20 results remain explicitly supplemental; any genuinely missing values are blank/null rather than zero.
- `METHOD_ENDPOINT_RESULTS.json`: per-seed rows, three-seed mean and sample standard deviation (`ddof=1`), same-seed APBR-versus-baseline paired differences, and provenance.
- `METHOD_ENDPOINT_RESULTS.md`: human-readable main and supplemental tables.

The method-specific main table is a complete-flow comparison, not a strict causal ablation of the scoring rule.

