# Group-level Joint OBS CPU Numerical Validation

## Status

`CPU_NUMERICAL_GATE_PASS`

This gate used deterministic synthetic positive-definite matrices on CPU only. It did not load CIFAR-100, construct a Test loader, access CUDA, contact a server, train a model, or prune a complete network.

Environment: Python 3.12.10, PyTorch 2.13.0+cpu, Windows 11. Final affected regression: **34 passed in 5.45 s**.

## Numerical results

| Check | Observed error/result | Status |
|---|---:|---|
| Six pair enumeration | 6 unique pairs in frozen lexicographic order | PASS |
| Q vs direct inverse, max absolute | `8.881784197001252e-16` | PASS |
| Q vs direct inverse, relative L2 | `7.345770020578445e-16` | PASS |
| Upper-factor reconstruction, max absolute | `8.881784197001252e-16` | PASS |
| Upper-factor reconstruction, relative L2 | `7.557137094991966e-16` | PASS |
| Extracted Q_DD vs direct inverse | `6.245004513516506e-16` | PASS |
| Joint cost vs explicit constrained optimum | `1.7763568394002505e-15` | PASS |
| Deleted-coordinate constraint residual | `4.440892098500626e-16` | PASS |
| APBR pair cost vs scalar reference | `0.0` | PASS |
| Shuffled geometry multiset invariant | `1.7763568394002505e-15` | PASS |
| Tau=0 degeneration | Same Group-OBS pair | PASS |
| C_min=0 handling | Only numerical minima retained | PASS |
| FP32 small-cost zero-minimum case | No unrelated absolute candidate window | PASS |
| Stable tie-breaking | First frozen pair selected | PASS |
| Strict 2:4 / dense tail / finite values | All satisfied | PASS |
| Ill-conditioned Q_DD | Explicit hard failure, no damping change | PASS |

## Native OBS replay identity

The native SparseGPT selector was first allowed to produce its dynamic final mask. That exact mask was then supplied to the existing fixed-mask OBS provider using the same dense weights, Hessian, damping, block size, dtype, and column order.

| Compute dtype | Mask equal | Weight bitwise equal | Max abs. error | Relative L2 error | Deleted exact zero |
|---|---|---|---:|---:|---|
| FP32 | yes | yes | `0.0` | `0.0` | yes |
| FP64 | yes | yes | `0.0` | `0.0` | yes |

This proves CPU replay identity for the tested linear system. It does not replace the later two-layer GPU replay gate on the real CIFAR-ResNet18 activations.

## Near-optimal selector checks

The controlled case has three candidates within 1% of the local Group-OBS minimum. Group-OBS selects frozen pair index 0. Correct APBR geometry selects pair index 1 while keeping relative GPT cost increase within 1%. At tau zero, APBR degenerates to the Group-OBS result unless a numerical tie exists.

The C_min=0 test confirms that tau introduces no multiplicative neighborhood. Numerical tolerance is `64 * machine_epsilon * max_candidate_cost_within_that_block`; an all-zero Block has zero tolerance. A dedicated FP32 small-cost case prevents reintroduction of an arbitrary absolute window.

## Preserved failed/superseded evidence

The first evidence-run failure is preserved as `cpu_validation_results_attempt1_failed.json` and `run_attempt1_failed.log`. The algorithm was correct; the audit incorrectly assumed shuffled APBR pair-cost sums remain invariant under nonuniform fixed weights. That assumption was replaced by the correct geometry-multiset check using unit weights.

The second successful run is preserved as `cpu_validation_results_attempt2_superseded.json`. It was superseded after a review found that `max(1, block_cost_scale)` could create an unrelated FP32 absolute tolerance for small-cost Blocks. The final implementation uses only the Block's own cost scale and was rerun from scratch.

## Evidence

- `cpu_validation_results.json`: authoritative machine-readable final gate.
- `candidate_costs.csv`: every row/group/candidate cost and explicit reference objective.
- `pytest_group_joint_obs_final_v2.xml`: authoritative affected regression report.
- `run.log`: authoritative final runner output.
- Earlier attempts are retained and are not PASS evidence.

## Boundary

The CPU gate validates formulas, selection mechanics, replay identity, invariants, and deterministic failure behavior. It makes no claim about GPU cost, real-layer reconstruction, task accuracy, or superiority over Original SparseGPT or shuffled geometry.

