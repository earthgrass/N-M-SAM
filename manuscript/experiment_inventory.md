# Experiment inventory

| Evidence | Actual state | Permitted interpretation |
|---|---|---|
| CPU core numerical audit | 12 initial tests passed; integration audit recorded in HQ reports | Arithmetic/implementation checks |
| GPU numerical/BN/resume gate | PASS | Feasibility, not accuracy |
| GPU performance gate | FAIL: 30.55% calibration/train vs 30% limit | Preserve failed gate |
| Research override | Explicitly approved after pause | Execution authorization only |
| Formal HC seed0 | Training; last paper read-only snapshot 2026-11-05 08:34:51 UTC, 18 completed epochs | No final accuracy |
| Historical SAM native | 77.47% formal; 77.51% separate replay | Keep execution identities distinct |
| Historical APBR native | 76.99% | Reference only |
| Direct-Q matched control | Not run | PENDING |
| Coordinate alignment | Required energy logs absent | Cannot reconstruct from layer totals |
| Multi-seed HC | Not authorized/run | No stability or significance claim |

The old SUMMARY.json describes the paused gate-time state. It is deliberately preserved, not mistaken for current run status. execution_approval.json and the live snapshot supersede its execution decision, not its measurements.

No new Test access, checkpoint evaluation, training, or parameter change was made by the writing task. Evaluation import is a local JSON/hash operation only.
