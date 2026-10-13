# Research readiness readout

## Hypotheses

A: residual curvature may improve candidate robust-related ranking.
B: new support and robust increase may change ordinary-loss candidate rankings.
C: cross-block coupling/compensation may outperform equal-budget recovery.

## Implementation and unit validation

M2–M7 code and common tools are implemented; current small-tensor unit suite
has 111 passed, zero failures/skips. This is CODE_COMPLETE / UNIT_TESTED evidence.
See IMPLEMENTATION_STATUS, ARCHITECTURE_AND_API and GPU_DEPLOYMENT for boundaries.

## Observations and counterexamples

No authorized GPU A/B/C mechanism observations exist. Analytic quadratic tests
include a one-step robust-proxy counterexample (zero gradient/positive curvature)
and indefinite compensation cases. These validate mathematical boundaries only.
Pre-override smoke records are historical runtime checks, not research evidence.

## Limitations and next gates

DATASET_PENDING; GPU_VALIDATION_PENDING. AMP/DDP execution is not implemented.
Full-seed/epoch accuracy, long-run resume, actual solver cost and native 2:4 speed
are unverified. No new selection policy is installed, no generalization gain claimed.
Authorize server/data checks, bounded GPU smoke, then separate A/B/C probe budgets
before deciding whether any new selection design warrants implementation.
