# Skeptical self-review

Verdict: usable research draft; insufficient evidence for submission claims.

## High priority

- **No final HC result.** Abstract/conclusion explicitly avoid improvement claims. Table1 separates historical references from pending current cohort. Needed: immutable final checkpoint and one-time evaluation JSONs.
- **History not isolated.** Direct-Q control missing. Equation recurrence alone does not establish a benefit. Needed: same protocol eta=1 comparison and matched seeds.
- **Alignment cannot be reconstructed.** training.py accumulates per-layer energies at lines 310-324, losing within-layer support. No invented curves. Needed: future coordinatewise instrumentation; distinguish mean-step from pooled.
- **Prior art narrows novelty.** preSAM includes normalized objective preconditioning, GEAR-SAM includes history and a preserved global radius. Method and Related Work acknowledge both. Needed: empirical evidence that this particular activation signal is valuable.
- **Compute policy failed.** Retain measured 30.55% and original 30% limit. Override is permission, not scientific performance evidence. Do not imply cost-free geometry.

## Mathematical review

- OBS theorem uses a fixed SPD active-space objective and original weight displacement; no incorrect inverse of Hessian principal block.
- Group interactions absent from Q diagonal; explicit correlated two-coordinate counterexample included.
- History exponents sum below one during warm start; unit initial scale accounts for remaining mass.
- Global centering counts output multiplicity and fixes relation to unit-scale noneligible parameters.
- Physical fixed radius valid only for nonzero direction; exact SAM inner-max claim explicitly rejected.
- Budget derivatives are partial derivatives; they are not an APBR feedback law or a monotonic performance theorem.
- Alignment random reference requires complete groups and mask independence. Dense-tail caveat included.

## Protocol review

Native SparseGPT, static mask-only, and fixed-mask OBS remain three endpoints. Training cannot access the final mask. Final calibration inputs come from dense parent. BN fixed during calibration; zero recovery. Hardware/lifecycle differences constrain historical ranking.

No execution changes were made to training, calibration, or original APBR assets for the paper. Human authors must complete source/citation and scope review before submission.
