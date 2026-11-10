# SNOWS isolated repair investigation — full method remains BLOCKED_SOURCE

Current extension: the user subsequently approved the minimal rejected-step
commit fix. It is implemented in a **second** isolated patch; see
[SNOWS_STEP_ACCEPTANCE.md](SNOWS_STEP_ACCEPTANCE.md). Original math patch and
evidence below are preserved. Full-model/Armijo data semantics remain blocked.

Date2026-10-25. Official checkout/pin stays
750497fe13454253056307956a56d581fedf1128. A fresh `git ls-remote` reports only
HEAD/main at that same commit, no advertised newer branch/tag. History was
fetched without changing checkout files. No newer official correction was
found among those advertised refs; this does not rule out unpublished fixes.
Evidence: `evidence/smoke_readiness_20261025/official_snows_check.json`.

## Explicit patch boundary

`patches/snows_minimal_diagnostic.patch` is an independent review diff;
`benchmark/snows_patch.py` verifies source/file hashes and applies the diff
in memory. Original code is never overwritten. Patched output is not called
unmodified official SNOWS. The full controlled module can import with real
joblib1.5.2 and pinned helpers; this is not a complete LayerPruner integration.

1. The unfinished unused trailing `prune_and_update` receives an explicit
   NotImplementedError body. It is not implemented, skipped silently or used
   by the approved path. A test proves calling it fails.
2. Correct chunked HVP to evaluate the intended Hessian-vector operator.
   If K selects retained coordinates, g_K is the masked gradient. Partition
   v=Σv_j into disjoint input chunks. H_KK v=Σ K^T H K v_j. Therefore each
   chunk contributes **all retained output rows**. Merely fixing the Boolean
   interval and retaining only output rows of that chunk incorrectly discards
   cross-chunk Hessian entries. The patch fixes both issues. Original objective
   and loss reductions remain unchanged.
3. CG retains original x/r/p/alpha/beta recurrence, regularizer1e-4, tolerance
   and iteration budget. Zero/converged initial residual returns zero in0steps.
   Nonfinite inputs/iterates or nonpositive curvature raise explicitly. No
   adaptive damping, trust region, MINRES, clipped update or substitute solver
   is chosen. The candidate may correctly stop on an indefinite actual Hessian;
   it is not guaranteed to finish real SNOWS training/pruning.

## New CPU evidence

Twelve distinct new cases: patched module import/placeholder; four chunk sizes
on dense off-diagonal analytic Hessian; chunked-vs-unchunked equality; gradient,
HVP and regularized Newton direction under permutation/inverse; zero/near-zero/
ordinary SPD CG; unsafe CG rejection; native MP_MN on canonical supplied groups
with transpose/restore; actual CIFAR18 functional-ReLU/projection interface audit.
All12 now pass. Original first attempt had9PASS/3FAIL: joblib absent for2tests,
and FX test failed to specialize optional y=None for1test. Install was isolated
to ignored artifacts/python_deps, joblib pinned1.5.2; only those3cases reran
and passed. Both initial failure and retry JUnit files are retained.
No old complete suite or HVP failure experiment was repeated.

These tests establish kernel identities and local Mask mapping. They do not
prove the full nonlinear reconstruction/Armijo/model bridge correct.

## Further confirmed solver blocker — no behavioral patch applied

In native `solve_for_W_given_Z`, after all backtracking attempts fail, the
failure branch writes W_original, but the unconditional write after the loop
writes W_new again. Exact extracted AST statements reproduce reset=[1] then
final=[7]. Evidence includes the original statements and source locations in
`solver_blockers.json`. Correcting step acceptance changes executable optimizer
behavior, so implementation stops here pending user review. Also review the
use of the last calibration batch versus current Newton batch in Armijo and
the unused `grad_W_val`; no replacement semantics were invented.

An earlier lifetime suspicion was **withdrawn**: last_batch_ydata_dense is
deleted after backtracking, not before use. An out-of-context diagnostic
harness initially selected that cleanup incorrectly; this is recorded as
NOT_A_BLOCKER, not a claimed upstream defect.

## Full N:M/model adapter still blocked

The native MP_MN routine accepts arbitrary group lists; supplying canonical
OIHW consecutive groups preserves its top-|w| projection and matrix transpose
coordinates. Generic permutation tests establish corresponding gradient/HVP/
Newton coordinate identities. However the full LayerPruner derives Mask from
nonzero weights (retained zeros can be lost), selects model-specific stages,
and hooks module ReLUs. Current CIFAR18 has four stages, functional ReLUs and
three eligible shortcut Conv weights. Exact representation target/hook order,
k_step meaning and projection-branch coverage have not been proved equivalent.
No dummy resnet20 name or renamed layers were used to bypass those checks.

Decision: BLOCKED_SOURCE despite successful isolated math tests. A full SNOWS
GPU smoke command is deliberately not supplied. Needed review: controlled
kernel diff; rejected-step behavior and Armijo batch/gradient semantics;
exact CIFAR18 representation bridge; explicit Mask handling; native optimization
budgets and calibration. No official-original reproduction claim is made.
