# Approved minimal SNOWS step-acceptance repair

The original official pin and first math diagnostic patch are preserved.
`patches/snows_step_acceptance.patch` applies after the first patch, changes
only the accepted/rejected commit boundary, adds strict finite checks/status
diagnostics and adapts both active native callers. It does not change objective,
lambda, CG/HVP, candidate sequence, comparison condition or calibration data.

## Exact behavior and compatible return

Each native line search starts step_accepted=False; only the original Armijo
condition sets it True. The final weight write is W_new if accepted, otherwise
W_original. Thus every rejected candidate can still be evaluated exactly as
before, but the final rejected update cannot overwrite the restore. Original
alpha=1, alpha*=.9, c=1e-5,50attempts and comparison expression are retained.

Both return sites now produce SNOWSSolverResult, a tuple subclass containing
exactly(weight,current_loss). Existing unpack/index/length semantics remain2.
Additional .status and .line_search attributes record last-step accepted/rejected
and every attempted search (trials/alpha/loss/dot/grad_W_val norm). Both actual
call sites in Layer_pruner_new unpack the same2items and append status/evidence
to self.solver_diagnostics; no retry or replacement optimizer is selected.
For a multi-step solve, .status describes the last step; the full trace preserves
prior accepted/rejected steps. Nonfinite input/trial restores original weights
and raises SNOWSStepFailure carrying diagnostics, rather than accepting it.

## CPU evidence

Nine distinct new step cases: acceptance on trials1/3/50;50rejections with
bitwise-original weights through subsequent result/caller assignment; nonfinite
gradient and nonfinite loss; original comparison/data assignments/CG/HVP AST
unchanged; both exact active caller statements under both statuses; controlled
full solver module import. Initial harness run had7failures due to missing
CPU device/owner context and an inconsistent fixture dtype, with2cases passing.
Only those7cases reran after fixture corrections and passed. Raw failure/retry
JUnit and environments are preserved. No numerical acceptance condition or
tolerance was weakened. Old HVP/CG/coordinate proofs are reused and the patched
function ASTs are asserted unchanged, rather than rerunning the entire suite.

Two additional actual-model probes confirm remaining incompatibility: native
get_blocks(actual CIFAR18) raises for missing model.name; without assigning a
fake name, native hooks on layer2.0 capture only conv1/conv2 and miss its real
shortcut.0 Conv and functional ReLUs. No forward-equivalence bridge is claimed.

## Armijo data and grad_W_val review — unchanged

Newton's gradient/HVP use the current xdata[batch_start:batch_end]. Armijo always
uses xdata[xdata.size(0)-batch_size:], the final calibration slice. grad_W_val
is computed from that last-batch reconstruction loss but is unused by the
native comparison, which still uses dot(grad_W,current_direction). When batches
differ, this is not the usual same-objective Armijo directional derivative.
If the calibration data fit one full batch, slices coincide; this does not
justify silently restricting the future protocol. The fix records grad_W_val
norm only and preserves the original comparison/data exactly. Changing them
is outside this authorization and requires a separate decision.

## k_step, representations, Mask and remaining boundary

Native k_step is an operation-hook horizon, not SGD epochs or calibration
images. count_operations_in_block skips the starting module, while the hook
routine can include it and uses operation_count>K, making the horizon inclusive.
It counts module Conv/ReLU by child traversal; functional ReLU and this model's
projection shortcut are not covered. Current4-stage/19Conv scope cannot be
mapped by pretending it is native3-stage ResNet20. Residual additions, branch
ordering, exact captured representations and gradient/HVP equivalence must
be defined and tested before a full adapter is accepted.

Native MP_MN locally accepts canonical coordinate groups, with transpose/inverse
mapping already tested. But the caller rebuilds Mask from abs(w_sol)>0, losing
retained zero coordinates. An explicit selection-Mask bridge remains unproved.
All these facts keep SNOWS BLOCKED_SOURCE despite the acceptance fix. The next
minimal work is an explicit representation/operation specification and review
of same-batch versus native-last-batch Armijo semantics, then separate bridge
tests. No other solver behavior was altered or full GPU run prepared.
