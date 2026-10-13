# Research roadmap — PLANNED, no implemented research scoring

First M2: pure masked forward vs STE, fixed-support perturbation policy, one global
norm including dense coordinates, unchanged mask version for each SAM step.
Validate support, legality, full rollback and task/optimizer loss separation on CPU.

## Shared diagnostic contract

ProbeContext=(snapshot_id, fixed batch, functional buffers, modes, RNG,
task_loss_definition, parameter layout, support version). No training-state mutation.
Inputs/outputs carry CE-vs-J and true-vs-surrogate derivative labels.
`loss_functional.py`: (Z, context)->true scalar task L, no old mask applied again.
`mask_candidates.py`: (layout, block_id)->six MaskCandidate supports.
`mask_delta.py`: (W,M,M')->MaskTransition(q,supports,snapshot).
`gradient_path.py`: (Z,epsilon,context)->true g1,g2,path label.
`hvp.py`: (Z,v,context)->Hv and cost; small diagnostic samples only.
`curvature.py`: (q,u,Hu,optional Hr)->CurvatureProbeResult with observed,
residual, approximation flags and error reference.
`robust_probe.py`: (Z,M,rho,context,solver_budget)->RobustProxyResult
with solver label, support, value, cost; one step is `one_step_robust_proxy`.
`ranking.py`: (candidate estimates, reference values)->Spearman, Top-1 agreement,
regret, error, runtime; deterministic ties and constant-score handling must be explicit.
All these modules are PLANNED and have no executable placeholder functions.

## A: Direction gap

Estimate Hu from (g2-g1)/rho, then diagnose r^T Hr by small-sample HVP or
[L(Z+h*r)-2L(Z)+L(Z-h*r)]/h^2. r may include old inactive coordinates.
Test constant-Hessian quadratic path identity, q=au+r decomposition, zero rho,
finite difference convergence and no state pollution. Compare observed-direction
only vs residual-informed prediction across six candidates; report absolute error,
Spearman, Top-1 agreement, regret and extra cost. Stop if extra cost brings no
stable ranking improvement across seeds/layers/checkpoints.

## B: Objective gap

F_rho(Z,M)=max_{P_M e=e, ||e||<=rho} L(Z+e), S=F-L;
Delta F=Delta L+Delta S. New support differs from old support.
From the SAME old center, proposed quadratic robust proxy:
L(Z)+g^T q+0.5 q^T Hq+
max_{P_M' e'=e', ||e'||<=rho} [(g+Hq)^T e'+0.5 e'^T H e'].
This is a research proxy, not an implemented exact robust score.
Use four-dimensional symmetric positive/indefinite quadratics, nonzero g,
six supports and an independent trust-region reference to test exact quadratic
optimum vs one-step ascent. Real networks: compare Delta L, Delta S and Delta F
separately on fixed samples. No largest-eigenvalue sharpness assertion unless
projected g is near zero and quadratic locality is verified.

## C: Coupling / compensation

q=q_A+q_B includes q_A^T Hq_B. Compensation U lies on NEW support,
P_M' U=U, ||U||<=delta. Compare min_(M',U) F(P_M' W+U,M') to mask-only.
Ordinary positive-definite quadratic special case:
q_S*=-H_SS^-1(g_S+H_SD q_D), with fixed removed-coordinate q_D.
This overlaps OBS/OBC and is not claimed novel. Test two-block coupling,
indefinite/singular H, damping/budget constraints and legal compensation support.
Start only after A/B evidence; compare to equal-cost ordinary recovery SGD.
Stop if gain disappears under the matched recovery budget.

O3: bucket by ||q||/rho and ||q||; with K-Lipschitz Hessian,
|R3(q+e')|<=K/6 ||q+e'||^3. Small rho alone does not bound mask-switch error.
O4: sample single blocks, then small block groups; never full-network six-way
enumeration or full Hessian by default. Record search/compute budgets.

