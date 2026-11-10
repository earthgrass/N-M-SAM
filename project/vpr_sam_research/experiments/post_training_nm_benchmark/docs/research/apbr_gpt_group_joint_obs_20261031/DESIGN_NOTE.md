# APBR-GPT Group-level Joint OBS + Geometry-guided Selection

## 1. Research question and frozen boundary

The only new component is 2:4 mask selection. APBR Dense training, its saved geometry, calibration inputs, Hessian construction, damping, numerical dtype, SparseGPT column order, OBS error propagation, model architecture, eligible-layer scope, and N:M layout remain frozen.

The study asks whether persistent APBR coordinate geometry can choose usefully among deletion pairs that are already near-optimal under a local group-level OBS model. It does not introduce APBR-Q, ridge regularization, Recovery, BN calibration, or a new compensation objective.

## 2. Local constrained quadratic derivation

For one output row, let the damped calibration Hessian be symmetric positive definite:

\[
\widetilde H = H + \lambda_{\mathrm{damp}} I,
\qquad Q = \widetilde H^{-1}.
\]

For a deletion pair \(D=\{i,j\}\) in one four-coordinate group, let \(E_D\) select those coordinates. Deleting them imposes

\[
E_D^\top \Delta w = -w_D.
\]

Consider

\[
\min_{\Delta w}\;\frac12\Delta w^\top\widetilde H\Delta w
\quad\text{s.t.}\quad E_D^\top\Delta w+w_D=0.
\]

With Lagrange multiplier \(\nu\),

\[
\mathcal L(\Delta w,\nu)
=\frac12\Delta w^\top\widetilde H\Delta w
+\nu^\top(E_D^\top\Delta w+w_D).
\]

Stationarity gives

\[
\widetilde H\Delta w+E_D\nu=0
\quad\Longrightarrow\quad
\Delta w=-Q E_D\nu=-Q_{:,D}\nu.
\]

Applying the constraint,

\[
-Q_{DD}\nu=-w_D
\quad\Longrightarrow\quad
\nu=(Q_{DD})^{-1}w_D.
\]

Therefore

\[
\boxed{\Delta w^*=-Q_{:,D}(Q_{DD})^{-1}w_D}
\]

and

\[
\boxed{C_{\mathrm{GPT}}(D)
=\frac12w_D^\top(Q_{DD})^{-1}w_D}.
\]

The inverse-Hessian submatrix appears because the constraints are imposed in coordinate space after all unconstrained coordinates are optimized out. Using \(\widetilde H_{DD}\), its diagonal, or a submatrix of a Cholesky factor would solve a different problem.

For \(D=\{i,j\}\),

\[
Q_{DD}=\begin{bmatrix}Q_{ii}&Q_{ij}\\Q_{ij}&Q_{jj}\end{bmatrix}.
\]

The off-diagonal \(Q_{ij}\) explicitly couples the two deleted weights. When \(|D|=1\), the formula reduces to the familiar static OBS cost

\[
C_{\mathrm{GPT}}(\{i\})=\frac12\frac{w_i^2}{Q_{ii}}.
\]

This single-coordinate static formula must not be conflated with the frozen SparseGPT implementation's sequential conditional score `w_current^2/U_ii^2`, which is evaluated after preceding updates.

## 3. Exact scope of the optimality statement

For one output row, one deletion pair, the fixed damped quadratic model, and all other coordinates free to compensate, the formula above is the exact constrained optimum.

It is not an exact objective for simultaneously deleting pairs in every 2:4 group. Different groups are coupled by off-block entries of \(Q\), while the selector evaluates each group in isolation on the original dense row. We therefore name it **Static Local Group-OBS Cost**. Summing per-group costs is a diagnostic surrogate, not the exact global pruning loss and not a global optimality guarantee.

The final compensation remains the frozen sequential SparseGPT OBS update under the chosen fixed mask. Consequently, static selection cost, sequential compensation loss, layer reconstruction MSE, and task accuracy are separate measurements.

## 4. APBR geometry cost

The checkpoint stores FP64 log-scales \(s_i\), with coordinate scale \(a_i=\exp(s_i)\). The existing APBR score is

\[
I_i=|w_i|\exp(-s_i).
\]

The corresponding pair deletion cost is

\[
\boxed{C_{\mathrm{APBR}}(D)=\sum_{i\in D}w_i^2\exp(-2s_i)}.
\]

It is evaluated with the same original dense weights used by the group OBS selector and the matching stored geometry. No global-radius parameter, new normalization, per-block recentering, or learned coefficient is introduced. Any common positive geometry scale within the compared candidates leaves their ordering unchanged.

## 5. Candidate enumeration and deterministic selection

For each complete four-coordinate group, enumerate deletion pairs in the fixed order

\[
(0,1),(0,2),(0,3),(1,2),(1,3),(2,3).
\]

For every output row and pair, compute `C_GPT` and `C_APBR` in that row's coordinates.

### A. Original SparseGPT

Use the unmodified dynamic selector and compensation implementation. It is the native reference, not a special case of the static selector.

### B. Group-OBS Only

\[
D^*_{\mathrm{group}}=\arg\min_D C_{\mathrm{GPT}}(D).
\]

Exact ties use the fixed pair order above. This method generates a complete mask before compensation.

### C. Group-OBS + APBR near-optimal selection

For preregistered \(\tau\in\{0.005,0.01,0.02\}\), define

\[
C_{\min}=\min_D C_{\mathrm{GPT}}(D),
\qquad
\mathcal C_\tau=
\{D:C_{\mathrm{GPT}}(D)\le(1+\tau)C_{\min}+\epsilon_C\}.
\]

Then choose

\[
D^*=\arg\min_{D\in\mathcal C_\tau}C_{\mathrm{APBR}}(D).
\]

APBR ties again use the fixed pair order. The main setting is \(\tau=0.01\); the other two are sensitivity results and cannot be chosen using Test.

Because all valid quadratic costs are nonnegative, the robust tolerance is

\[
\epsilon_C = c\,\epsilon_{\mathrm{mach}}\max_D|C_D|,
\]

with fixed \(c=64\). The multiplicative term remains \(\tau C_{\min}\). Thus if \(C_{\min}=0\), only candidates tied within a tolerance relative to that Block's own candidate scale enter the set; an all-zero Block has zero tolerance. The method introduces no unrelated absolute cost window. For \(\tau=0\), it equals Group-OBS Only except for numerical ties.

### D. Shuffled APBR control

Within each output row and each four-coordinate group, permute the four stored log-scales while leaving weights and GPT costs fixed. Use three preregistered seeds: `20261031`, `20261032`, and `20261033`. The same permutations are reused for all three tau values. This preserves each block's geometry-value multiset and destroys only coordinate correspondence.

## 6. Correct extraction of Q and Q_DD

The current adapter computes a lower factor \(L\) with

\[
\widetilde H=LL^\top,
\]

then forms \(Q\) using `torch.cholesky_inverse(L)`, followed by an upper factor \(U\) satisfying

\[
Q=U^\top U.
\]

The implementation must retain the actual `Q` tensor from the same frozen damping path for selection. `Q_DD` is obtained as `Q[index][:, index]`. It is forbidden to use `U[D,D]` as a substitute. A reconstruction test must check both `Q ~= inverse(H_damped)` and `Q ~= U.T @ U` against an independent FP64 reference.

Every principal `Q_DD` should be positive definite when `Q` is SPD. Each 2x2 candidate is checked for finite entries, successful Cholesky, positive eigenvalues/determinant, and a preregistered condition-number ceiling. Failure stops the affected layer; damping is not changed. Solving the 2x2 system via Cholesky solve is preferred over an explicit inverse.

For the maximum `K=4608` layer, one FP32 full Q is approximately 81 MiB. Retaining Q together with H and U increases lifetime memory but not asymptotic matrix size. The Pilot must measure this rather than infer feasibility.

## 7. Mask-provider and compensation contract

Methods B--D produce bool keep-masks only. All then call one external-mask provider that executes the frozen Original SparseGPT compensation semantics:

- same calibration Hessian and dead-column handling;
- same relative damping `0.01`;
- same FP32 decomposition;
- same block size `128` and left-to-right column order;
- same upper inverse-Hessian factor and error update;
- same block-local and cross-block propagation;
- same final zero enforcement and OIHW writeback.

No method may inherit compensated weights from another method. Each begins from the same dense parent and the same stored H.

Before any new selector is used, replay the native SparseGPT final mask through this provider. On the same device and dtype, require identical masks, deleted zeros, and target bitwise-equal final weights. Record maximum absolute error, relative L2 error, and local compensation loss even when zero. A mismatch is a hard stop, not a tolerance-adjustment opportunity.

## 8. CPU numerical gate specification

The isolated implementation must pass all of the following before any GPU access:

1. Enumerate all six pairs exactly once in fixed order.
2. Reconstruct `Q` from the stored upper factor and match a direct FP64 inverse.
3. Match every extracted `Q_DD` to direct indexing of the reference inverse.
4. Match `C_GPT` to an independently solved constrained quadratic program.
5. Recover the single-coordinate OBS formula when `|D|=1`.
6. Match `C_APBR` to a direct scalar reference.
7. Construct near-optimal sets correctly at tau 0, 0.5%, 1%, and 2%, including `C_min=0` and near-tie cases.
8. Preserve determinism under exact GPT and APBR ties.
9. Verify within-block shuffling preserves the multiset and changes only coordinate association.
10. Produce strict bool 2:4 masks with dense tails.
11. Keep all candidate costs and outputs finite.
12. On a small linear case, match the explicit constrained optimum and independently compare Group-OBS, Hybrid, and shuffled masks.
13. Replay the native dynamic mask through the external-mask compensation path and meet the identity contract.

No classification accuracy can substitute for these tests.

## 9. Later GPU Pilot design (not executed in this phase)

After CPU gates pass, audit the then-current GPU and run only:

- small layer: `layer1.0.conv1` if still eligible, `K=576`;
- large layer: `layer4.0.conv2` if still eligible, `K=4608`.

Use the frozen 1,024 calibration images only. For each layer run Original SparseGPT, Group-OBS, three Hybrid tau values, and three shuffled controls. Save masks, necessary layer checkpoints, candidate-level costs, candidate-set sizes, reconstruction diagnostics, time, and peak VRAM. Test data remains inaccessible.

The large layer contains 589,824 output-row groups and 3,538,944 pair candidates, so vectorization and memory profiling are required. This estimate is not a performance claim.

## 10. Claim boundaries and decision rule

Positive evidence requires all of the following trends together: real geometry changes some near-optimal choices; local GPT cost increase remains bounded by construction; APBR deletion cost decreases versus Group-OBS; true geometry separates from shuffled controls; and compensated reconstruction does not materially degrade. A lower APBR cost alone is tautological because APBR selects that objective within the candidate set.

Mask change without reconstruction benefit is only evidence of different ranking information. Similar true/shuffled behavior, worse reconstruction, uncontrolled GPT cost, or unacceptable runtime is negative evidence. Neither local cost nor layer MSE can be presented as final classification improvement.

Only the two-layer Pilot may determine whether to propose a Seed 0 full-model experiment. No full model or multi-seed queue is authorized by this design document.

