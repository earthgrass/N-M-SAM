# N:M structural contract

Linear [out,in] and Conv2d [out,in,kh,kw] flatten separately per output row.
Consecutive groups of M=4 each have exactly N=2 binary ones. Tail policy rejects
the entire unsupported layer; a future runner may explicitly exclude it with a
report. No silent partial-group padding. Bias/BN and excluded layers remain dense.

All C(4,2)=6 supports enumerate in lexicographic kept-index order:
(0,1),(0,2),(0,3),(1,2),(1,3),(2,3). Magnitude selection sorts descending abs(W)
with stable lower flattened index winning exact ties; nonfinite inputs reject.
Selection must not mutate weights, and records N/M/layout/tie policy.

PLANNED STE: forward W_eff=W*M, surrogate dW_eff/dW=identity.
PLANNED pure_effective_loss(Z): unmasked effective tensor enters functional model;
true autograd derivatives in Z, no STE and no hidden reapplication of old M.
This matters when q/r includes newly active coordinates.

PLANNED Sparse-SAM: fix mask version during the entire step; allowed d=P_M g1
with identity on dense coordinates, normalize all allowed directions once
globally. epsilon outside sparse support is zero. This changes the perturbation
domain relative to Google Dense-SAM and must be labelled a custom extension.

