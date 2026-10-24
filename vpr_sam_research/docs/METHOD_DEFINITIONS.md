# Frozen methods and candidate scope

SAM: epsilon=rho*g1/||g1||₂. Global VPR:
epsilon_D=rho*A²*g1/||A*g1||₂. VPR-L2:
epsilon_L=rho*A²*g1/||A²*g1||₂. Both VPR variants use
p_i=epsilon_i²/sum epsilon² and s_i+=s_i+gamma*(1-d*p_i),
A=diag(exp(s)); zero initialization and full-parameter volume conservation.
Actual frozen fixed_l2 code reuses its stable log-domain VPR path and rescales
the computed FP32 buffer, preserving the approved rounding and safety gates.
First gradient defines perturbation; second gradient updates base SGD.

Magnitude retains largest |w| in each legal N:M block. Radius-only retains
smallest s. Geometry-guided retains largest |w|exp(-s). Stable sorting keeps
lower coordinate on ties. BN affine never enters pruning Masks.
DA=sum_deleted(w*exp(-s))² is a diagonal geometric proxy, not measured loss.

Block-wise is a separate untrained candidate, described in
../results/research_delivery/BLOCKWISE_VPR_MATH.md. It is not the frozen Global
method and must never be used to relabel or rewrite historical experiments.
