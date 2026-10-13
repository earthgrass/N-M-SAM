# Implemented architecture and mathematical API

`config -> data/models -> experiment.training -> sam(core)`; sparse registry/policy
is composed by runner, never imported by SAM. `diagnostics -> core.functional_model`.

| Boundary | Implemented inputs -> outputs |
| --- | --- |
| data/cifar_archive | local archive/cache -> safe official-file-verified external root + SHA |
| data/cifar10,transforms,sampler | fixed split/augmentation/order -> loaders, hashes, resumable cursor |
| sparsity/ste,registry | W,M -> masked forward; pure or identity surrogate; version/support |
| sparsity/schedule | pre-step W,g1/g2 -> magnitude/Taylor masks for next full step |
| sam/train_step | batch,J,allowed policy -> two-pass epsilon/report, exact restore and SGD |
| experiment/training,evaluation | six-method single core -> labelled training J/CE and fixed-state task eval |
| experiment/checkpoint | full model/master/BN/mask/optimizer/scheduler/RNG/sampler/config -> local state |
| core/functional_model | all effective Z + fixed batch/state -> true CE, no old mask reapplied |
| diagnostics/hvp | true loss,z,v -> Hv; finite difference and <=64-dimension Hessian references |
| gradient_path | fixed L,Z,old support,rho -> g1,g2,epsilon,h_obs,Hu or undefined_direction |
| candidate_mask | explicit scoped block(s) -> 6 legal candidates or budgeted 36 pairs, q=(P'-P)W |
| direction_gap | six q -> actual/first/second loss, observed/residual curvature, ranking/cost |
| robust_probe,quadratic_robust | new support -> one/multi-step proxies or certified small quadratic reference |
| objective_gap | L,Z,q,M' -> DeltaL,DeltaS_proxy,DeltaF_proxy, local Q robust and rankings |
| coupling,compensation | qA,qB,new support,budget -> cross term; restricted quadratic/robust/recovery controls |
| deployment,compare | validated config + explicit execute -> future GPU logs/CSV/checkpoints/readout |

Core formulas:
J=L(P_M W)+lambda/2||W||²; g_train is STE surrogate when configured.
d=P_M g1; epsilon=rho*d/||d||global. g2 updates master, including inactive STE
coordinates. Mask frozen throughout step. Dense semantics unchanged.
True diagnostics use L(Z), g=grad_Z L, H=Hessian_Z L, no L2 or STE.
g2-g1=[integral_0^1 H(Z+t epsilon)dt]epsilon; finite-path observation is not exact H(Z).
q=au+r; q^T Hq=a²u^THu+2a r^THu+r^THr.
F(Z,M)=max_supported_ball L(Z+delta); finite ascent returns robust_proxy only.
DeltaF=DeltaL+DeltaS; Q robust uses the SAME old-center quadratic and NEW support.
Cross coupling: q^THq=qA^THqA+qB^THqB+2qA^THqB.
Compensation v has P_new v=v and norm<=budget. Damping changes the local proxy;
no generic Hessian inverse or guarantee for indefinite network curvature.

No innovative Mask Selection policy is installed. Production schedule exposes
only Magnitude/Taylor-g1/Taylor-g2 baselines. Research evaluator rankings are
diagnostic outputs, not automatic training decisions.
