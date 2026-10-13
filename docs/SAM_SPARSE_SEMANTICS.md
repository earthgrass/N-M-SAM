# Sparse training semantics

Dense train_step.py is reused unchanged. Registry functional forward replaces
explicit target weights by W*M (pure gradient) or MaskSTE(W,M) (identity surrogate).
Optimizer owns original master parameters. Master-space L2 is default, including
inactive coordinates. Effective-space L2 is an explicitly named optional ablation.
Outer g2 STE updates inactive masters; this is surrogate training, not the exact
fixed-mask true min-max gradient. Bias/BN, invalid tails, stem and shortcuts are dense.

Registry acts as allowed-support perturbation policy: project gradients, then
normalize ALL allowed coordinates once globally. Mask scope/version is locked for
both passes and optimizer update. Illegal direct mask mutation and API updates
throw and roll back mask/master/buffers/optimizer/grads/RNG. Core handles BN and
RNG exactly as Dense-SAM. rho=0 is sparse SGD. Support order matches named master
parameter order; masks are bool, on the same device as their masters.
