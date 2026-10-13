# Effective-space research contract

Training J includes explicit master L2; STE master gradients are labelled surrogate.
EffectiveLoss injects ALL effective coordinates with functional_call into a copied
plain model. Old masks are never applied again. Default eval_fixed_buffers; optional
train_fixed_batch uses copied buffers and replayed RNG per loss evaluation. Neither
mode changes the training model/RNG. CE only. Vector order is all named parameters,
including dense bias/BN; candidate newly active coordinates remain differentiable.

HVP uses true CE autograd, finite difference supplies independent reference;
full Hessian has an explicit <=64-dimension toy limit. Research rows must include
objective/derivative/state_mode/mask_id/snapshot_id; incompatible labels cannot mix.
Checkpoint stores masters/BN/masks/version/optimizer/scheduler/RNG/sampler epoch,
cursor/order/config and split/source hashes. Trusted self-produced checkpoints only;
num_workers=0 deterministic environment required for exact resume.
