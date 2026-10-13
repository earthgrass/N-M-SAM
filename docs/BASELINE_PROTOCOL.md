# Six-method fair baseline protocol

One BaselineTrainer composes the unchanged dense kernel or sparse wrapper.
All six share seeded initialization, materialized batches/augmentation, master
L2=.001, lr=.05, beta=.9, Nesterov, constant scheduler, rho=.05 for SAM,
same model/scope and update interval. Training metrics are perturbed-point CE
for SAM; separate fixed-state evaluation is required for predictive comparisons.

Scores use W_pre and g1_pre/g2_pre from captured trace. Entire optimizer step
finishes before selecting/committing mask for NEXT batch. Never W_post*g_pre.
Magnitude/Taylor are baselines, not novel Hessian methods. Logical masking does
not imply hardware sparse speedup. Model ResNet18 CIFAR factory has 3x3 stem,
no maxpool; stem/shortcut/BN/bias excluded, 16 internal conv + FC eligible.
Short runner bounded at 16 batches. Full training requires explicit authorization.
