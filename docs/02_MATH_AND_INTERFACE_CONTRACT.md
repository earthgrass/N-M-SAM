# Mathematics and interfaces

Optimizer loss J(W)=CE(W)+(lambda/2) sum p^2; task loss L=CE only.
g1=grad J(W), epsilon=rho*g1/||g1||global, g2=grad J(W+epsilon).
Restore W, clip g2 globally, apply momentum/Nesterov. No derivative through epsilon.
No extra optimizer weight decay. Fixed augmented StepBatch and RNG replay.

Effective parameters Z=P_M W, with identity P on dense/bias/BN coordinates.
q=(P_M'-P_M)W is a permanent structural displacement, distinct from epsilon.
For a fixed differentiable task loss and state,
g2-g1=[integral_0^1 H(Z+t epsilon) dt] epsilon.
Training gradients with L2, STE or changing buffers are not true effective-space HVPs.
Delta L_actual=L(Z+q)-L(Z), Delta L_1=g^T q,
Delta L_2=g^T q+0.5 q^T Hq.
u=epsilon/||epsilon||, a=u^T q, r=q-au;
q^T Hq=a^2 u^T Hu+2a r^T Hu+r^T Hr. r is known; residual curvature is missing.
For zero epsilon the decomposition must return an explicit undefined-direction
status, never divide by zero. Finite rho measures path average, not exact H(Z).

## Implemented / planned interface boundary

| Module | Input -> output; responsibility | Dependencies / status at M0 |
| --- | --- | --- |
| config/schema, loader | YAML -> DenseConfig + original text; validation | yaml, dataclasses; IMPLEMENTED |
| core/batch | materialized x,y -> immutable StepBatch container (tensors must not mutate) | torch; M1 |
| core/loss | model, batch -> LossReport(task_loss, penalty, optimizer_loss, logits) | core/batch, torch; M1 |
| core/step_types | scalar diagnostics + optional detached trace -> SAMStepReport | torch; M1 |
| sam/perturbation | gradient tuple, rho, PerturbationPolicy -> epsilon tuple + norm | torch only; M1 |
| sam/train_step | model, SGD, StepBatch, loss callable -> report and one committed update | core, sam, RNG utilities; M1 |
| sam/reference_sgd | parameters, lr,beta,nesterov -> checked torch SGD | torch; M1 |
| models/tiny_mlp | [B,1,4,4] -> [B,3] | torch; M1 |
| data/fake_data | size,seed -> torchvision FakeData/DataLoader | torchvision; M1 |
| experiment/runner, metrics | config -> CPU toy metrics/provenance | config, models, data, sam; M1 |
| sparsity/layout, mask | Linear/Conv tensor -> complete groups / reversible legal masks | torch; M1.5 |
| sparsity/ste, policy | master W,M -> STE effective W / allowed gradient policy | PLANNED M2 |
| resnet_cifar, cifar10 | explicit model/data factories | PLANNED; no downloads |
| experiment/checkpoint | full training state -> restartable checkpoint | PLANNED |
| diagnostics | fixed ProbeContext + effective Z + candidate -> labelled research result | PLANNED; no sam -> diagnostics dependency |

SAM policy Protocol only accepts a gradient tuple and returns allowed directions
of identical structure; concrete sparse policy will live in sparsity, imported by
the runner, never by the SAM kernel. Future ForwardState owns buffers/modes/RNG;
MaskSpec, MaskLayout, MaskCandidate and MaskTransition include shape, N/M,
support and q with snapshot identity. These are design contracts, not implemented
research algorithms.

