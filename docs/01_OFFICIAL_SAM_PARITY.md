# Official JAX/Flax -> PyTorch semantic mapping

Reference [flax_training.py](https://github.com/google-research/sam/blob/dae9904c4cf3a57a304f7b04cecffe371679c702/sam_jax/training_utils/flax_training.py)
at `dae9904c4cf3a57a304f7b04cecffe371679c702`.
`get_sam_gradient` is nested in `train_step`, not a top-level function.

| Official location | Port contract |
| --- | --- |
| `global_norm`, `dual_vector`, lines 388–425 | sqrt of sum of squares over all trainable leaves; one global normalization |
| `forward_and_loss`, 455–481 | CE + lambda/2 sum p^2 at each evaluation point |
| `no_weight_decay_on_bn`, 474–476 | exactly `p.ndim > 1`, excluding one-dimensional bias as well as BN affine parameters |
| `get_sam_gradient`, 492–507 | g1 at W; epsilon=rho*g1/norm; g2 at W+epsilon on the same materialized batch |
| `inner_state`, 494 and 507 | commit first-forward buffers; discard second-forward buffers, use batch statistics in both passes |
| same `prng_key`, 464 | replay torch CPU/CUDA, Python and NumPy RNG for pass two; no cross-framework random-bit parity |
| rho branch, 511–516 | rho=0 performs one backward with the same loss |
| `pmean` then clip, 518–522 | single device: synchronization is identity; exact global rescale after g2, before optimizer |
| `create_optimizer`, 201–203 | torch SGD, momentum beta, dampening=0, nesterov=True, weight_decay=0 |
| metrics, 531–534 | task CE/accuracy from perturbed logits for SAM, not regularized optimizer loss |

Momentum recurrence: b_t=beta*b_(t-1)+g_t, b_0=0;
Nesterov direction=g_t+beta*b_t; W_(t+1)=W_t-lr*direction.
PyTorch's first momentum buffer equals g1, identical to this zero-buffer recurrence.
Parameters are restored by exact snapshot copy before applying this update.

State details: Flax calls both forwards with the original collection, returns
first collection only. The port snapshots original and first-pass buffers; second
pass starts with original buffers and disables BN running-stat tracking while
keeping BN in training mode. After pass two all buffers revert to first-pass values.
PyTorch BN running-variance estimator/epsilon defaults can differ from Flax;
this is a state-update contract, not bitwise cross-framework BN parity.

Failure contract extends immutable upstream semantics to mutable torch: rollback
parameters, all registered buffers, prior grads, optimizer state, training flags,
BN tracking switches and RNG if forward/backward/optimizer throws. Supported
models must not mutate batch inputs, parameter/module registration, external state
or non-buffer attributes during forward. Loss callbacks must be scalar, differentiable,
and side-effect free apart from normal model buffers. Full rollback snapshots cost
memory/time and are appropriate for this correctness-first bootstrap.

Zero g1 produces zero epsilon, reports `zero_direction=True`, still evaluates g2
for rho>0. This is a finite safeguard, not upstream's undefined 0/0 behavior.
No Shake-Shake/Shake-Drop `true_gradient` emulation, RMSProp, learning-rate
schedule, multi-replica synchronization or EMA is implemented.

