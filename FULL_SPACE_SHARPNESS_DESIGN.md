# Full-space Sharpness Design

## Status

Design only. The current experiment path implements constrained sharpness and does not execute this full-space diagnostic.

## Center definition

For every sparse weight with dense master value `W` and current N:M mask `M`, the probe center is the actual sparse model

`W_s = M ⊙ W`.

The center must never be evaluated by exposing latent values in `(1-M) ⊙ W`. Before a full-space probe, each sparse layer therefore needs an isolated effective-center tensor copied from `M ⊙ W`.

## Full-space direction

At the sparse center, compute the gradient while treating the temporary effective-center tensor as an ordinary dense parameter. The direction is not multiplied by `M`. Dense weights, sparse-center weights, biases, and normalization parameters all enter one global norm:

`epsilon_full = rho * g_full / ||g_full||_2`.

This lets positions outside the current mask participate in the perturbation without allowing the latent dense master values to enter either the center or the direction.

## Safe implementation protocol

1. Snapshot every parameter, sparse mask, module training flag, and BatchNorm running buffer.
2. Materialize each sparse effective center `M ⊙ W` into a temporary dense probe parameter/module.
3. Evaluate `L(W_s)` in evaluation mode and compute `g_full` with respect to the temporary dense center.
4. Apply one global-radius perturbation, including positions where `M=0`.
5. Evaluate `L(W_s + epsilon_full)` and compute `S_full`.
6. Remove the temporary modules and restore all parameters, masks, modes, gradients, and buffers exactly.
7. Assert bitwise equality for masks and BatchNorm buffers and numerical equality for parameters after restoration.

The temporary dense probe must be structurally separate from the training-time STE wrapper. Simply disabling a mask on the dense master weight is invalid because it would expose latent `(1-M) ⊙ W` values and move the center away from `W_s`.

## Derived diagnostics

With the already implemented constrained probe at the same center, batch, loss, and radius:

- `Hidden Sharpness Gap = S_full - S_constrained`
- `Sharpness Capture Ratio = S_constrained / S_full`

The ratio needs an explicit zero-denominator convention and should be reported as `NA` when `|S_full|` is below a numerical tolerance.

## Required tests before implementation

- Full-space epsilon norm equals `rho` for non-zero gradients.
- An initially masked position can receive a non-zero full-space perturbation.
- Latent dense master values do not affect the center loss.
- Parameters, masks, gradients, BatchNorm buffers, and train/eval mode are restored.
- Constrained and full-space probes use the same samples and `rho`.

Implementation is intentionally deferred until this definition is reviewed.
