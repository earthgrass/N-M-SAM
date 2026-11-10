# Original APBR call-chain audit

The audit follows `vpr_sam.optim.step.Stepper.step` and `vpr_sam.optim.geometry.Geometry` directly.

1. `Geometry` owns every unique trainable FP32 parameter returned by `model.named_parameters()`, not only the 19 pruning-eligible convolutions. Its mapping is frozen in checkpoint state. BN affine parameters and the classifier therefore participate in APBR perturbation and geometry, while pruning remains restricted separately.
2. The center forward runs in training mode. `center.backward()` populates each used `parameter.grad` with the first-pass gradient.
3. `Geometry.perturbations` reads `parameter.grad` directly. A `None` gradient is treated as a zero coordinate. For VPR it applies the unchanged global log-domain normalization over all managed coordinates and enforces the global ellipsoid radius.
4. `Geometry.prepare` runs immediately after perturbation generation and before weights are perturbed. It computes the original energy-feedback candidate from the realized perturbation energy shares. It does not commit the geometry yet.
5. Base weights are cloned, the perturbation is added, and first-pass gradients are cleared with `base.zero_grad(set_to_none=True)`.
6. The adversarial forward runs at perturbed weights with BN running-stat tracking disabled. Its CE backward creates the final optimizer gradients from a clean destination.
7. The exact base weights are restored before `base.step`. The optimizer therefore applies adversarial gradients to unperturbed parameters. Geometry is committed only after the optimizer step succeeds.
8. The outer exception path restores saved weights and BN buffers, discards the pending geometry transaction, and poisons the stepper. The finalizer clears gradients.

The repaired JointStepper preserves this order. `GradientHandoff` temporarily installs the detached `autograd.grad` results into `parameter.grad`, preserving `None`, solely while the unchanged original `Geometry.perturbations` and `Geometry.prepare` execute. It then clears the handoff before adversarial backward. The pruning weight gradient is added once to the adversarial gradient, and the pruning scale update is combined once with the already-prepared original energy candidate before commit.
