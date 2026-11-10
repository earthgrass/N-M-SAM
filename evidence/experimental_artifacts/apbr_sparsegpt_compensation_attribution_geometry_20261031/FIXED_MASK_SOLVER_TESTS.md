# Fixed-Mask Solver Tests

Local targeted suite: **22 passed**.

Verified properties:

- Mask-only projection preserves every retained Dense weight.
- Deleted coordinates are exactly zero.
- BN affine parameters and BN running buffers are unchanged.
- Radius-Guided SparseGPT degenerates exactly to Original SparseGPT for unit radius, constant radius, or `lambda=0` in its ranking fusion gate.
- The fixed-mask quadratic solver satisfies its linear system on an independently constructed positive-definite problem.
- Geometry regularization at zero strength is exactly identical to the standard fixed-mask solution.
- Nonconstant APBR geometry can change the retained-coordinate compensation direction beyond an isotropic ridge while leaving deleted coordinates fixed.

CUDA layer residual, timing, and memory gates are pending because the registered GPU host became unreachable before this task was deployed. No CPU result is reported as a GPU pilot.
