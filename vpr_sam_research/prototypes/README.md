# Untrained Block-wise VPR-L2 candidate

This branch extends the research archive only with an independent CPU prototype.
It does not modify frozen Global geometry.py, step.py, or training/engine.py.
CandidateStepper refuses formal purpose and any non-CPU device.

Only complete blocks selected by the existing pruning layer selection learn
FP64 geometry. BN affine, excluded parameters and dense tails have fixed s=0,
while all trainable coordinates still participate in the global L2 perturbation.
This default is a provisional prototype decision, not an approved formal method.

gamma_block must be explicit. The smoke value .01 is an engineering fixture,
not a formal hyperparameter choice or an empirical search result.
See results/research_delivery/BLOCKWISE_VPR_MATH.md and
BLOCKWISE_IMPLEMENTATION_PLAN.md for proofs, risks and unresolved research choices.

From this project directory (POSIX):

```bash
PYTHONPATH=src:. python -m pytest -q tests prototypes/test_blockwise.py
```

Windows PowerShell: set `$env:PYTHONPATH='src;.'` first. No CIFAR data or GPU is
required. The synthetic TinyNet two-forward and resume smoke is included in the
tests. Final full-suite result: 116 passed (107 existing + 9 prototype), CPU
PyTorch 2.13.0. This is not RTX5090 acceptance; historical training used 2.8.0.

No 200-epoch experiment, Gamma_B search, extra seed, or Test evaluation is run.
