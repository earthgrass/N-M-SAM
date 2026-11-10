# HQ-APBR V1 CPU numerical audit

Date: 2026-11-05. Scope: standalone reference math, synthetic matrices/data, and read-only inspection of the frozen CIFAR-100 model implementation. Python 3.12.10, PyTorch 2.13.0+cpu. **No CUDA, CIFAR-100 Test, HQ dense training, or HQ checkpoint was used.**

## Final gate

`python -m pytest -q tests --basetemp=artifacts/pytest-temp --junitxml=reports/hq_apbr_cpu_pytest.xml` finished with **12 passed**. The full [JUnit report](hq_apbr_cpu_pytest.xml) names every test. `python -u scripts/run_cpu_numerical_gate.py` produced [numeric metrics](hq_apbr_v1_cpu_numerical_metrics_v2.json) with `CPU_NUMERICAL_GATE_PASS`. The JSON records source SHA256 and the predeclared checks used by the runner.

| Gate | Result and measured error |
|---|---|
| A: inverse diagonal against explicit inverse | Max absolute error `1.11e-15` |
| A: singleton OBS cost against constrained SPD solve | Absolute error `1.04e-17`; damped and undamped costs differ by `1.25e-3` |
| A: Gram symmetry policy | Runner correction `0`; separate test accepts and explicitly symmetrizes a `5e-8` asymmetric entry, checks inverse of returned effective matrix, and rejects material asymmetry |
| B/D: log target and globally weighted center | Sum error `2.22e-16`; reconstructed Q max error `1.78e-15` |
| C: three-state EMA | Explicit history max error `2.78e-17`; final zero-sum drift `6.66e-16`; malformed uncentered history rejected |
| E/F: fixed global L2 and Direct-Q convention | Radius relative error `1.78e-8`; eligible-only direction max error `2.63e-10`; noneligible scale caveat separately tested |
| G: Conv2d and 2:4 | Actual ResNet18 module list: 19 eligible Conv2d, 11,157,504 coordinates and 5,578,752 deletions; each full quartet keeps exactly two; bool Mask and dense-tail behavior checked. Unfold/Conv forward equivalence checked on representative 3×3 layers and a grouped Conv. |
| H: deterministic short resume | Synthetic CPU two-pass train: model, optimizer, cosine scheduler, FP64 geometry, epoch, train/calibration RNG order and configuration hash matched bitwise after interruption and reload. |

Boundary tests reject nonfinite cost even when finite inputs overflow their ratio; the final static 2:4 rank uses log scores so finite extreme weights/Q do not collapse into `Inf` ties. First iterations exposed a missing scratch directory and an overstrict exact-symmetry check; the final recorded run reflects both corrections. The prior [v1 numeric record](hq_apbr_v1_cpu_numerical_metrics.json) is retained as superseded audit history and its source hashes differ from the final implementation.

**Status: `CPU_NUMERICAL_GATE_PASS` for the standalone reference and listed CPU tests only.** It does not establish full HQ training integration, BN isolation on the real model, actual CIFAR-100 calibration, GPU numerical stability, runtime, memory, accuracy, or 200-Epoch resume reproducibility.
