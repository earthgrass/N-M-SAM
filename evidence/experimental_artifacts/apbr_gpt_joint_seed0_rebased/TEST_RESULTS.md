# Phase-3 test results

- Joint implementation, calibration, orchestration, original baseline adapters, legacy Group-OBS, and pipeline: **55 passed** in 10.01 s.
- Original APBR/VPR mathematics and step transaction regressions: **34 passed** in 7.23 s.
- Combined recorded result: **89 passed, 0 failed**.
- Real CPU orchestration smoke: `CPU_ORCHESTRATION_PASS`; evidence is in `cpu_orchestration_smoke_attempt3/cpu_orchestration_evidence.json`.

The smoke used official CIFAR-100 training records only. CUDA and Final Test were not accessed.
