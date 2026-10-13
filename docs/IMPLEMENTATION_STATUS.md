# Implementation status

Implementation-v1: M2.0 G0/G1 IMPLEMENTED_AND_TESTED (see EXPERIMENT_LOG).
M2.1 G2 IMPLEMENTED_AND_TESTED: 73 tests; four core methods two batches
on FakeData and real CIFAR. Subsequent M2.2–M7 gates pending.
Bootstrap status below is historical.

M0: configuration, src-layout, attribution, interface and research contracts IMPLEMENTED.
M0 validation PASSED: `python -m pytest -q`: 10 passed in 0.20s (2026-10-13).
M1: core loss/batch/report, dense SAM, SGD, TinyMLP, FakeData, runner IMPLEMENTED.
M1 validation PASSED: 35 passed in 9.38s; actual CPU SAM smoke 3 steps,
task CE 1.3439754247665405, optimizer J 1.3500630855560303,
epsilon norm 0.05000000074505806, restoration error 0.0.
Environment: Python 3.12.10, torch 2.13.0+cpu, torchvision 0.28.0+cpu,
pytest 9.1.1. No GPU verification performed.
M1.5: MaskSpec/MaskLayout, magnitude builder, legality, six candidates and
structural q IMPLEMENTED. Validation PASSED: 57 passed in 9.44s;
CPU mask smoke reports 6 candidates and exactly 2 retained entries per group
for Linear [3,8] and Conv [2,2,2,2]. No sparse training performed.
M2: STE, pure effective loss, support policy, Sparse-SAM PLANNED.
Diagnostics A/B/C, HVP, functional probes, ranking, compensation PLANNED.
ResNet/CIFAR, schedulers, full checkpoint/resume, CUDA/DDP/AMP PLANNED.

Dense-SGD CPU smoke PASSED: 3 steps, task CE 1.294036626815796,
J 1.3001116514205933, epsilon 0.0, one forward per step, restoration error 0.0.
M1/M1.5 validation above ran on work-in-progress before their phase commits;
runner records git_dirty and source SHA explicitly. Final clean-tree validation
is recorded in VALIDATION.md when completed.

