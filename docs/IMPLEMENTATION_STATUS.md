# Implementation status

M0: configuration, src-layout, attribution, interface and research contracts IMPLEMENTED.
M0 validation PASSED: `python -m pytest -q`: 10 passed in 0.20s (2026-10-13).
M1: core loss/batch/report, dense SAM, SGD, TinyMLP, FakeData, runner IMPLEMENTED.
M1 validation PASSED: 35 passed in 9.38s; actual CPU SAM smoke 3 steps,
task CE 1.3439754247665405, optimizer J 1.3500630855560303,
epsilon norm 0.05000000074505806, restoration error 0.0.
Environment: Python 3.12.10, torch 2.13.0+cpu, torchvision 0.28.0+cpu,
pytest 9.1.1. No GPU verification performed.
M1.5: mask layouts/builders/legality/candidates PLANNED.
M2: STE, pure effective loss, support policy, Sparse-SAM PLANNED.
Diagnostics A/B/C, HVP, functional probes, ranking, compensation PLANNED.
ResNet/CIFAR, schedulers, full checkpoint/resume, CUDA/DDP/AMP PLANNED.

