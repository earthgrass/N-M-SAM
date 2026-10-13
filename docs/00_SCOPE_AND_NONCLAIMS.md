# Scope

M0 architecture; M1 single-device FP32/FP64 Dense-SGD/SAM; M1.5 deterministic
N:M primitives after M1 passes. No inherited Git history/source. No CIFAR run,
GPU claim, DDP, AMP, upstream accuracy reproduction, hardware sparse acceleration,
Sparse-SAM training, pruning schedule or research scoring algorithm in v0.1.

Only actual passing tests receive PASSED. Planned modules exist as documentation,
not functions returning fabricated values. Historical six-run results from the
Word input are not rebuild results. Upstream zero-gradient division is undefined;
our explicitly tested finite zero-direction fallback is a documented extension.

