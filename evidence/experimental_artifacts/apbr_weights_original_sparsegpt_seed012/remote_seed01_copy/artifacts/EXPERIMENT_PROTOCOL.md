# Experiment Protocol

Original SparseGPT only: dynamic 2:4 mask plus native sequential OBS compensation; 1,024 official CIFAR-100 Train calibration images; FP32; damping 0.01; block size 128; OIHW output-row contiguous layout; 19 eligible Conv2d layers. APBR contributes only dense model/BN state. No geometry, external mask, Recovery, BN-only update, Test selection, or parameter tuning. Seeds 0 and 1 were authorized as a partial execution while Seed 2 is unavailable.
