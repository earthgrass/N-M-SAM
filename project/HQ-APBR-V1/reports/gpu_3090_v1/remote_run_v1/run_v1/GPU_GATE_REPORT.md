# HQ-APBR V1 GPU gate

{
  "status": "GPU_SMOKE_BLOCKED",
  "gpu": "NVIDIA GeForce RTX 3090",
  "epochs": 3,
  "calibration_to_train_ratio": 0.30552562057541494,
  "estimated_200_epoch_compute_seconds": 11993.895852814117,
  "every_epoch_refresh_feasible": false,
  "resource_limits": {
    "peak_allocated_bytes": 21474836480,
    "peak_reserved_bytes": 23622320128,
    "maximum_calibration_to_train_ratio": 0.3
  },
  "test_accessed": false,
  "resume_bitwise_equal": true
}

Real data: 1024 fixed Train images, all19 Conv2d, sequential layer matrices. No Test access. Temporary3Epoch smoke is separate from formal Seed0.
