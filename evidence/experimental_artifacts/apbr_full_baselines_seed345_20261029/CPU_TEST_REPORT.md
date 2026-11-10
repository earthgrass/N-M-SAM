# CPU targeted regression report

Date: 2026-10-29

## Command

```text
python -m pytest -q tests/test_cifar100_50k10k_protocol.py tests/test_cifar100_baseline_adapters.py tests/test_apbr_parent_magnitude_ablation.py --basetemp=artifacts/pytest-temp-seed345-preflight
```

## Result

- Status: PASS
- Tests: 24 passed
- Warnings: 1
- Runtime: 9.65 seconds

The warning is the expected PyTorch notice that pinned host memory is unused when no accelerator is present. This run verifies CPU protocol and adapter behavior only; it does not satisfy the required CUDA smoke or formal GPU gate.
