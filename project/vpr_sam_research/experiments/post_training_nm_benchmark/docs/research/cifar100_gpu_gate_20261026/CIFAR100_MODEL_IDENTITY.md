# CIFAR-100 ResNet18 Model Identity

## Current status

`CIFAR100_MODEL_IDENTITY_PASS`

The 100-class structure is frozen, its structural tests pass, and the required `DATA_IDENTITY_PASS` prerequisite is now linked by audit-file SHA256 and archive SHA256 in the machine-readable manifest.

## Isolated construction entry

- Factory: `baseline_adapters.cifar100_model.build_cifar100_model`
- Model ID: `ssam2022_cifar_resnet18_c100_v1`
- Legacy CIFAR-10 factory left unchanged: `benchmark.model.build_model`
- Parameters: `11,220,132`, all trainable
- State-dict entries: `122`
- Architecture SHA256 over ordered key/shape manifest: `95367b739c42cca14cb51d1a776e5dbdbe71f9ed4a1afbd2c5c0b1bb994827ea`
- Seed-20261026 initialized state SHA256: `3b05fd6f017e5d33dcde5c517c34d14457cf7311a5339683c5d281f0a321f874`

No CIFAR-10 checkpoint is loaded. The model used for identity and future GPU resource testing is randomly initialized and is not a trained CIFAR-100 model.

## Frozen architecture

- Stem: `Conv2d(3,64,3,stride=1,padding=1,bias=False)` + BN + ReLU; no max-pool.
- Stages: BasicBlock counts `[2,2,2,2]`, channels `[64,128,256,512]`, first-block strides `[1,2,2,2]`.
- BasicBlock: two 3×3 bias-free convolutions with BN; the projection shortcut is 1×1 bias-free Conv2d + BN when stride or channels change.
- Pool: `F.avg_pool2d(kernel_size=4)`.
- Head: `Linear(512,100,bias=True)`.
- Initialization: unchanged `torch.nn` module `reset_parameters` defaults; no project custom initializer.

Compared with the existing CIFAR-10 model, only these tensor shapes differ:

| Key | CIFAR-10 | CIFAR-100 |
|---|---:|---:|
| `linear.weight` | `[10,512]` | `[100,512]` |
| `linear.bias` | `[10]` | `[100]` |

All other state-dict keys, key order, and shapes remain identical.

## Forward, gradient and pruning contract

- Forward output passed at `[2,100]`; loss and all outputs were finite.
- Every trainable parameter received a finite gradient.
- Pruning scope contains 19 internal Conv2d modules.
- `conv1`, `linear`, every BN affine parameter, and every bias are excluded.
- Layout is OIHW output-row contiguous.
- Each complete four-column group keeps exactly two positions.
- An incomplete tail stays dense.
- Masks are boolean keep-masks (`True` means retained).
- Magnitude ties use stable descending order, so lower contiguous indices are retained first.
- SparseGPT pruning-metric ties use stable ascending order, so lower contiguous indices are removed first.

The full machine-readable structure, state shapes, keys, hashes, checks, and eligible layer names are in `GPU_GATE_TESTS/model_identity.json`.

## Regression evidence

- Focused identity/gate tests: 5 passed.
- Existing adapter tests after FP32 instrumentation: 14 passed.
- Scoped project tests after isolating the new factory outside the protected `benchmark/*.py` historical audit: `110 passed`.
- JUnit report: `GPU_GATE_TESTS/pytest_tests_rerun.xml`.

