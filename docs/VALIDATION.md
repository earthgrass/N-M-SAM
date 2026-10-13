# Actual Bootstrap v0.1 validation — 2026-10-13

Validated algorithm/test commit: `11e58f9aa6ba07028e6e632937b9f6598765c846`.
Worktree was clean (`git_dirty=false`) for both final dense CLI runs.
This evidence commit changes documentation and dependency minimum metadata only;
no algorithm or tests changed after these checks. Minimum torch version is 2.4
for the device-type-aware autocast check; actual validation used the versions below.

Windows CPU; Python 3.12.10; torch 2.13.0+cpu; torchvision 0.28.0+cpu;
pytest 9.1.1. No CUDA/AMP/DDP check and no CIFAR training performed.

## Test result

Command: `python -m pytest -q`

```text
......................................................... [100%]
57 passed in 9.93s
```

Counts: 57 passed, 0 failed, 0 skipped. Tests verify analytic g1/g2 locations,
one global epsilon norm, exact restoration, three-step rho=0 SGD agreement,
four-step momentum/Nesterov recurrence, explicit L2 in both gradients,
upstream ndim-based exclusions, post-g2 clipping, same batch, stochastic replay,
BN first pass only (including momentum=None and tracking=False), rollback after
forward/backward/optimizer exceptions, zero/nonfinite loss behavior, seeded CPU
smoke reproducibility, strict N:M layout/legality/ties/candidates/q.

M0: 10 passed in 0.20s. M1: 35 passed in 9.38s. M1.5: 57 passed in 9.44s.
These phase checks preceded their corresponding commits; final clean-tree check
above establishes the code snapshot used by the final CLI runs.

## Dense CPU FakeData CLI

Setup used `python -m pip install -e ".[test]" --no-deps` since required libraries
were already installed. Standard setup without --no-deps is documented in README.
Each run: seed=0, 3 steps, batch=8, TinyMLP 16->8->3 with BatchNorm1d/Tanh,
lr=.05, beta=.9, Nesterov=True, L2=.001 (all trainable parameters), clip=5,
FakeData [1,4,4], ToTensor only, CPU FP32. No download or test split.

| Actual final step metric | SAM rho=.05 | SGD rho=0 |
| --- | --- | --- |
| task CE | 1.3439754247665405 | 1.294036626815796 |
| optimizer J | 1.3500630855560303 | 1.3001116514205933 |
| L2 penalty | 0.0060876826755702496 | 0.0060750325210392475 |
| training batch accuracy | .25 | .25 |
| g1 norm | 1.0061630010604858 | 1.0108637809753418 |
| g2 norm | 1.0161802768707275 | 1.0108637809753418 |
| epsilon norm | 0.05000000074505806 | 0.0 |
| gradient norm after clip | 1.0161802768707275 | 1.0108637809753418 |
| parameter restoration error | 0.0 | 0.0 |
| forwards per step | 2 | 1 |

Actual SAM run_id: `519b69c6f9d840218cb7e3dfd3c6c9ff`.
Actual SGD run_id: `bfb16a8dabda4752a0cd04f6a66976cb`.
Full local JSON under `results/dense_<run_id>.json` stores config raw/hash, seed,
SHA/dirty flag, versions, dataset/augmentation/model/loss/optimizer/device/scope
and all three steps. Results are gitignored and not uploaded.

Commands:
`python scripts/smoke_dense_cpu.py --config configs/dense_sam_official_toy.yaml`
and `python scripts/smoke_dense_cpu.py --config configs/dense_sgd_toy.yaml`.
SAM reports perturbed-point CE, consistent with official metrics, not exact
robust maximization. Toy batch accuracy is not a research accuracy result.

## Structural CPU smoke

Command: `python scripts/smoke_nm_cpu.py`. Actual `candidate_count=6`, `valid=true`;
Linear [3,8] gives group counts [[2,2],[2,2],[2,2]],
Conv [2,2,2,2] gives [[2,2],[2,2]]. Sparse training remains PLANNED.

## Isolation evidence before push

Root commit: `913cfbe1a6f46c3eee8e711e7408d0402daebcf1`, no parents.
M1 commit: `f5aa55a26269ba63d45ad8364ba55ecbd4245521`.
M1.5 commit: `11e58f9aa6ba07028e6e632937b9f6598765c846`.
Initialized empty directory `C:/Users/earthgrass73/Desktop/N-M-SAM-REBUILD`;
no checkout/fetch/merge/cherry-pick or old source migration.
Remote initial old branch `初版待完善版本` HEAD:
`ffd318031aedf1b123a8c667bf854bf0b0913275`.
Repeated ls-remote before push confirms the same old HEAD and no rebuild branch.
Initial GitHub default branch is `初版待完善版本`. Push and final remote integrity
are verified separately and reported in the final chat response.

