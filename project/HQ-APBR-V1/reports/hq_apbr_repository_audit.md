# HQ-APBR V1 — Phase 0 repository audit

Audit date: 2026-11-05 (Asia/Shanghai). Scope: read-only inspection of the current local workspace and existing records. Source paths below are relative to `C:/Users/earthgrass73/Desktop/NM稀疏/APBR-post-training-benchmark/vpr_sam_research` unless stated otherwise. This report does not claim an HQ implementation, GPU benchmark, pilot result, or Test result.

## 1. Workspace and provenance

| Item | Observed value |
|---|---|
| Git root | `C:/Users/earthgrass73/Desktop/NM稀疏/APBR-post-training-benchmark` |
| Project | `vpr_sam_research` |
| Branch / HEAD | `research/post-training-nm-benchmark` / `55fe3e300dbbd09c1178a6316afebc9ecf367947` |
| Worktree | Dirty: 195 `git status --porcelain=v1` entries at inspection, mostly pre-existing untracked research files; `experiments/post_training_nm_benchmark/scripts/run_cifar100_50k_formal_training.py` is modified. No reset/clean/branch switch was performed. |
| Local Python / Torch | Python 3.12.10; PyTorch 2.13.0+cpu; `torch.version.cuda=None`, `torch.cuda.is_available()=False`, device count 0. `nvidia-smi` is unavailable. |

The uncommitted formal-trainer diff adds an optional `--initial-asset` and associated provenance/geometry load for a separate rebased cohort. The frozen historical checkpoints retain their own source SHA256; the dirty trainer must not be treated as bitwise identical to that historical source. `AGENTS.md` protects historical assets and the original global APBR energy-feedback definition. The user has explicitly requested HQ as a separate method, so an HQ EMA must live in a new implementation and must not silently replace the original APBR code.

## 2. Frozen CIFAR-100 baseline actually present

`experiments/post_training_nm_benchmark/configs/cifar100_50k10k_final_v1.yaml` specifies protocol `cifar100_50k10k_final_v1`, Seeds `[0,1,2]`, official 50,000 Train / 10,000 Test, no validation split, Dense Final at Epoch 200, 2:4, and 19 internal Conv2d modules. Its historical handoff variant retains the same training/numerical protocol with hardware/output metadata changes. The current local GPU state above is **not** the historical remote GPU environment.

The official archive is currently present at `C:/Users/earthgrass73/Desktop/NM稀疏/data/cifar100_official_20261026/cifar-100-python.tar.gz`: 169,001,437 bytes, MD5 `eb9058c3a382ffc7106e4002c42a8d85`, SHA256 `85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7` (independently recalculated for this audit). `experiments/post_training_nm_benchmark/docs/research/cifar100_gpu_gate_20261026/CIFAR100_DATA_IDENTITY.md` records train/test array and label-distribution checks. New code must use the project's own explicit data path/configuration; `AGENTS.md` forbids runtime imports or implicit data/config/checkpoint reads from the parent historical engineering tree.

The frozen 100-class factory is `baseline_adapters/cifar100_model.py:11`. It constructs `benchmark/model.py`'s ResNet18 with BasicBlock `[2,2,2,2]`, a 3×3 CIFAR stem and 100-class `linear` head. The model has 11,220,132 trainable parameters and 122 state-dict entries. `docs/research/cifar100_gpu_gate_20261026/CIFAR100_MODEL_IDENTITY.md` records architecture SHA256 `95367b739c42cca14cb51d1a776e5dbdbe71f9ed4a1afbd2c5c0b1bb994827ea`. A fresh CPU instantiation in this audit independently produced the expected parameter and eligible-layer counts.

### Exact eligible traversal

`benchmark/nm.py:7` takes all `Conv2d`/`Linear` modules in module traversal order and removes the first and last. Thus the stem `conv1`, final `linear`, all BN affine tensors, and biases are excluded. The resulting **19 Conv2d modules include three 1×1 projection shortcuts**; they are not all 3×3. All current modules have `groups=1`.

| Order | Layer | OIHW weight shape | Per-row input width K |
|---:|---|---|---:|
| 1 | `layer1.0.conv1` | `[64,64,3,3]` | 576 |
| 2 | `layer1.0.conv2` | `[64,64,3,3]` | 576 |
| 3 | `layer1.1.conv1` | `[64,64,3,3]` | 576 |
| 4 | `layer1.1.conv2` | `[64,64,3,3]` | 576 |
| 5 | `layer2.0.conv1` | `[128,64,3,3]` | 576 |
| 6 | `layer2.0.conv2` | `[128,128,3,3]` | 1152 |
| 7 | `layer2.0.shortcut.0` | `[128,64,1,1]` | 64 |
| 8 | `layer2.1.conv1` | `[128,128,3,3]` | 1152 |
| 9 | `layer2.1.conv2` | `[128,128,3,3]` | 1152 |
| 10 | `layer3.0.conv1` | `[256,128,3,3]` | 1152 |
| 11 | `layer3.0.conv2` | `[256,256,3,3]` | 2304 |
| 12 | `layer3.0.shortcut.0` | `[256,128,1,1]` | 128 |
| 13 | `layer3.1.conv1` | `[256,256,3,3]` | 2304 |
| 14 | `layer3.1.conv2` | `[256,256,3,3]` | 2304 |
| 15 | `layer4.0.conv1` | `[512,256,3,3]` | 2304 |
| 16 | `layer4.0.conv2` | `[512,512,3,3]` | 4608 |
| 17 | `layer4.0.shortcut.0` | `[512,256,1,1]` | 256 |
| 18 | `layer4.1.conv1` | `[512,512,3,3]` | 4608 |
| 19 | `layer4.1.conv2` | `[512,512,3,3]` | 4608 |

Sum: **11,157,504 eligible weight coordinates** and, under strict 2:4, **5,578,752 deletions**. Every K above is divisible by four, so these 19 layers have no incomplete group. The shared utility `baseline_adapters/common/nm.py:12` defines bool keep masks with stable tie handling and dense tails for general shapes. `benchmark/nm.py:25` validates every complete quartet and zeroed deleted positions; `benchmark/nm.py:40` projects weight, gradient, and SGD momentum under a fixed mask. OIHW output-row contiguous grouping is also documented in `docs/research/cifar100_baseline_audit_20261026/NM_LAYOUT_SEMANTICS.md`.

## 3. Dense training and APBR state call chain

The 0–2 formal entry is `experiments/post_training_nm_benchmark/scripts/run_cifar100_50k_formal_training.py:195`; the later SAM extension entry is `scripts/run_cifar100_50k_sam_seed245_training.py`, and APBR-L2 uses its own scripts. Do not conflate APBR-L2's fixed-radius ablation with the original APBR-SAM `vpr` baseline.

Formal training checks archive identity before unpickling and verifies all 50,000 Train samples/classes (`run_cifar100_50k_formal_training.py:224–241`). Each epoch's shuffled DataLoader is deterministically seeded; training augmentation is `RandomCrop(32,padding=4)`, horizontal flip at 0.5, tensor conversion, and frozen CIFAR-100 normalization (`:242–249`). It uses batch size 128, `drop_last=False`, 391 batches/epoch, SGD learning rate 0.1, momentum 0.9, weight decay 0.0005, no Nesterov, epoch-end cosine to zero over 200 epochs, and global SAM `rho=0.05` (`:266–273`, `:318–370`). AMP is false in the frozen config.

`src/vpr_sam/optim/step.py:49–122` is the actual two-pass step. It zeros grads, runs center forward/backward to populate `p.grad`, gets the perturbation from `Geometry.perturbations`, calls `Geometry.prepare`, saves and perturbs parameters, zeros first-pass grads, runs adversarial forward/backward under BN running-stat protection, restores unperturbed weights, then applies SGD using second-pass grads. `Geometry.commit()` occurs only after the optimizer step succeeds. Failure restores saved parameters/BN buffers, discards pending geometry, poisons that Stepper, and forbids further valid checkpointing. The original SAM path has zero geometry logs; original `vpr` has persistent FP64 logs.

`src/vpr_sam/optim/geometry.py:16–42,65–158` maps **all 11,220,132 trainable parameters**, not just pruning-eligible coordinates. Original APBR computes the elliptical perturbation from `exp(2s)g` with denominator `||exp(s)g||`, prepares global energy feedback `p_i=epsilon_i²/||epsilon||²`, and commits `s_i += gamma(1-d p_i)`. Original `vpr` does not force the physical perturbation L2 norm to equal rho. `fixed_l2` rescales this perturbation and is a separate ablation. The proposed HQ method instead needs eligibility-aware EMA targets, non-eligible `s=0`, fixed **global** L2 normalization, and no energy-feedback update. It therefore cannot be claimed to be a parameter-only use of the original `Geometry` class. The original persisted geometry is under `checkpoint['stepper']['geometry']` with `mapping`, `d`, `config`, `method`, `logs`, update counts (`geometry.py:176–195`). New HQ state requires its own schema and explicit checkpoint compatibility checks.

Checkpointing is atomic and reloaded/validated after save (`run_cifar100_50k_formal_training.py:71–95`). `last.pt` is published each epoch; `final_epoch_200.pt` is fixed at 78,200 optimizer steps. Payload includes model, model SHA256, Stepper (SGD plus geometry), scheduler, history, identity, and Python/NumPy/Torch/CUDA RNG states (`:99–122`, `:184–194`, `:356–371`). Identity includes config/source/data hashes, seed, model architecture, GPU and software version; `test_loader_constructed=false` and `test_accessed=false` during training.

## 4. Calibration, Hessian, pruning, and Test isolation

`scripts/run_cifar100_50k_postprocess.py:86–95` freezes 1,024 Train calibration IDs via NumPy PCG64 permutation seed 20261027, uses no augmentation, collects Conv inputs through forward pre-hooks, and saves `raw_hard.pt` with parent/mask provenance. The fixed-mask recovery path is SGD for 20 epochs (`:109–123`); it is a separate historical endpoint from HQ's proposed no-recovery pilot. Final Test is a separate evaluation stage, not a training or mask-selection input.

`baseline_adapters/common/conv2d.py:34–61` flattens OIHW as `[Cout,Cin/G*kH*kW]` and uses `F.unfold` in channel–kernel-row–kernel-column order. Grouped convolutions produce a separate patch matrix per group. `conv2d_via_unfold` (`:80`) is an existing forward-equivalence reference. `baseline_adapters/sparsegpt/adapter.py:115–143` accumulates one Gram per Conv group over true input patches. **Actual numerical convention is `H=2XXᵀ/n`**, not the HQ prompt's `XXᵀ/n`; the adapter stores full `K×K` Gram matrices. Its `sparsegpt_matrix` (`:29–109`) adds relative damping `0.01*mean(diag(H))`, uses Cholesky → `cholesky_inverse` → upper Cholesky factor, chooses dynamic 2:4 masks while sequentially compensating, and writes back original Conv coordinates. The final HQ score is a static score from a frozen Final checkpoint and must not be labeled native SparseGPT dynamic mask selection.

Normalization discrepancy matters: replacing `2XXᵀ/n` by `XXᵀ/n` while retaining the same *numeric absolute* damping changes Q and the objective. A new estimator must document whether it uses the historic factor 2 or a converted convention, and scale the absolute damping consistently. Relative damping must remain defined against the same H actually used. The new inverse-diagonal-only estimator is not already present: existing SparseGPT constructs a full inverse/factor for OBS compensation.

## 5. Available assets and blocking gaps

- Local formal Seed 0 Dense-SAM and APBR-SAM `final_epoch_200.pt` exist under `artifacts/cifar100_50k10k_final_v1/seed0_remote_copy/`.
- Local formal Seed 1 Dense-SAM and APBR-SAM Final exist under `artifacts/cifar100_50k10k_final_v1_blackwell_handoff/seed1_remote_copy/`.
- A verified newer SAM Seed 2 Final exists under `artifacts/dense_sam_apbr_l2_seed1to4_20261103_local_copy/seed2/dense_sam/`; its protocol and config provenance differ from the original formal Seed 0/1 files even though earlier local audits found its model state reproducible. APBR-L2 Seed 2 assets are a separate method.
- This local project scan did **not** find an original APBR-SAM Seed 2 `final_epoch_200.pt`. The frozen formal config lists Seeds 0/1/2, but complete three-seed *local checkpoint availability* must be checked before planning a full paired HQ matrix. Do not substitute APBR-L2 for original APBR.
- `docs/research/cifar100_gpu_gate_20261026/SPARSEGPT_GPU_GATE_REPORT.md` records a historical remote RTX 4080 SUPER K=4608 numerical PASS but resource FAIL against its frozen in-process free-memory threshold. That does not constitute an HQ GPU cost audit. Current local environment is CPU-only. **Phase 3 actual GPU timing, VRAM, conditioning, and HQ feasibility remain BLOCKED pending an explicitly available GPU environment.**

No formal training, Test, remote login, branch switch in the historical repository, or historical asset modification was performed in Phase 0. Following the user's version-management instruction, the HQ reference implementation was placed in the separate sibling directory `C:/Users/earthgrass73/Desktop/NM稀疏/HQ-APBR-V1`, initialized as its own Git repository on `research/hq-apbr-v1`, with independent source, reports, and hashes.
