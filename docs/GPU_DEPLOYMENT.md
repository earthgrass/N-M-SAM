# GPU deployment — prepare now, execute only after explicit authorization

Status: CODE_COMPLETE / UNIT_TESTED for CPU tensor contracts. DATASET_PENDING;
GPU_VALIDATION_PENDING. No server connected, no GPU queried or launched in this stage.
This project supports a **single-device full-precision** execution path. AMP and
DDP flags deliberately fail closed; neither backend is implemented or claimed validated.

## Installation and data placement

Clone `research/implementation-v1` into a new server checkout. Create an isolated
Python >=3.10 environment, install a mutually compatible torch/torchvision CUDA
build appropriate to the server's driver, then `python -m pip install -e '.[test]'`.
Exact CPU unit environment is recorded in EXPERIMENT_LOG; GPU package/driver
compatibility must be established on that server, not inferred from CPU tests.

Place complete official CIFAR-10 **Python** archive outside the Git checkout.
Set environment variables (replace these example paths):

```bash
export NMSAM_ARCHIVE=/srv/datasets/cifar-10-python.tar.gz
export NMSAM_CACHE=/srv/datasets/nmsam-cache
export CUDA_VISIBLE_DEVICES=0  # replace with the specifically approved GPU allocation
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

Future integrity checks: `sha256sum "$NMSAM_ARCHIVE"`; loader then checks the
official per-file MD5 before unpickling, format/shape/counts, path safety, and
fixed split fingerprints. It never downloads. Do not reuse truncated archives.
Root /data,/datasets,/results,/checkpoints and archives/weights are ignored;
src/nmsam/data is tracked. Server dataset status remains DATASET_PENDING until
the explicitly authorized execution verifies it. Do not commit paths/credentials
or data contents. Checkpoints are trusted local torch files, not untrusted inputs.

## Commands available now (dry-run only)

```bash
python scripts/run_experiment.py --config configs/gpu_smoke.yaml
python scripts/run_experiment.py --config configs/gpu_baselines.yaml
python scripts/run_experiment.py --config configs/research_direction_gap.yaml
python scripts/run_experiment.py --config configs/research_objective_gap.yaml
python scripts/run_experiment.py --config configs/research_compensation.yaml
python scripts/summarize_results.py
```

Dry-run parses configuration, displays matrix/budget and compile-time capabilities.
It does NOT read archives/checkpoints, instantiate ResNet, query CUDA availability,
connect a server or launch jobs. Linux `bash scripts/launch_gpu.sh --config ...`
has the same default guard. Legacy smoke CLIs also require --execute.

## Only after the user approves GPU resources and experiments

1. Verify server packages/device and run authorized bounded GPU smoke:
   `python scripts/run_experiment.py --config configs/gpu_smoke.yaml --execute`.
2. Review memory/time, correct pending numerical/device issues, approve formal matrix:
   `python scripts/run_experiment.py --config configs/gpu_baselines.yaml --execute`.
   Use `--method '2:4-SAM-Magnitude' --seed 0` to run one declared matrix element.
3. Exact resume, same config/source/data fingerprints:
   `python scripts/run_experiment.py --config configs/gpu_baselines.yaml --method '2:4-SAM-Magnitude' --seed 0 --resume /srv/run/latest.pt --execute`.
4. Point at a NEW-project sparse checkpoint (never old-project weights):
   `export NMSAM_CHECKPOINT=/srv/run/latest.pt`.
   Run A/B/C commands above with --execute ONLY after their separate budget approval.
5. Readout: `python scripts/summarize_results.py --input-dir results/formal --execute`.
   Repeat for route_A/B/C output directories. No hypotheses are automatically
   marked experimentally supported by the report generator.

Sampler cursor/order/epoch, masks/version, optimizer momentum, scheduler and
Python/NumPy/torch CPU/CUDA RNG are saved. num_workers=0 and same deterministic
software/hardware are required; multi-step GPU resume is GPU_VALIDATION_PENDING.
The unit test checks a single tiny-model step and immediate full-state restoration.

## CUDA / AMP / DDP compatibility boundary

CUDA FP32/FP64: kernel uses one model/batch device and restores state transactionally;
masks loaded from CPU checkpoint are transferred to master device. Runtime pending.
AMP design: each SAM backward must unscale before norm/projection, g2 must be
unscaled before clipping, FP32 master/epsilon norm and atomic overflow rollback
must include optimizer/BN/RNG/Mask. Current scaler/autocast activation is rejected.
DDP design: final g2 all-reduce before clipping; optional g1 synchronization must be
explicit; identical masks/versions and global norm domain; rank-local RNG/sampler
checkpoints and collective failure handling. No torchrun compatibility is claimed.
Do not use AMP/DDP flags to silently change accepted Dense-SAM semantics.

## Resource plan (estimates, no benchmark or time claim)

ResNet18 ~11 million parameters: one FP32 full parameter vector ~44 MB. This
correctness-first implementation snapshots masters/grads/optimizer and optional
SAM trace vectors, so several hundred MB of extra state precede activation memory.
Budget 8–16 GB GPU RAM as an initial **planning envelope**, not a measured minimum;
start with batch 4 and measure peak before batch 128. FP64 probes need more memory.
CPU CIFAR arrays/cache require ~0.5–1 GB RAM plus model/state; reserve external disk
for ~0.17 GB archive, ~0.31 GB extracted data and multiple full-state checkpoints.

Formal default: train-core 45000, batch128/drop_last, 351 updates/epoch, 200 epochs,
6 methods x3 seeds=18 runs, 1,263,600 optimizer updates. SGD methods one backward,
SAM methods two; approximately 2,106,000 base backwards, excluding evaluation/probes.
Per-server wall time must be estimated from the approved GPU smoke; no ETA is inferred.
All seeds share split_seed=0; methods within a seed share model/data/augmentation,
cosine schedule, master L2, sparsity scope and mask interval.

Diagnostic budgets are intentionally separate from production selection cost:
A one block six candidates: up to 13 HVPs +2 gradients (~28 reverse passes).
B defaults 5 steps x2 restarts: about 359 reverse passes for six candidates,
including repeated quadratic-operator probes. Start at 1 step/1 restart on server.
C defaults two blocks but only first 6 of 36 combinations, HVP cap4; matched
recovery and common measurements add work. Record actual calls/time before expanding.
No all-network candidate enumeration or full network Hessian is permitted by default.
