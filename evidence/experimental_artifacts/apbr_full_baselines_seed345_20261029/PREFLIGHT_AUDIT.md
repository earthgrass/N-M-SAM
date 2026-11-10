# APBR-SAM CIFAR-100 Seed 3/4/5 preflight audit

Date: 2026-10-29 (Asia/Shanghai)

## Status

**GPU GATE PASS. FORMAL SERIAL QUEUE RUNNING.**

The replacement server passed the environment, official-data, source, targeted-test, and single-batch CUDA gates. The fail-closed Seed 3/4/5 controller was then launched. Its first live stage is Seed 3 Dense-SAM; Epoch 1 completed with 50,000 samples, 391 steps, finite values, and `test_accessed=false`.

## Verified remote environment

- GPU: NVIDIA GeForce RTX 4080 SUPER vGPU, UUID `GPU-05cb4980-8f60-df90-148d-e162f0789327`.
- Driver: 595.91.07; total VRAM 32,760 MiB; approximately 32,230 MiB free before launch.
- Python 3.12.3; PyTorch 2.8.0+cu128; PyTorch CUDA build 12.8; compute capability 8.9.
- `torch.cuda.is_available()` was true; the CUDA matrix test was finite.
- Official archive size 169,001,437 bytes, MD5 `eb9058c3a382ffc7106e4002c42a8d85`, SHA256 `85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7`.
- Dataset identity audit returned `DATA_IDENTITY_PASS` for 50,000 Train / 10,000 Test and the frozen label distributions.
- Single-batch APBR CUDA smoke returned PASS with no Test access, finite output, perturbation norm 0.05, peak allocated 1,390,219,776 bytes and peak reserved 1,776,287,744 bytes.

## Local repository protection

- Git root: `C:/Users/earthgrass73/Desktop/NM稀疏/APBR-post-training-benchmark`
- Branch: `research/post-training-nm-benchmark`
- HEAD: `55fe3e300dbbd09c1178a6316afebc9ecf367947`
- Worktree: contains pre-existing untracked research files; none were reset, cleaned, moved, overwritten, or deleted.
- Protected files `benchmark/model.py`, `benchmark/nm.py`, and `benchmark/obc.py` were read-only and unchanged.
- The requested output directory did not exist before this audit and was created independently.

## Frozen protocol confirmed locally

- Protocol: `cifar100_50k10k_final_v1`
- Official CIFAR-100 archive MD5: `eb9058c3a382ffc7106e4002c42a8d85`
- Official CIFAR-100 archive SHA256: `85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7`
- Split: 50,000 Train / 10,000 Final Test / no validation split.
- Model: `ssam2022_cifar_resnet18_c100_v1`, architecture SHA256 `95367b739c42cca14cb51d1a776e5dbdbe71f9ed4a1afbd2c5c0b1bb994827ea`.
- Dense training: 200 epochs, batch size 128, FP32, SGD, cosine schedule, fixed Epoch-200 parent.
- SAM: rho 0.05, non-adaptive.
- APBR geometry: gamma `5.482145661475941e-8`, FP64 state, no warmup, unchanged feedback rule.
- Recovery: 20 epochs, fixed mask, fresh SGD, fixed Epoch-20 endpoint.
- N:M: 2:4, OIHW output-row contiguous, 19 internal Conv2d layers, dense tail, bool keep-mask.
- Calibration: 1,024 official training samples, seed 20261027, no augmentation.
- Final Test is excluded from training, calibration, mask selection, checkpoint selection, and hyperparameter decisions.

## Existing evidence confirmed locally

- Formal Seed 0/1/2 analysis reports all three seeds complete.
- Formal main analysis SHA256: `9bc91b35dcda06d8e4996e24b6e4a1481988ae70f37efb2201fcc20ec7b36e21`.
- APBR-parent Geometry/Magnitude ablation reports Seeds 0/1/2 complete.
- Ablation evidence SHA256: `547bdff60721f33214924dd4e393339d5238007f487b8429ca009bef9012dbb7`.
- Existing audits record, for every formal Seed 0/1/2 method: 19 layers, 11,157,504 eligible positions, 5,578,752 deletions, strict 2-of-4, dense tail, fixed recovery mask, and zero deleted weights.
- The existing same-parent comparison uses the same APBR parent and matched Recovery-20 order per seed.

## Versioned implementation used for launch

The old controllers remain unchanged. New isolated Seed 3/4/5 entry points were reviewed and tested to:

1. train Dense-SAM and APBR-SAM from scratch for Seeds 3/4/5;
2. run Magnitude, Wanda, SparseGPT, and ROSE from each Dense-SAM parent;
3. run Geometry and Magnitude from the identical APBR-SAM parent;
4. preserve each native pruning endpoint and run fixed-mask Recovery-20 for all six sparse branches;
5. freeze hashes before Final Test;
6. emit the requested scale and mask-overlap diagnostics without changing training or selection;
7. fail closed without automatic retries.

The reference YAML is preserved byte-for-byte under `config_snapshot/`. The extension configuration changes only hardware binding, seed list, output root, status, and explicit execution metadata; all data, model, optimization, pruning, calibration, and recovery fields compare equal to the frozen reference.

Final deployed SHA256 values:

- extension config: `7f754c7fae82aa6d280d59641261ca90f93ffdba1b99d23bffdaa9153d364a74`
- Seed 3/4/5 trainer: `c28a71f723615859a1cb2413f3c42f23d0ada0b14cf369a9e477912bfbdc57b6`
- queue controller: `15e63eda194dae03a4d517559f3359822d6683a252ec560f13406555da7038c3`
- APBR-parent Magnitude wrapper: `290008ffb7573f220aca4abecbbd0c81b77ed88c372b2ad571f9efe1fc060bd6`
- frozen postprocessor: `b530ddf6c44e4e7468e3c5cd446b849651114b5c00707025d8d064c57a1592db`
- frozen evaluator: `b4fb238a6a28cf4c6c943e8f125de88430bbcead6e08c40425f2d5a896ec6fd7`

## Local CPU regression

The local preflight suite passed. After deploying the final version, the broader remote targeted suite also passed:

`python -m pytest -q tests/test_cifar100_50k10k_protocol.py tests/test_cifar100_baseline_adapters.py tests/test_apbr_parent_magnitude_ablation.py --basetemp=artifacts/pytest-temp-seed345-preflight`

Remote result: **28 passed, 1 skipped in 1.93 s**. The new extension-only unit suite separately passed **5/5** locally and remotely.

## Resource estimate

The matrix requires six new 200-epoch dense trainings and eighteen Recovery-20 runs. The first Dense-SAM epoch took about 53.9 seconds on this vGPU. A current serial estimate is approximately 20–24 hours for the full queue, subject to measured APBR and recovery throughput.

## Launch state

- Controller PID: 2545; first training child PID: 2550 at verification time.
- Queue phase: `seed3_dense_sam`.
- Automatic retry: false.
- Test accessed: false.
- Existing Seeds 0/1/2 and protected benchmark sources were not modified.
