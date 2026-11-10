# RTX 4090 OBC technical smoke freeze (2026-10-25)

This stage supersedes only the active technical-smoke device binding. Historical
5090 receipts, formal benchmark device bindings, input checkpoints, frozen IDs,
upstream algorithms and SNOWS patches remain unchanged.

- Base commit: `f642c8a860db9e786843c14072c7be9c389c826c`.
- Previous smoke UUID: `GPU-36255fcb-5df2-3c02-2935-79778f083920`.
- New exact name: `NVIDIA GeForce RTX 4090`.
- New UUID: `GPU-6d21beb3-4922-a03d-55b9-b081fff80cee`.
- New config SHA256: `f1bd7395e4256ba3971f9f60916344fb5ebf7defb7664db59893243ea5d90e91`.

The runner formerly required the substring 5090. It now requires an exact
configured GPU name together with the exact UUID. This is a device-portability
change, not removal of the hardware guard. Nine rebinding cases and two affected
freeze/readiness tests passed (11 passed, zero failures/skips). Evidence:
`artifacts/server4090_transfer_20261025_1792933192612411200/device_tests.xml`.

Approved scope remains 32 frozen train images, seed 0, batch 8, one calibration
round without augmentation, parent Normalize, only layer1.0.conv1, independent
BN-only copy and two fixed-mask SGD steps. Reserve 8 GiB free VRAM / 4 GiB disk.
The config's historical review status text is retained byte-for-byte; the user's
latest task explicitly authorizes this single CUDA smoke after all gates pass.
It does not approve a formal calibration budget, full-model pruning, additional
seed, SNOWS run or 20-epoch recovery.

Initial server samples (three, separated by three seconds) showed 0% GPU
utilization, 24082 MiB free and no compute PID. Installed Python 3.12.3,
torch 2.8.0+cu128, torchvision 0.23.0+cu128; driver 595.71.05. CUDA execution is
not yet accepted by these offline tests or idle samples. All server input/source
gates and real smoke results are recorded separately in the deployment receipt.

Independent root: `/root/autodl-tmp/apbr_obc_smoke_4090_20261025`.
Project experiment path below that root:
`project/vpr_sam_research/experiments/post_training_nm_benchmark`.
Input root: `inputs`. CIFAR root: `data/cifar10`.

Obtain the new frozen commit with `git rev-parse HEAD` after this document is
committed. The deployment receipt records that literal commit and the command
actually executed. Use that commit and the config SHA above, never the old
5090 expected hash. Before launch require a clean tree, matching deployed hashes,
all frozen CIFAR files/IDs and upstream OBC commit, sufficient idle CUDA resources
and a unique nonexistent output path. Launch once; preserve failures, do not retry.

Example from the deployed experiment directory (substitute the post-commit hash
and a unique run ID, not placeholders):

```bash
nohup /root/miniconda3/bin/python -B -u -m benchmark.obc_smoke \
  --config configs/obc_cuda_smoke.yaml \
  --expected-commit "$FROZEN_COMMIT" \
  --expected-config-sha256 f1bd7395e4256ba3971f9f60916344fb5ebf7defb7664db59893243ea5d90e91 \
  --input-root /root/autodl-tmp/apbr_obc_smoke_4090_20261025/inputs \
  --data-root /root/autodl-tmp/apbr_obc_smoke_4090_20261025/data/cifar10 \
  --output "artifacts/cuda_smoke/obc_seed0_${RUN_ID}" \
  > "artifacts/logs/obc_smoke_${RUN_ID}.log" 2>&1 < /dev/null &
```

Successful launch alone is not acceptance. Require the actual CUDA PASS report,
legal mask, finite compensation, BN isolation, two masked SGD steps, exact reload,
unchanged parents, output hashes and measured resources before READY_FOR_REVIEW.
