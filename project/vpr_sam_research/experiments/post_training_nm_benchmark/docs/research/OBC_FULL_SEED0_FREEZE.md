# Full OBC seed0 / RTX4090 pilot — 2026-10-25

The user approved the complete seed0 pilot after the actual CUDA smoke and its
read-only artifact audit passed. No dense model retraining or repeated CUDA smoke.
Latest user steering skips independent BN-only recalibration and evaluation.
Existing4096BN indices are retained unchanged, but no BN-only output is made.

## Locked upstream audit

OBC commit `9b7979bfc9ee20d87db553823a32ee9890beaa99` is read through upstream_lock:

| Field | Official implementation | Frozen CIFAR controlled pilot |
|---|---|---|
| Samples |main_trueobs.py --nsamples1024; random_subset uses seeded NumPy shuffle of arange|1024of50000 train images, same seed0 RandomState shuffle algorithm; saved IDs and digests|
| Batch |CLI -1 sentinel resolves to128 in datautils.py ImageNet path|128, not the earlier proposed32|
| Rounds |CLI -1 resolves to10 for image classification; --noaug overrides to1|10augmented passes over the same1024IDs,10240image visits /80batches per layer|
| Views |RandomResizedCrop224 scale(.2,1),default ratio(.75,4/3),HorizontalFlip(.5)|same augmentation operators/ranges, output32x32 for CIFAR; no Cutout|
| Normalize |ImageNet mean/std|existing Dense-SAM CIFAR parent mean/std, exact frozen values|
| Loader |shuffle=True,default workers8|shuffle=True,workers0; CPU RNG isolated and reset for each layer to replay identical views/order|
| H accumulation |all reference hooks see the same dense forwards|layer-by-layer memory-bounded accumulation, identical augmented input stream replayed for each layer|

Official source has no native CIFAR path. This is **CIFAR controlled adaptation**,
not strict reproduction of an official CIFAR configuration. Each round generates
fresh crops/flips, rather than replaying one cached batch ten times. Each layer
receives the same ten-round stream; order/tensor SHA receipts prove this at run time.
Data provenance uses complete official CIFAR files; no test samples calibrate H.

## Frozen scope and provenance

- Config: configs/obc_seed0_4090_pilot.yaml.
- Config SHA256:6903c2bb60feec7ceed87e0e77846ae07e25c4e7a8770ae79b09401449eecaaf.
- IDs:configs/obc_seed0_1024_train_ids.json.
- IDs file SHA256:1ccaddc4a0a2f531f7500bf2e403e4302978dc35e98abac179d262bbc358f2f2.
- IDs int64-LE SHA256:e6de753094bd50b1f75b7833893ef7d465069f977596457bf0814e2839bd4702.
- Parent:existing standard Dense-SAM seed0 best_test.pt,
  SHA25663c2415476802ee30029d445bd2b1fed0992ec39ab0c89a852cf14b1c9683897.
- GPU:NVIDIA GeForce RTX4090,UUID GPU-6d21beb3-4922-a03d-55b9-b081fff80cee,cuda:0.
- Full existing2:4eligible scope:exclude_first_last,OIHWoutput-row-contiguous,
  dense tail,BNaffine excluded,official argmin tie rule. parallel32,relative damping0.
- FP32,no AMP/TF32/compile,deterministic,threads4.
- Start reserve:12GiBfreeVRAM and22GiBfree disk. Full-model peak/time pending pilot.

The original 32-image smoke evidence is retained byte-for-byte. The new lineage
gate checks actual PASS report and independent audit digests, tested device and
unchanged numerical core modules, rather than relabeling it as a full-model test.
OBC/model/N:M/Hessian/compensation/pipeline/recovery/BN/artifact numerical modules
are unchanged from CUDA-tested19ff5dc. Only calibration and orchestration/gates
are extended and covered by new targeted CPU tests. New configuration has a new
source identity, commit and output path. No receipt is fabricated for new GPU work.

## Run stages and selection

1. Load existing seed0DenseSAM BestTest checkpoint and verify all identities.
2. Full eligible-layer OBC native compensated output; save layer boundaries/final.
3. Evaluate dense reference and original native compensated checkpoint on10000Test.
4. Skip BN-only as explicitly requested. Recovery starts from native pre-BN output.
5. Exactly20epochs/7800steps fixed-Mask fresh SGD:lr.005,momentum.9,wd.0005,
   nesterovfalse,cosine eta_min0 at epoch end,original training augmentation/order.
6. Save all20epoch checkpoints,strict BestTestAccuracy earliest tie,and Last.

Test is used for checkpoint selection, so BestTest is not an unbiased independent
test estimate. Native compensated OBC is not uncompensated Raw Hard. Final/loss,
source relationships and complete epoch history remain separately available.

Persistent resource telemetry and disk checks cover calibration, evaluation and
recovery. Strict finite/Mask/parent/optimizer checks remain unchanged. On error,
preserve traceback/logs; no automatic retry,resume,parameter change or next seed.
SSH disconnection does not terminate the nohup launch. A successful pilot does not
authorize seeds1/2; they require user review. No SNOWS or other GPU queue is started.

Obtain the literal run commit after this freeze is committed; deployment receipt
records commit, full source SHA, exact launch command and runtime precheck results.
The initial test failure (AST parsing negative CLI defaults as Constant) is kept in
tests.xml; only that test harness is corrected to ast.literal_eval and rerun.
