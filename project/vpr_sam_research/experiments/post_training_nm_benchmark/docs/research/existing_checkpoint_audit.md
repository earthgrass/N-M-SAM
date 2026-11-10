# Existing experiment audit — Stage 1 PASS

New isolated worktree: `APBR-post-training-benchmark`, branch
`research/post-training-nm-benchmark`, base commit
`d0025b48664091444cc1ad363253010c336b23e6`; initial status clean.
The shutdown GPU server was not contacted. All15original Best files passed
SHA256/schema2/healthy/unperturbed/model digest/seed checks. Six exact dense
loads passed synthetic CPU eval forwards; this is not a dataset measurement.

The user confirmed the alias: APBR-SAM = completed Full VPR-L2 + Geometry-guided.
Keep historical `fixed_l2`/`vpr_guided` names in provenance. The weight-only
ablation is `vpr_magnitude`, sharing its dense parent and FP64 geometry with
the full APBR row. No files are renamed, retrained, or regenerated.

Exact architecture: SSAM2022 CIFAR-ResNet18,3x3 stride1 stem, no ImageNet
maxpool, four2-block stages, fixed4x4 average pooling,10-class head;
11,173,962 parameters and62parameter tensors. Not torchvision ImageNet ResNet18.

Artifact inventory `artifacts/stage1_inventory.json` records all15paths,
checkpoint/model/geometry/Mask hashes, epochs, producer commits, roles and
runtime. It distinguishes Dense-SAM, APBR dense+embedded geometry,
SAM+Magnitude recovery, APBR weight-only recovery, and full APBR recovery.

Local runtime: Python3.12.10, PyTorch2.13.0+cpu, torchvision0.28.0+cpu;
no CUDA. Historical training runtime: PyTorch2.8.0+cu128/torchvision0.23.0+cu128.
CPU success is not CUDA validation. Model/data definitions are copied into
this isolated package; there are no runtime Python imports from old projects.

Recovery copied from original configuration:20epochs,50000train/10000official
Test, batch128/drop_last,390steps/epoch; fresh SGD lr.005,momentum.9,
weight_decay.0005,nesterov=False; epoch-end CosineAnnealingLR eta_min0.
RandomCrop32 padding4, Flip.5, exact SSAM Normalize, post-normalize
Cutout16 p.5. Project weights/gradients/momentum onto fixed Mask.
Strict maximum Test Accuracy selects Best, earliest tie; retain Final/all epochs.
Test participates in both dense and recovery selection, so it is not an
independent unbiased endpoint. This limitation is retained, not retroactively fixed.

Existing BN-only control: independent Raw copy; frozen4096train IDs,
no augmentation; BN-only train mode, reset running statistics, momentum=None,
32batches of128; no trainable weight changes. Recovery starts before BN-only.

Raw/BN/Last/all-epoch large weights remain on the shutdown server. All15Best,
their9Masks, full logs and metrics are local. Missing large historical files
are explicitly recorded, not recreated; they are unnecessary for reusing
existing E0/E1 results or loading the new baselines' exact Dense-SAM parents.
