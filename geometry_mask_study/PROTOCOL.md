# Frozen three-mask diagnostic: VPR-L2 seed0 final

Explicitly authorized extension; no training, optimizer steps, feedback changes,
geometry changes, additional seeds or Test metrics. Existing Magnitude recovery
is completed independently before this GPU evaluation. Radius-only or Geometry-
guided recovery is NOT authorized in this phase and has no execution path here.

Read only the original200epoch/70400step schema2 final checkpoint, its FP64
geometry state and original Magnitude artifact under producer commit f742733.
Use the original model parameter name/shape/order mapping and strict Geometry
state loader. d remains global; BN affine geometry is retained in the mapping
but never included in mask-eligible Conv2d/Linear weights. Layer selection,
per-output flattened blocks, N=2/M=4 and dense tail come from the original
pruning protocol and make_masks selection. Reject incompatible producer/config.

A keeps the two largest |w|. B keeps the two smallest s (deletes two largest s).
C keeps the two largest |w|exp(-s). All use stable descending keep-score sorting
with lower coordinate index winning a tie; B's keep-score is -s. This matches
the existing Magnitude tie rule. No score clipping or adaptive choice. Use FP64
score arithmetic and fail if inverse radii/costs are nonrepresentable.

Report pairwise A/B, A/C and B/C Hamming flip rates over complete-block
coordinates, plus a separate including-dense-tail denominator; different full
blocks / total full blocks. Report per-layer and count-weighted global values.
For each mask report ||w_deleted||_2, its square, the geometric deletion cost
sum_deleted (w exp(-s))^2 and its square root. C minimizes the declared diagonal
geometric cost by construction; this cannot by itself establish better loss,
validation accuracy or recoverability.

All models start as independent copies of the same final dense model. Raw Hard
preserves its BN buffers exactly. Evaluate all5000 held-out validation samples,
batch256. Then recalibrate only BN buffers on each independent pruned copy with
the previously frozen4096 train IDs, no augmentation, fixed batches128×32,
reset_running_stats, momentum=None, only BN train/others eval, no_grad and all
trainable weights frozen. Evaluate the same5000 validation samples afterwards.
Reproduce the previously recorded Magnitude Raw and BN values exactly.

Hash every existing dense/Magnitude/recovery artifact and prior report before
and after. Do not overwrite original source/config, checkpoint, mask or reports.
New artifacts go only under artifacts/geometry_guided_mask_seed0. Preserve
new mask tensors, layer CSV, per-method metrics, source/geometry/checkpoint
identities and final audit. Stop on any numerical/provenance/replay discrepancy.
