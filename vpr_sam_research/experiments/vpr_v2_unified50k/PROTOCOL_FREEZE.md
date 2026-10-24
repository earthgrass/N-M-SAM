# VPR-v2 / SSAM50k dual-BN paired protocol

Authorized seed: 0 only. One new fixed_l2 dense trajectory (200 epochs,
78000 updates), followed by three matched pruning/recovery branches.
Existing SAM is imported read-only, never retrained or used as initialization.

Anchor: Dense-SAM best_test.pt, epoch193, Test Accuracy96.77%,
Test CE0.10218338812887669; producer commit
1dffe6d269bb1ec9fd67611a59f2bb538183b30e; checkpoint SHA256
63c2415476802ee30029d445bd2b1fed0992ec39ab0c89a852cf14b1c9683897.

All common training fields are copied from the anchor's frozen config:
full50000/official10000, no validation split; exact SSAM2022 CIFAR-ResNet18;
batch128, drop_last=True; SGD .05/.9/.001/no Nesterov; rho=.1;
original SSAM cosine and epoch-end zero-based call order; crop4, flip.5,
exact SSAM Normalize and post-normalization Cutout16 probability.5.
Both SAM forwards update BN running mean/variance and counters. This follows
the user's final dual-BN clarification, not the earlier single-update proposal
and not F-SAM's second-forward momentum0 policy.

VPR-only difference: byte-identical audited fixed_l2 geometry path;
epsilon=rho*A^2*g/||A^2*g||; p=epsilon^2/sum(epsilon^2);
s_next=s+gamma*(1-d*p). Gamma=5.482145661475941e-8, FP64 geometry,
zero initialization, global fixed dimension, no warmup, no EMA/clipping or
volume recentering. FP32 model; deterministic; AMP/TF32/compile disabled.
All initialization and per-epoch sample-order hashes must match the anchor.

Dense checkpoints: best_test.pt (strict Accuracy improvement, earliest tie)
and last.pt only. Test participates in selection; Best Test is not an
independent unbiased generalization estimate. Parent for all pruning is
best_test.pt. VPR weights and geometry must come from that same checkpoint.

Branches: SAM+Magnitude; VPR+Magnitude; VPR+|w|exp(-s).
2:4 output-row flattened groups; exclude first/last Conv/Linear; dense tail;
stable ties keep lower coordinate. BN affine and biases are not masked.
BN-only uses an independent Raw copy and frozen4096 training IDs, no
augmentation, BN-only train mode, reset running statistics, momentum=None;
no trainable weight update. Raw remains unchanged.

Each recovery starts from untouched Raw, fresh SGD .005/.9/.0005/no Nesterov,
20epoch cosine eta_min0; full50k, batch128/drop_last, matched augmentation
and data order across branches. Fixed masks project weights, gradients and
momentum. Every recovery epoch saved, plus Best Test and Last. No dense
warm-start, extra seed, parameter search, new scoring rule or historical overwrite.

CPU/CUDA tests, native exact400-field LR comparison, real-data two-step
smoke, geometry resume, source/data/GPU/disk and historical-integrity gates
must pass before formal training. All smoke weights discarded. Any gate or
runtime failure stops the queue with evidence and no automatic restart.

Outputs stay in this project's artifacts directory. Existing dataset is
explicit read-only input. Immutable anchor geometry is referenced by parent
SHA instead of redundantly copied into each recovery checkpoint.
