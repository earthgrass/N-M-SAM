# Seeds 0/1/2 extension — user-authorized overnight queue

Seed0 remains immutable at commit cea8f8c, on its original storage partition.
Reuse its existing SAM and currently running VPR experiment, without restarting
or altering either. After its entire three-branch evaluation/recovery passes,
run SAM1 -> VPR1 -> three-branch postprocess1 -> SAM2 -> VPR2 -> postprocess2.
No GPU tasks run concurrently. Stop the queue on any failure; no automatic retry.

New seeds1/2: identical fifty-thousand/official-ten-thousand full-data protocol,
200epochs/78000steps, exact SSAM model/augmentation/cosine sequence, dual-BN,
SGD .05/.9/.001/no Nesterov; rho.1; deterministic FP32, AMP/TF32/compile off.
VPR uses the same byte-identical fixed_l2 geometry, FP64 gamma
5.482145661475941e-8, no warmup, clipping, EMA or volume correction.
Within each seed both methods start from the same random initialization and
must match data identity and every training-order hash. No warm start.

For each seed: SAM+Magnitude, VPR+Magnitude, VPR+|w|exp(-s); same VPR Best
weights/geometry for both Mask branches. Same 2:4 scope/group/tail/tie rules.
Independent Raw and BN-only evaluation; common4096 training calibration IDs;
three matched20epoch fixed-mask fresh-SGD recoveries. Recoveries save every
epoch plus Best/Last; dense saves Best/Last only. Test selects Best (strict
Accuracy, earliest tie), so Best Test is not independent unbiased evaluation.

New isolated worker directories are deployed on /root's separate filesystem;
no cleanup of historical data or checkpoints is needed. All four configs and
worker scripts are committed before queue launch. Config deployment is a
byte-copy from configs/{method}_seed{seed}.yaml to worker config.yaml; source
manifests record this Git path mapping explicitly. No source/config changes
are made during training. Each worker gets a full source/config/environment,
native exact400-field LR, RNG, initialization, data, and checkpoint audit.

The SAM parent digest for VPR cannot be known before SAM finishes. It is
resolved read-only from the same frozen bundle's completed audited SAM run,
cross-checked against atomic publication and final audit manifests, and then
saved in the VPR run's immutable anchor receipt. Seed, common configuration,
Best selection, initialization and data hashes are verified; no guessed hash.

Primary report endpoints: Raw, BN-only, Recovery Best Accuracy/Epoch/Loss,
Recovery Final Accuracy/Loss, pruning damage and recovery gains. Report all
three seeds, mean/sample SD, within-seed paired differences and exploratory
95% paired t intervals (df2); seeds are independent units. No epoch/batch
pseudoreplication, no result-driven changes, no claims of SOTA or universal
advantage. Keep individual source commits and checkpoint hashes, since
seed0's already running audited snapshot must remain unchanged.

Queue completion stops further GPU work and generates the aggregate report.
