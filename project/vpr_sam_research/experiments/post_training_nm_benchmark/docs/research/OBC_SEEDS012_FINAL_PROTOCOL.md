# OBC seeds0/1/2 — Final endpoint continuation

User approved seeds1 and2 after seed0 completed and passed the independent audit.
Run seed1 -> CPU artifact audit -> seed2 -> CPU artifact audit, strictly serial on
the approved RTX4090. Any run/audit failure stops the queue; no retry or resume.
No dense training, repeated CUDA smoke, BN-only, SNOWS or extra seed.

New config `configs/obc_seeds12_4090_final.yaml` has SHA256
`7c6b6d96c4840ab83a59c2e010f46089ab15c4a01139df238172698473d28721`.
Only name, registered seeds and explicit reporting fields differ from the frozen
seed0 configuration. Existing numerical/calibration/training/runner modules
remain byte-identical to commit6c4e4d08b54813a07ba181a1f061c9690f6137b7.
The separate new CPU audit and queue orchestration do not change computation.

Use each model seed's approved existing standard Dense-SAM best_test.pt input.
The new Final reporting request selects recovery epoch20/last.pt; it does not
silently replace the already approved dense parent convention. Report dense
parent selection explicitly. Seed0 remains untouched in its original workspace.
Same frozen calibration1024IDs, calibration RNGseed0 and ten augmented rounds
are shared across model seeds; recovery uses its corresponding seed1 or2.
Batch128,parallel32,2:4scope/tails/ties,precision,Normalize,SGD20epoch schedule,
augmentation and checkpoint publication are unchanged. Best checkpoints remain
auxiliary existing outputs, never the primary summary endpoint.

Every recovery Final must be epoch20/7800steps, with unchanged Mask and zero
deleted-coordinate weights/momentum. CPU audit validates all24saved checkpoints,
parent hashes,19-layer calibration coverage/replay,selection/publication and
source identity. Seed0's audited Final is included in the three-seed summary.
Report native compensated and recovery Final accuracy/loss for each seed,
mean and sample standard deviation. No mixing best epochs into Final statistics.
Native OBC compensation is not uncompensated Raw Hard; Test-selected dense
parents retain their selection limitation even though recovery endpoint is Final.

After completion, synchronize reports,all epoch metrics/logs,configuration/hash
receipts and three Final recovery checkpoints to a new local delivery directory;
verify complete file SHA256. Other large epoch/native checkpoints remain on the
server unless separately requested. Do not delete server or local history.

The queue creates a provenance-bound pilot-review receipt from the actual seed0
PASS audit, unchanged implementation bytes and exact scientific protocol equality;
it does not pretend seed0 ran with the new source/config digest. Freeze a new
clean commit and record it in the launch receipt before starting either seed.
