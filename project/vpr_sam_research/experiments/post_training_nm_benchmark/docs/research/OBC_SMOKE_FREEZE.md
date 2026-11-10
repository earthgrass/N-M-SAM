# OBC approved smoke freeze —2026-10-25

Approved scope: CIFAR10 official train32IDs, seed0,batch8,noaug,1round, parent ToTensor/Normalize; exact CIFAR-ResNet18; partial layer1.0.conv1; FP32/no AMP/TF32/compile; independent BN copy;2fixed-Mask SGD steps. No GPU run authorized this turn.

## Frozen sample IDs (draw order, never resampled)

```json
[46742, 31828, 13864, 3760, 825, 31604, 27988, 13481, 48520, 42327, 30320, 2047, 38242, 19704, 27173, 15383, 45619, 136, 42865, 40784, 1679, 33537, 36481, 40645, 32457, 25171, 8759, 8782, 36463, 42504, 27712, 25542]
```

Index JSON byte SHA256: `ef5df4e0b96367926d2d4608eed154b401b070239b404b4b77eaa266d8e5ad5e`.
Canonical int64-LE index SHA256: `74a176121486328ddaf2716b86e90f2ce23a67bfa5df1393dafa3280685a3fb1`.
Parent checkpoint SHA256: `63c2415476802ee30029d445bd2b1fed0992ec39ab0c89a852cf14b1c9683897`.
Checkpoint-bound full data-manifest SHA256: `8cf7d9fe3fb38c390718a7fa52ee4977519388287cc1cc61d5ddb20c5fd57011`.
Smoke config SHA256: `0c06373b1dc50da13241188c6e1ebf7b8d6e6adcfab9e798c6ed3487dcaa14d8`.
Source/config/scripts/patch digest: `365271c98cd693c1055b2f1c660cc622942acd6e9398d40bb6f9a3ecfd904de3`.

Generation: existing deterministic_indices helper, NumPy default_rng(0)/PCG64 choice(50000,32,replace=False), draw order retained. The index file is loaded at execution and is never regenerated. Freeze helper refuses an existing production index file.

## Parent source and preprocessing evidence

The actual seed0 Dense-SAM file passed byte SHA/schema/healthy/model digest checks. Its embedded source hashes matched SOURCE_FREEZE; original data.py/config.yaml bytes and embedded config digest matched. The actual Images class AST matches the runtime independent copy. Training=False executes only ToTensor then Normalize, no crop/flip/Cutout/TTA.
Normalize mean=[0.4913725490196078, 0.4823529411764706, 0.4466666666666667]; std=[0.24705882352941178, 0.24352941176470588, 0.2615686274509804].
Five training files are ordered data_batch_1..5,10000rows each. Every selected ID maps to file/row and file SHA in configs/obc_smoke_freeze.json; no test_batch row is selected. Full dataset-file SHA map is retained there. Raw server files are not locally re-read while server is off; their checkpoint-bound identity is verified, and future runner validates actual full file manifest/order before pruning.

## Execution acceptance

Source/config and index file bytes are included in the source digest. The new clean Git commit is obtained with git rev-parse HEAD after committing this report; the post-commit FINAL_READINESS receipt records that exact commit, source digest, config digest and fresh pilot checkpoint byte check. This avoids a self-referential commit hash inside its own content.
New/affected OBC9tests pass:5freeze/preprocessing/tamper/exact optimizer checks and4affected smoke/gate checks. Prior compensation/N:M/BN/recovery proofs are reused. SNOWS9step tests plus2actual-interface checks pass after preserving7test-harness failures and targeted retries; no old full suite rerun.
Evidence packet: docs/research/evidence/approved_smoke_stepfix_20261025_final. Earlier preparation snapshot remains separately preserved.
Expected planning reservation:8GiB free VRAM/4GiB disk, approved RTX5090 UUID. No GPU peak/time is measured. CUDA smoke command and strict failure/no-overwrite/logging policy: post_training_nm_gpu_runbook.md. Formal OBC and SNOWS budgets remain PENDING_APPROVAL.
