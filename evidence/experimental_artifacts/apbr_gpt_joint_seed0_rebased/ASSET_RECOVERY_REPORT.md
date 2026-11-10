# Asset recovery and rebase decision

Status: `REBASED_COHORT_READY`.

The historical Seed 0 initialization was not recovered. The search evaluated 66 loadable model-state records and found no tensor hash equal to the registered historical target `87aa9dd7...759d`. Seed 0/1/2/3 final checkpoints preserve their original initialization hashes as metadata, but none contains the epoch-0 tensors. Reconstructing with the current model definition and runtime produced a different hash, so it is not described as historical recovery. Evidence is retained in `../apbr_gpt_joint_seed0/historical_tensor_scan.json` and `../apbr_gpt_joint_seed0/cross_seed_asset_inventory.json`.

Following the user-authorized rebase path, a new cohort named `apbr_gpt_joint_seed0_rebased` was created. Its single immutable initialization is `assets/initial_common_seed0.pt` (file SHA256 `4317514e...03c`; model-tensor SHA256 `6c16e81a...2ddc`). Dense-SAM, APBR-SAM, and Joint APBR-GPT branch references all point to this exact checkpoint. APBR and Joint APBR-GPT also share the same all-zero FP64 initial geometry (`20771c3a...ae09`).

No historical final checkpoint is used as an initialization. Consequently, all three rebased branches require new 200-epoch training before a paired result can be formed. Historical Seed 0 results remain descriptive references only and must not enter rebased paired statistics.

No CUDA, server, validation set, Final Test, or formal training was used in this decision.
