# Joint APBR / SAM-Joint archive mapping

Paths in `joint.json` are relative to the NM workspace root. This is an archival inventory; no rerun was performed.

## Research lines

- `joint_optimization/` and `baseline_adapters/apbr_gpt_joint/`: training-time second-order pruning-aware Joint objective, repaired APBR gradient handoff, shared-Hessian candidate factors and dynamic post-training OBS.
- `baseline_adapters/apbr_gpt/` and `baseline_adapters/group_joint_obs/`: independent static post-training research. Keep in another branch, not labeled Joint training.

## Executed cohorts

| Cohort | Dense | Native GPT | Groupwise | True geometry/probe | Shuffled |
|---|---:|---:|---:|---:|---:|
| apbr_joint_seed0_full | 78.65% | 78.36% | 78.55% | 78.10% | 78.10% |
| sam_joint_seed0_full | 78.45% | 78.43% | 78.53% | 77.73% | 77.72% |
| sam_joint_seed1_full | 78.75% | 78.62% | 78.58% | 78.05% | 78.05% |
| apbr_joint_seed1_full | 78.34% | 78.29% | 78.33% | 77.94% | 77.95% |
| apbr_joint_seed0_epoch10 | 78.58% | 77.74% | 77.87% | 77.81% | 77.80% |

All rows above have 200-epoch/78,200-step completion and five frozen Final Test records. No Recovery-20 is represented in these endpoints.

## Identity and historical boundaries

- Joint training is joint_optimization plus baseline_adapters/apbr_gpt_joint. baseline_adapters/apbr_gpt and group_joint_obs are separate post-training experiments.
- Original Joint phase4 GPU smoke was BLOCKED by missing gradient handoff. phase4_1 fixes the interface and records APBR_PARITY_PASS, JOINT_CPU_GATE_PASS, GPU_SMOKE_PASS; retain both histories.
- The top-level rtx5090 YAML describes an earlier three-branch/recovery plan. Actual executed cohort is narrowed by queue source and per-run manifests; do not label those historical planned recoveries as completed.
- Full Joint guidance epochs are 6 through 200. Epoch10 guidance runs at one-based epochs 10,20,...,200. The generic schedule list in epoch10 manifest is stale; the guidance_epochs_one_based field and runner guidance_due/actual hessian_refresh.csv determine execution.
- APBR Joint uses original VPR/ellipsoid APBR, not APBR-L2. SAM Joint uses SAMJointStepper and does not learn APBR geometry.
- All five listed cohorts have 200 epochs/78200 steps and five final endpoints. APBR Seed1 epoch10 was cancelled and must remain incomplete.
- Current local source snapshots are not automatically bitwise identical to executed historical versions. Preserve source hash manifests and source copies per run; do not claim commit55fe3e contains the later untracked Joint implementation.
- No experiments, Test evaluations, remote connections or large-checkpoint hash scans were performed by this inventory.

## Package selection

Include the three artifact roots named in `joint.json`, with small text evidence and separate external-asset manifests. Also include workspace `joint_seed0_copy/`; its Seed 0 evidence is more complete than the mostly empty `phase5_phase6_final_remote_copy` subdirectory in the experiment tree. Exclude virtual environments, cached bundle duplicates, test temp directories, binary weights and datasets from Git. Preserve the original bundle integrity manifests and phase reports.

## Resume entry points

- Existing completed runs should remain completed; archive does not authorize resume or Test reruns.
- For inspecting the original complete Seed0 plan: `scripts/run_joint_seed0_only_queue.py`.
- Seed1 full and time-sparse plan: `scripts/run_joint_seed1_sparse_queue_v2.py`; historical parameter mapping in `scripts/launch_joint_seed1_sparse_v2.sh`. Do not execute its four-stage queue unmodified: the last stage was explicitly cancelled.
- Real lambda/eta calibration: `artifacts/apbr_gpt_joint_seed0_rebased/real_cpu_calibration_attempt4/`; lambda_p = 1.3526042410542212, eta_s = 7036.300421582199.
- REBASED initialization identity and asset requirements: `REBASE_DECISION.json`, `ASSET_RECOVERY_REPORT.md`, `PHASE3_FINAL_REPORT.md`, and per-run `manifest.json`.
- Runtime source mapping must include `run_cifar100_50k_formal_training.py`, `src/vpr_sam`, model factory, benchmark N:M helpers, common Conv2d/N:M adapter utilities and SparseGPT adapter. Preserve original relative package layout.
