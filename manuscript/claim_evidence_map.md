# Claim/evidence map

| Claim | Evidence | Boundary |
|---|---|---|
| Historical SAM + Magnitude/Wanda/GPT/ROSE means and sample SD | `evidence/historical_three_seed_analysis.json`, `three_seed_summary` | All are historical three-run endpoints; GPT/ROSE include native compensation |
| SAM-Joint + native GPT two-run mean 78.53%, sample SD 0.13 pp | `evidence/sam_seed0_final_test.json`, `evidence/sam_seed1_final_test.json` | Joint cohort does not share verified identical initial weights with historical SAM |
| Final GPT selection and OBS are separate from training proxy | `scripts/run_apbr_gpt_joint_matrix_prune_seed01_v2.py`, `selector == original_gpt` calls `sparsegpt_matrix` | Audited independent adapter, not literal official-code import |
| Six-pair soft-min is differentiable | `baseline_adapters/apbr_gpt_joint/adapter.py`, `softmin_pruning_regularizer` | No hard training mask |
| Computation time | `COST_PROFILE.md` and CSV evidence | Deliberately excluded from current manuscript pending operator-level profile |
