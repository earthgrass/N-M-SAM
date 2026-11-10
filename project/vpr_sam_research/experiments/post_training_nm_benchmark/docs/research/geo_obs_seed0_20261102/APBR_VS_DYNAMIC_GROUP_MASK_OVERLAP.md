# APBR Raw-Hard vs Dynamic Group-OBS Mask overlap, Seed 0

This read-only comparison loads the saved APBR Raw-Hard Mask and saved Dynamic Group-OBS Mask for each **identical Dense parent**. Parent SHA256 values match within each pair. All 19 eligible layers contain 11,157,504 positions and 2,789,376 complete four-weight groups; each Mask deletes exactly 5,578,752 positions and is strict 2:4. No model was run or Test data accessed for this audit.

| Same Dense parent | Identical keep/delete positions | Different positions | Identical complete 2:4 groups | Different complete groups | Groups sharing one deleted position | Groups sharing none |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| APBR-L2 Seed 0 | 8,637,862 / 11,157,504 (**77.42%**) | 2,519,642 (**22.58%**) | 1,587,467 / 2,789,376 (**56.91%**) | 1,201,909 (**43.09%**) | 1,143,997 | 57,912 |
| Original APBR-VPR Seed 0 | 8,645,870 / 11,157,504 (**77.49%**) | 2,511,634 (**22.51%**) | 1,591,604 / 2,789,376 (**57.06%**) | 1,197,772 (**42.94%**) | 1,139,727 | 58,045 |

Because both masks delete half of every group, the fraction of identical keep/delete positions also equals the fraction of APBR-deleted positions deleted by Group-OBS. The latter count is 4,318,931 on APBR-L2 and 4,322,935 on original APBR-VPR. A different group usually swaps one deletion choice, hence 22.5% position disagreement coexists with approximately 43% group disagreement.

This comparison does **not** isolate the scoring formula alone. APBR creates a Mask once from the original Dense weights using `abs(w)*exp(-s)`. Dynamic Group-OBS makes each group's decision after earlier groups have been compensated and compares six joint deletion costs. Both the score and the state at decision time differ.

Full per-layer counts and matching parent hashes are in `APBR_VS_DYNAMIC_GROUP_MASK_OVERLAP.json`. The source masks are preserved in the formal APBR Raw-Hard checkpoint and the isolated Dynamic Group-OBS method checkpoint; no files were changed.
