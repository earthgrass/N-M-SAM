# Independent Joint paper draft

This folder is an isolated new manuscript under the pre-existing `NM稀疏/paper` directory. It does not replace `APBR-SAM_draft_1030.pdf` or `paper/main.tex`.

* `main.tex`, `main.pdf`: English draft following the author's A4, Times New Roman, Roman-section and lettered-subsection format.
* `COST_PROFILE.md`: separate research audit. Concrete timing and APBR results are intentionally absent from the manuscript for now.
* `evidence/`: copied Seed 1 training/final-test records, an analysis script and its JSON output.
* `paper_story.md`, `claim_evidence_map.md`, `experiment_inventory.md`: scientific scope and provenance.

The draft calls the proposed pipeline Joint and focuses on its SAM training plus native SparseGPT/OBS readout. Its abstract leads with the method, and its main table contains only method and accuracy. APBR remains in the earlier paper and the independent cost audit. Historical and Joint accuracy cohorts are not verified to share the same initial weights; their accuracy difference is not a causal ablation. The sparse-in-time experiment is still running and has no reported result in this version.
