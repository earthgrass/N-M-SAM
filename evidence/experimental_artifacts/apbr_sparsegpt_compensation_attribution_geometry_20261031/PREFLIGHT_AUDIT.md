# Preflight Audit

Date: 2026-10-31 (Asia/Shanghai)

Status: **PARTIAL PASS / GPU BLOCKED**

- Repository branch: `research/post-training-nm-benchmark`; HEAD observed before this task: `55fe3e300dbbd09c1178a6316afebc9ecf367947`.
- APBR Dense Final Seed 0 and Seed 1 are present and retain their registered SHA256 identities.
- Original SparseGPT and Radius-Guided SparseGPT masks/native checkpoints for Seed 0 and Seed 1 are present.
- The downloaded Radius-Guided evidence manifest contains 42 files; local verification reported zero SHA256 failures.
- Six independent Mask-only counterfactual checkpoints were constructed from the matching Dense parent and frozen mask: four APBR comparisons plus two Dense-SAM Original-SparseGPT references.
- The prior GPU host stopped accepting connections before deployment of this task. Local PyTorch is CPU-only. Final Test evaluation and the CUDA single-layer pilot were therefore not run or relabeled as passed.
- Seed 2 remains excluded because the authentic APBR Seed-2 Dense Epoch-200 parent is unavailable.

No existing Original SparseGPT, Radius-Guided SparseGPT, or APBR+OBS artifact was modified.
