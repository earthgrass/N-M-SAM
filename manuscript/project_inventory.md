# Project inventory (2026-11-05)

Independent paper root: HC-SAM-paper. Original `paper/` and HQ-APBR-V1 are read-only sources for this task.

- Original APBR PDF/TeX inspected for motivation and structure; no original paragraph copied. Inspection metadata in evidence/original_draft_inspection.txt.
- HQ-APBR-V1 README, docs/hq_apbr_v1_math.md, math_core.py, training.py, calibration.py, evaluation.py and run_authorized_seed0.py inspected.
- Formal source 90499e96a87ac8cfdbd198f9d31184f6ae5e0959; current local repository snapshot can contain later reporting-only commits. Do not attribute later helpers to the running training process.
- Config SHA256: 56ff197b0860dfd0f4a0768c5609031214e6405c51557009a9977582b38400e7.
- Official CVPR author-kit pinned by template/provenance.json. Upstream main.tex retained under template/; paper main.tex is independently authored.
- Frozen snapshots and hashes: evidence/source_snapshot_manifest.json. No checkpoint is copied into the paper repository.
- Literature primary-source metadata and citation lock reside in literature/ and citation_lock.json.

Original APBR is an energy-feedback method. HQ/HC uses EMA of centered inverse-Gram logs; it does not inherit APBR's balance-point or learned-mask claims.
