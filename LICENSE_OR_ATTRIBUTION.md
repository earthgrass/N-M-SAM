# Sources and attribution

Algorithm reference: Google Research SAM, Copyright 2020 The Authors,
Apache License 2.0: https://github.com/google-research/sam/blob/dae9904c4cf3a57a304f7b04cecffe371679c702/LICENSE

Pinned source: https://github.com/google-research/sam/blob/dae9904c4cf3a57a304f7b04cecffe371679c702/sam_jax/training_utils/flax_training.py
Supplementary: `sam_jax/train.py` and repository README at the same commit.
The PyTorch code here is independently written from mathematical/update semantics;
no upstream source files or prior project code are copied.
No project-wide license grant is inferred for the user's new research code.

Research inputs read on 2026-10-13: `NMSAM_FROM_SCRATCH_SPEC.md` (v0.1),
`10.12_NM_SAM_summary.docx` (V2.0, including OMML equations and tables),
and the supplied **`N_M+SAM.pdf` (15 pages)**. A file named `N_M+SAM(2).pdf`
was not supplied. PDF text extraction is imperfect for handwritten formulas;
the residual decomposition on page 14 was also inspected visually. The SPEC
and Word equations are the explicit mathematical contract. Historical accuracy
tables are background only, never validation of this rebuild. The attachments
are not republished or copied into the new project.

