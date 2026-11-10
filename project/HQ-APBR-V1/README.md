# HQ-APBR V1

Independent research workspace for **Historical Compensability-Guided Adaptive Perturbation Budget Reallocation**. The APBR/SAM/SparseGPT project under `../APBR-post-training-benchmark/vpr_sam_research` is a read-only reference; this folder has its own Git history and does not import historical model code at runtime.

Current scope: repository and mathematical audit, standalone CPU numerical reference and tests, and resource feasibility audit. No 200 Epoch training, Final Test, recovery, or HQ performance claim has been run.

Run the local CPU tests with `python -m pytest -q tests --basetemp=artifacts/pytest-temp`. The [numerical audit](reports/hq_apbr_cpu_numerical_audit.md) records the final CPU gate. The [phase status](reports/hq_apbr_phase_status.md) separates mathematical results, CPU verification, and the blocked real GPU performance gate. Versioned evidence belongs under `reports/`; scratch data belongs under ignored `artifacts/`. Verify the frozen file list with `python scripts/sha256_manifest.py --verify`. A real GPU measurement remains required before any pilot training.
