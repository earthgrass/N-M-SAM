# APBR-GPT Preflight Audit

Date: 2026-10-31 (Asia/Shanghai)

Status: **SOURCE GATE PASS / GPU PILOT NOT RUN**

- Seed 0 and Seed 1 APBR Dense Epoch-200 parents, matching geometry states, Dense-SAM parents, Original SparseGPT results, and Radius-Guided results are available and hash-bound.
- Seed 2 remains excluded: its authentic APBR Dense Epoch-200 checkpoint is unavailable.
- Historical results are read-only and remain in their original directories.
- The previously used RTX 5090 endpoint stopped accepting connections. Local PyTorch is CPU-only, so no CUDA Pilot or Final Test is claimed.
- Full-network APBR-GPT is not authorized until the new single-layer Exact/Shared Pilot passes its numerical and resource gates.
