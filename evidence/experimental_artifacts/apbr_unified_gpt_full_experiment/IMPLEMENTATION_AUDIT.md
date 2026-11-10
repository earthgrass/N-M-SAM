# Implementation Audit

New isolated module: `baseline_adapters/apbr_gpt/adapter.py`.

The frozen Original SparseGPT adapter was not edited. The new module provides:

- global eligible-coordinate `exp(-2s)` normalization;
- Q construction after the original damping rule;
- Exact Row-wise Q factorization;
- explicitly labelled Shared Geometry approximation;
- dynamic N:M selection and sequential OBS updates driven by the same Q;
- direct Original SparseGPT dispatch at `lambda=0`.

Verified locally:

1. A common additive shift in all eligible `s` leaves normalized q unchanged.
2. Global normalized q has mean one.
3. `lambda=0` produces the exact Original SparseGPT Mask and weights.
4. Exact and Shared modes match exactly when q is shared across output rows.
5. Nonconstant row geometry remains finite, strictly 2:4, and keeps every deleted coordinate exactly zero.

Targeted APBR-GPT tests: 4 passed. Combined affected regression selection: 25 passed. CUDA behavior remains unverified because no GPU is currently reachable.
