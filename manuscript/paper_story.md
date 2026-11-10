# Paper story (provisional)

**Question:** Can dense SAM training be guided by an activation-derived, pairwise N:M deletion objective so that frozen native SparseGPT works better afterward?

**Mechanism:** A differentiable soft minimum over six pairwise 2:4 deletion costs augments the SAM adversarial objective. The model stays dense until the fixed final checkpoint is pruned by the audited SparseGPT adapter with native dynamic selection and OBS-style compensation.

**Observed signal:** The two available Joint runs yield 78.53% ± 0.13% at native SparseGPT, compared with 77.26% ± 0.42% for the three-run historical SAM-to-SparseGPT reference. Cohorts are not initialization-matched, so this is exploratory association, not a causal treatment effect.

**Main table:** Magnitude, Wanda, SparseGPT, ROSE, and Joint, with only method and mean ± sample SD accuracy columns. Joint denotes SAM training with the pairwise objective followed by native SparseGPT. No individual run IDs or Recovery-20. SparseGPT, ROSE, and Joint are compensated at their immediate endpoints.

**Deferred:** APBR and quantitative computational cost remain outside this manuscript version. Do not claim state of the art, significance, matched causality, or a cost advantage.
