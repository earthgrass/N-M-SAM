"""Independent seed is the statistical unit, including post-pruning outcomes."""
import math
import numpy as np
from ..repro import seed_for


def seed_statistics(observations, baseline="sam"):
    """observations[method][seed] is one finite scalar; no epoch pseudoreplicates."""
    groups, paired = {}, {}
    for method, by_seed in observations.items():
        values = list(by_seed.values())
        if not values or not all(math.isfinite(float(x)) for x in values):
            raise ValueError("Statistics require nonempty finite seed-level observations")
        groups[method] = {"n": len(values), "mean": float(np.mean(values)),
                          "sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else None}
        if method == baseline:
            continue
        reference = observations.get(baseline, {})
        seeds = sorted(set(by_seed) & set(reference))
        differences = np.array([by_seed[s] - reference[s] for s in seeds], dtype=float)
        interval = None
        if len(seeds) >= 2:
            rng = np.random.default_rng(seed_for(0, "paired-bootstrap", method, baseline))
            means = differences[rng.integers(0, len(seeds), size=(10000, len(seeds)))].mean(1)
            interval = np.quantile(means, [.025, .975]).tolist()
        paired[method] = {"baseline": baseline, "seeds": seeds, "n": len(seeds),
                          "differences_by_seed": {str(s): float(x) for s, x in zip(seeds, differences)},
                          "mean_difference": float(differences.mean()) if len(seeds) else None,
                          "paired_bootstrap_95_percentile_interval": interval,
                          "unpaired_seeds": sorted(set(by_seed) ^ set(reference))}
    return {"statistics": groups, "paired": paired}
