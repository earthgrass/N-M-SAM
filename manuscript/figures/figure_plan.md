# Figure plan and data interfaces

1. Overview: exact four-box TeX schematic in introduction.tex. Deterministic text/math layout was chosen instead of generated raster art so epoch causality, formulas, and labels remain editable and unambiguous. It depicts the algorithm only.
2. Evolution: explicit TODO in experiments.tex. Future source: completed formal training/history.json, not three-epoch smoke. Required: epoch, geometry used/saved hash, drift, Q statistics, energy distributions. Plot actual logged values; no interpolation masquerading as measurement.
3. Alignment: explicit TODO. Current logs insufficient. Future coordinatewise accumulators and final native mask required; define pooled versus mean-step estimator before collection. Random-mask expectation is analytical, not an empirical curve.

Result importer: scripts/import_final_results.py reads local completed run with actual checkpoints and freeze/evaluation JSONs. It verifies hashes, protocol, sample counts, report consistency; no dataset loading or new evaluation. It generates only HC row and import evidence. Update prose and numeric registry and rebuild after import; no automatic positive conclusions.

Author revision: only the conceptual overview remains in the manuscript. Evolution/alignment figure structures and prior experimental tables are archived in review/previous_experiments_audit_only.tex. No empirical figure is currently displayed.
