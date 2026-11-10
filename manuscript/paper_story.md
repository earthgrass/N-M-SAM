# HC-SAM story boundary

Question: can historical conditional layer-reconstruction compensability guide a fixed SAM perturbation budget, improving compatibility with unchanged native SparseGPT?

Thesis to test, not an observed conclusion: the optimizer can expose the dense trajectory to input-feature directions that are conditionally replaceable under reconstruction, without introducing a sparse support or a new final compensation algorithm into training.

Narrative: structured deletion differs from gradient perturbation -> training and reconstruction use different geometry -> estimate inverse-Gram diagonal -> center and smooth its log scale -> redistribute a fixed physical radius -> evaluate unchanged native SparseGPT.

Established: singleton quadratic identity; log-history product; centering; fixed radius; partial budget derivatives; CPU/GPU and resume validation. Existing training logs cannot establish final-mask coordinate alignment.

Not established: better final accuracy; history superiority over Direct-Q; SOTA; stable multi-seed gains; generalization theorem; end-to-end efficiency advantage; first preconditioned/historical SAM.

Closest-work boundary: preSAM covers the perturbation form; GEAR-SAM covers historical allocation and a fixed radius; CrAM/AdaSAP cover compression-aware sharpness training; OBC/SparseGPT supply reconstruction. Our proposed signal and downstream compatibility experiment must earn their value empirically.

Author revision: main method-level comparison is SparseGPT / ROSE / Wanda / HC-SAM, accuracy only; internal SAM/APBR/Direct-Q controls are possible mechanism studies, not replacements for the requested main table. Experiments is temporarily blank. Protocol matching must be frozen before importing any such table.
