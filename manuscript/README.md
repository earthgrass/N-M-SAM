# HC-SAM independent English research draft

Open **main_standard.pdf** for the requested regular reading format (A4 single column, Times New Roman, Roman main headings and A/B/C subsections). The **main.pdf** CVPR edition is preserved. Edit **sections/** for shared content; main_standard.tex and main.tex control formatting only. This paper is independent of the original APBR paper and does not alter HQ training.

## Build

Run `python scripts/build.py` from this directory (MiKTeX/pdflatex and bibtex required). For the standard edition run `python scripts/build_standard.py` (XeLaTeX). Official CVPR 2027 style is pinned in `template/provenance.json`.

## Evidence

- `evidence/`: immutable source snapshots, config and source hashes, smoke and historical measurements.
- `literature/comparison_matrix.md`: closest-work comparison; `references.bib` and citation lock.
- `claim_evidence_map.md`: established facts versus pending claims.
- `OPEN_ISSUES.md`: scientific and instrumentation gaps.
- `review/`: compiler, numerical checks, review and layout records.

Final results are PENDING. Historical SAM and APBR numbers are references only. Performance gate remains FAIL with an explicit research override. Draft completion is not experimental completion or submission readiness.

For a complete downloaded HC run, `python scripts/import_final_results.py --run-dir PATH` validates without modifying the draft. Adding `--write` imports a single version after all hashes pass; it never runs Test. Review prose and regenerate numeric evidence/build afterwards.

The combined PDF contains main text, references, and proof/protocol appendix. Before submission separate supplementary pages and complete venue/anonymity checks. Private audit metadata is not an anonymous submission bundle.

Latest author scope: Experiments is intentionally empty in both PDFs. Future main table compares SparseGPT, ROSE, Wanda and HC-SAM with accuracy only (tables/accuracy_only_planned.tex, currently not included). Detailed previous measurements remain audit artifacts, not manuscript results.
