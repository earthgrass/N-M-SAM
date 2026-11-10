# Layout inspection

Both editions were compiled with actual TeX engines and inspected as rendered PDFs.

- CVPR edition: official 2027 review style; 6 pages through conclusion, 9 pages including references and appendix. No padding added to reach the approximate eight-page target. The combined PDF is not a submission upload package.
- Standard edition: A4, single column, 10pt Times New Roman, 23mm margins; Roman main sections and A/B/C subsections; 10 pages including appendix and references. Its reference bibliography uses first-citation ordering like the provided APBR draft.
- Reference PDF font inspection confirmed Times New Roman regular/bold/italic with Computer Modern math. The standard edition follows that setup.
- CVPR overview initially wrapped its fourth box; reduced box widths and recompiled. Final overview is one readable four-box row with explicit epoch causality.
- Tables 1–4 now follow requested main result / ablation / diagnostic / performance order. Table5 is historical execution detail.
- Inspected full CVPR contact sheet and enlarged overview; standard title, formula/method and result-table pages inspected. No clipped tables, overlapping boxes, overfull boxes, unresolved references or undefined citations in final compiler logs.
- Figures2/3 visibly say TODO. No synthetic data curve is presented. Unavailable HC metrics say PENDING.

Machine page counts, warning counts and PDF hashes: layout_metrics.json. Source-format templates and result bodies are separate; sections/ is shared by both editions.

Final author revision: expanded explanatory method prose; Experiments intentionally blank. All numerical tables and TODO empirical figures removed from the manuscript, retained as audit/planning assets. Final table numbering is inapplicable until experiments are written. Both final logs have zero overfull boxes and zero undefined references/citations.
