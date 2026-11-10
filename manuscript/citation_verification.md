# Citation verification

Metadata existence, BibTeX structure, and sentence support are separate checks. `citation_lock.json` is produced by the skill verifier, not hand-labelled from search results. The exact approved sentences and evidence locations are in `citation_requests.json`; wider comparative claims require their own support audit. Checked 2026-11-05.

| Key | Identifier | Primary source | Cross-check | Claim supported | Relation | Status | Checked at |
|---|---|---|---|---|---|---|---|
| sam | arxiv:2010.01412 | https://arxiv.org/html/2010.01412v3 | Official arXiv API metadata plus versioned paper | SAM uses a normalized current gradient to approximate the Euclidean perturbation inner step. | direct | content verified; metadata lock recorded separately | 2026-11-05 |
| asam | arxiv:2102.11600 | https://arxiv.org/html/2102.11600v3 | https://proceedings.mlr.press/v139/kwon21b.html | ASAM adapts the perturbation region to parameter scale through a normalization operator. | direct | content verified; metadata lock recorded separately | 2026-11-05 |
| ssam | arxiv:2210.05177 | https://arxiv.org/html/2210.05177v2 | Official arXiv API metadata plus versioned paper | SSAM masks the SAM perturbation using Fisher-based or dynamically updated sparse masks. | direct | content verified; metadata lock recorded separately | 2026-11-05 |
| preconditioned | arxiv:2501.06603 | https://arxiv.org/html/2501.06603v1 | Official arXiv API metadata plus versioned paper | The preSAM framework separates objective preconditioning from constraint preconditioning. | direct | content verified; metadata lock recorded separately | 2026-11-05 |
| gear | arxiv:2607.18306 | https://arxiv.org/html/2607.18306v1 | Official arXiv API metadata plus versioned paper | GEAR-SAM uses an EMA of squared block-gradient norms to allocate a fixed global perturbation budget. | direct | content verified; metadata lock recorded separately | 2026-11-05 |
| cram | arxiv:2207.14200 | https://arxiv.org/html/2207.14200v4 | Official arXiv API metadata plus versioned paper | CrAM incorporates a compression operator into training to prepare dense models for one-shot compression. | direct | content verified; metadata lock recorded separately | 2026-11-05 |
| adasap | arxiv:2306.14306 | https://arxiv.org/html/2306.14306v2 | Official arXiv API metadata plus versioned paper | AdaSAP adapts neuron perturbations before channel pruning and subsequently trains for robustness. | direct | content verified; metadata lock recorded separately | 2026-11-05 |
| obc | arxiv:2208.11580 | https://arxiv.org/html/2208.11580v2 | Official arXiv API metadata plus versioned paper | OBC uses ExactOBS for layerwise post-training pruning and extends the procedure to N:M sparsity. | direct | content verified; metadata lock recorded separately | 2026-11-05 |
| sparsegpt | arxiv:2301.00774 | https://arxiv.org/html/2301.00774v3 | https://proceedings.mlr.press/v202/frantar23a.html | SparseGPT performs approximate second-order layer reconstruction for one-shot pruning, including 2:4 and 4:8 patterns. | direct | content verified; metadata lock recorded separately | 2026-11-05 |
| rose | arxiv:2603.05878 | https://arxiv.org/html/2603.05878v1 | https://mingluo-su.github.io/ROSE-Webpage/ | ROSE changes column and block processing order within the SparseGPT pruning workflow. | direct | content verified; metadata lock recorded separately | 2026-11-05 |
| nm | arxiv:2102.04010 | https://arxiv.org/abs/2102.04010 | Official arXiv API metadata plus versioned paper | SR-STE trains N:M sparse networks from scratch using a sparse-refined straight-through estimator. | direct | content verified; metadata lock recorded separately | 2026-11-05 |

| wanda | arxiv:2306.11695 | https://arxiv.org/abs/2306.11695 | Official arXiv API metadata and abstract | Magnitude times input activations; no compensating weight update | direct | verified | 2026-11-05 |

## Scope and unresolved identity

`preconditioned` identifies Zhang, Li and Giannakis, *Preconditioned Sharpness-Aware Minimization: Unifying Analysis and a Novel Learning Algorithm* (arXiv:2501.06603). Its metadata and scoped sentence are verified. The bare user label “Preconditioned SAM” remains `needsverification` as an intended identity: another work, *Avoiding spurious sharpness minimization broadens applicability of SAM* (https://openreview.net/pdf?id=vYMuAcbdIE), also uses that label. Do not silently merge these works or claim a unique baseline implementation.

Preprint years intentionally differ from conference years for SAM, CrAM and AdaSAP. The API supports these preprint records; ASAM and SparseGPT additionally cross-check against official PMLR proceedings. No conference acceptance is inferred solely from a year or title.

The comparison matrix distinguishes absent method components from unverified properties. N/A is not evidence of impossibility; “not established” is not a negative result. This audit does not establish exhaustive novelty.

## Automated validation record

Official `verify_citations.py` resolved eight records initially; OBC, ROSE and SR-STE encountered transient transport failures. `literature/retry_citation_metadata.py` retried those three through the same official arXiv resolver and succeeded. All 11 current lock records are verified with fresh provider timestamps and metadata digests.

`check_citations.py literature/citation_audit.tex references.bib` passed. `check_citation_lock.py . --main literature/citation_audit.tex` passed. This covers the bibliography-wide audit input, not an unfinished manuscript. Recheck against the final main TeX and keep sentence-support records synchronized with any newly added or expanded claims.
