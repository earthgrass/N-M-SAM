# Claim-to-evidence map

| Claim | Evidence | Status / boundary |
|---|---|---|
| Q is a damped reconstruction inverse, not CE Hessian | math_definition.md; math_core.py; Method Eq.(2) | Defined; no full-network loss curvature claim |
| Singleton deletion cost w²/(2Qii) | Proposition 1 and proof; OBC | Established existing OBS identity, not new contribution |
| Global coordinate-weighted centering | config + math_core.py; Eq.(3) | Output-row multiplicity required |
| Log-EMA is a relative geometric history | Proposition 2 + full proof | Algebraic fact, not loss-optimal update |
| Fixed global physical radius | Proposition 3; GPU gate + smoke norm records | Nonzero direction; all trainable coordinates |
| HC geometry is independent of APBR feedback | actual state update in math_core.py | No energy feedback in recurrence |
| State shared across rows | broadcast Q diagonal + zero initial state | Feature-wise geometry, not independent per-weight Q |
| Native final method unchanged | evaluation.py prepare_layer_endpoints -> sparsegpt_matrix | Dynamic selection + sequential OBS |
| Auxiliary static mask differs | evaluation.py; endpoint identities | Never label fixed-mask endpoint native GPT |
| Same baseline initialization | evidence/initialization.json | Tensor identity, not equivalent trajectories |
| GPU numerical feasibility | gpu_smoke_summary.json math_gate/resume_gate/numerics | PASS; not final accuracy evidence |
| Performance overhead 30.55% | smoke_history.json + recomputation script | FAIL at original 30%; research override preserved |
| HC improves sparse accuracy | No final evaluation yet | PENDING, no positive claim |
| History improves over Direct-Q | No Direct-Q result | PENDING |
| Energy support alignment improves | No coordinatewise energies | PENDING / instrumentation missing |
| Specific signal differs from related work | primary-source literature matrix | Narrow novelty hypothesis; not proof no prior art exists |

Numeric tables are generated from snapshots; selectors in numeric_evidence.json make individual numbers traceable. Recompute averages with scripts/prepare_evidence.py. Do not reuse old APBR accuracy as an HC result.
