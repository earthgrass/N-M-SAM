# Current implementation status — 2026-10-13

Current scope: CODE_COMPLETE / UNIT_TESTED; DATASET_PENDING; GPU_VALIDATION_PENDING.
No research performance, accuracy or novelty claim follows from unit tests.

| Stage | Implemented scope | Current status |
| --- | --- | --- |
| P0 | anchored ignores, tracked data package, independent-clone source test | CODE_COMPLETE / UNIT_TESTED |
| M2 | offline TAR safety/hash/parser/transforms/split, CIFAR ResNet18, master/PureMasked/STE, fixed sparse SAM, dynamic masks, six baselines | CODE_COMPLETE / UNIT_TESTED; real deployment DATASET_PENDING |
| M3 | pure effective functional CE, true gradients, fixed BN/RNG, candidate q, HVP/full small-toy Hessian, atomic full-state checkpoint and rollback, sampler | CODE_COMPLETE / UNIT_TESTED |
| M4 | task gradient path, h_obs/Hu, q/u/a/r, residual curvature, six candidates, ranking/errors/cost | CODE_COMPLETE / UNIT_TESTED; route A GPU_VALIDATION_PENDING |
| M5 | supported single/multi-step network proxies, quadratic robust proxy, small FP64 global quadratic reference including hard case | CODE_COMPLETE / UNIT_TESTED; route B GPU_VALIDATION_PENDING |
| M6 | legal two-block combinations, cross coupling, quadratic/Steihaug/robust compensation, matched-backward recovery controls | CODE_COMPLETE / UNIT_TESTED; route C GPU_VALIDATION_PENDING |
| M7 | default dry-run GPU entry, 18-run baseline matrix, A/B/C and ablation configs, cosine schedule/eval, checkpoint/resume, compressed logs, CSV/JSON/report export | CODE_COMPLETE / UNIT_TESTED; GPU_VALIDATION_PENDING |
| CUDA single device | FP32/FP64 execution path | CODE_COMPLETE; GPU_VALIDATION_PENDING |
| AMP / DDP | capability reporting, compatibility design and fail-closed activation guards | UNIT_TESTED; actual training backends NOT_IMPLEMENTED |
| Innovative Mask Selection | none installed; only Magnitude/Taylor baseline policies | DESIGN_ONLY pending review |

Latest full local unit suite: 111 passed, 0 failed, 0 skipped. Python 3.12.10,
torch 2.13.0+cpu, torchvision 0.28.0+cpu, pytest 9.1.1. compileall and diff check clean.
No user CIFAR file, FakeData loop, ResNet training or A/B/C study runs after the
user's code-only override. Tests use small fixed tensors/models, analytic quadratics
and a self-generated ten-image archive fixture. GPU/dataset-dependent tests are
not claimed passing. All experiment CLIs require explicit --execute.

Dense kernel src/nmsam/sam remains byte-identical to rebuild-v1@cd33d2e4.
Runtime data sources are tracked; no TAR/model/checkpoint enters Git.
See ARCHITECTURE_AND_API, GPU_DEPLOYMENT, RESULTS_LIMITATIONS and EXPERIMENT_LOG.
Historical Bootstrap/run notes are retained in VALIDATION and EXPERIMENT_LOG only.
