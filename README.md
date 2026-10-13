# N:M x SAM — Implementation v1 (code and unit validation)

Fresh src-layout PyTorch research project with independent Git root history.
No old source, results or model weights are migrated.

Google's reference is **JAX/Flax**, at commit
`dae9904c4cf3a57a304f7b04cecffe371679c702`. This is an independent port of its
mathematics and update order, not an official PyTorch implementation or a
cross-framework numerical/accuracy reproduction.

Setup: `python -m pip install -e ".[test]"`.
Validation: `python -m pytest -q`.
Default deployment validation: `python scripts/run_experiment.py --config configs/gpu_baselines.yaml`.
All experiment CLIs are dry-run by default. Real execution requires --execute
and separate user authorization for GPU/resources. DATASET_PENDING and
GPU_VALIDATION_PENDING remain; code availability is not experimental validation.

SGD smoke: `python scripts/smoke_dense_cpu.py --config configs/dense_sgd_toy.yaml`.
2:4 primitive smoke: `python scripts/smoke_nm_cpu.py` (no sparse training).
Legacy smoke CLIs also require --execute. No data downloads are implemented.

See [status](docs/IMPLEMENTATION_STATUS.md), [parity](docs/01_OFFICIAL_SAM_PARITY.md),
[interfaces](docs/02_MATH_AND_INTERFACE_CONTRACT.md), and [roadmap](docs/05_RESEARCH_ROADMAP.md).
Current [architecture/APIs](docs/ARCHITECTURE_AND_API.md), [GPU deployment](docs/GPU_DEPLOYMENT.md),
[unit log and historical pre-policy runs](docs/EXPERIMENT_LOG.md).
Bootstrap-only historical evidence: [validation](docs/VALIDATION.md).
Only magnitude/Taylor production mask policies are implemented; future selection
proposals are design documents and must be reviewed before implementation.

