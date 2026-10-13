# N:M x SAM — Bootstrap v0.1

Fresh src-layout PyTorch research project with independent Git root history.
No old source, results or model weights are migrated.

Google's reference is **JAX/Flax**, at commit
`dae9904c4cf3a57a304f7b04cecffe371679c702`. This is an independent port of its
mathematics and update order, not an official PyTorch implementation or a
cross-framework numerical/accuracy reproduction.

Setup: `python -m pip install -e ".[test]"`.
Validation: `python -m pytest -q`.
CPU toy entry point (M1): `python scripts/smoke_dense_cpu.py --config configs/dense_sam_official_toy.yaml`.
No dataset downloads or long training are authorized by these commands.

See [status](docs/IMPLEMENTATION_STATUS.md), [parity](docs/01_OFFICIAL_SAM_PARITY.md),
[interfaces](docs/02_MATH_AND_INTERFACE_CONTRACT.md), and [roadmap](docs/05_RESEARCH_ROADMAP.md).

