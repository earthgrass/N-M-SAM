# CIFAR-100 Seed 0: three dense parents with Original SparseGPT

All accuracies and CE values below come from fixed, once-only 10,000-image Final Test records. GPT entries are native dynamically selected 2:4 masks with OBS compensation, before any Recovery-20. APBR-VPR and APBR-L2 are different dense training methods; GPT does not read their saved geometry state.

| Dense training parent | Dense accuracy | Dense CE | Original SparseGPT accuracy | GPT CE | GPT minus dense |
|---|---:|---:|---:|---:|---:|
| Dense-SAM | 78.84% | 0.79989 | 77.47% | 0.85026 | -1.37 pp |
| APBR-VPR | 78.68% | 0.80311 | 76.99% | 0.85810 | -1.69 pp |
| APBR-L2 | 78.91% | 0.80720 | 77.12% | 0.86602 | -1.79 pp |

The new APBR-L2 experiment completed 200 epochs and 78,200 steps with 78,200 geometry updates; its first perturbation norm was checked at 0.05. The native GPT endpoint used 1,024 fixed training images for calibration, all 19 eligible convolution layers, 11,157,504 eligible coordinates, and exactly 5,578,752 deletions. It used no recovery or BN-only calibration. Its checkpoint SHA256 is `efc18374262af5b752fdf03b27dac65256a75daefaf232e0b4a804576c9c8ada` and its dense parent SHA256 is `d68fbffc5f862be07656c5b45b8dcde2a5411865c0e5e03f2f45c2f97926b742`.

On the same APBR-L2 parent, Geometry Raw Hard was 56.32% (CE 1.71381), Magnitude Raw Hard was 55.54% (CE 1.74743), Geometry Recovery-20 was 79.06% (CE 0.81916), and Magnitude Recovery-20 was 79.07% (CE 0.81933). These recovery endpoints are separate from the no-recovery GPT comparison.

The APBR-L2 + GPT result is 0.13 pp above APBR-VPR + GPT and 0.35 pp below Dense-SAM + GPT in this single seed. This is descriptive only; the runs used different GPUs and one seed cannot establish a stable method difference. No Geo-OBS run was launched.

Evidence:

- New run: `artifacts/run_v1/queue_status.json`, `artifacts/run_v1/evaluations/*.json`, and `local_verification.json` in this directory. The transferred archive SHA256 is `2d6d79ce27fc7169a88c022a8f298bb183948ac543bbaee425562aaf48e48fb3`.
- Historical Dense-SAM and APBR-VPR dense/GPT: `../../cifar100_50k10k_final_v1/seed0_remote_copy/test_final/{dense_final,apbr_final,sparsegpt_raw}.json`.
- APBR-VPR + original GPT: `../../apbr_radius_guided_sparsegpt_seed012/evidence/original_controls/seed0/{status,test_final}.json`.
