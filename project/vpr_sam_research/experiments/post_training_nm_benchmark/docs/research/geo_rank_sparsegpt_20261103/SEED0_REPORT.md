# APBR-normalized per-coordinate SparseGPT, Seed 0

## Question and implementation

Does the geometry learned by APBR improve the native SparseGPT 2:4 mask when only the selection score changes? The new isolated adapter computes the existing native score on the **current compensated physical weight** and multiplies it by `exp(-2s)`, equivalent to scoring `v=w exp(-s)` with the native inverse factor. It selects the lowest two of four per row, retains native stable ties, and runs the unchanged physical-weight sequential OBS propagation. There is no six-pair enumeration, groupwise compensation, training, BN-only pass, or Recovery.

The controls are native SparseGPT and a within-quartet shuffled geometry score, each beginning from the identical Dense Final and calibration data. APBR-L2 and original APBR-VPR are separate parent cohorts. The APBR-L2 parent SHA256 is `d68fbffc5f862be07656c5b45b8dcde2a5411865c0e5e03f2f45c2f97926b742`; APBR-VPR is `4332079becdb58fb50f0e68b8bd0dcc251db0184a6050facb524817520a51d82`. The official CIFAR-100 archive SHA256 is `85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7`.

Both runs used the existing 1,024 fixed Train calibration samples, the same Conv2d Hessian, 1% relative damping, 128-column blocks, CUDA FP32, 19 eligible layers, and exact 2:4. All three checkpoints and masks within each cohort were frozen before the fixed 10,000-image Test was opened. Each branch had one Test evaluation. A remote RTX 5090 with PyTorch 2.8.0+cu128 was used. The original and guided adapters share the same columnwise OBS arithmetic; the guided wrapper fixes the prior radius-guided implementation's exponent at 2 rather than the historical exponent 1.

## Fixed Final Test

| Dense parent | Selector + compensation | Accuracy | CE | Difference from same-parent native |
| --- | --- | ---: | ---: | ---: |
| APBR-L2 | Native SparseGPT | 77.19% | 0.865715 | reference |
| APBR-L2 | Geometry `exp(-2s)` + native OBS | 77.15% | 0.863562 | −0.04 pp |
| APBR-L2 | Shuffled geometry + native OBS | 77.05% | 0.861776 | −0.14 pp |
| Original APBR-VPR | Native SparseGPT | 77.06% | 0.858268 | reference |
| Original APBR-VPR | Geometry `exp(-2s)` + native OBS | 76.76% | 0.857968 | −0.30 pp |
| Original APBR-VPR | Shuffled geometry + native OBS | 77.15% | 0.856716 | +0.09 pp |

The geometric score changed 2.383% of eligible keep/delete positions on APBR-L2 and 2.404% on APBR-VPR relative to native. Those are position flip rates, not whole-group flip rates. The accuracy signal is inconsistent: a slight loss on L2 and a larger loss on original VPR; shuffled geometry beats true geometry on original VPR. CE changes are small and also do not establish a coherent advantage. One seed does not support a positive method claim or a causal explanation for the differences.

The external complete-pipeline reference is the historically audited **SAM Dense + native SparseGPT**, 77.47% / 0.850260 CE on Seed 0. APBR-L2 + `exp(-2s)` native GPT is 0.32 percentage points below it; original APBR-VPR + `exp(-2s)` native GPT is 0.71 points below it. This cross-parent comparison changes Dense training and uses a historical Test run on another GPU; it is not an isolated estimate of geometry's effect. The same-parent controls in the main table provide that isolation. Earlier APBR + six-pair Group-OBS combinations must not be used as evidence that the present per-coordinate selector improves native GPT: the operator also changes both group selection and compensation. The strongest new same-GPU Group-OBS Seed 0 result was instead **SAM + Dynamic Group-OBS, 77.93%**.

## Cost and numerical checks

| Parent | Native 19-layer pruning + compensation | Geometry score | Shuffled geometry |
| --- | ---: | ---: | ---: |
| APBR-L2 | 4.184 s | 4.892 s | 4.903 s |
| Original APBR-VPR | 4.191 s | 4.928 s | 4.919 s |

These synchronized wall times exclude calibration, checkpoint writing and Test. The guided branch includes diagnostic score tensors, so this is an implementation cost comparison, not an isolated estimate of the multiplication itself. Calibration took 0.676 s and 0.664 s, respectively. Peak CUDA allocated was 1,075,517,952 bytes; reserved was 1,415,577,600 bytes in each cohort.

Nine targeted CPU numerical tests passed locally and on the remote Python environment, including exact native replay for zero/constant geometry, exact `exp(-2s)` score, strict bool 2:4, finite outputs, and a case where geometry changes the selected two positions. `pytest` was unavailable remotely, so the test functions were invoked directly there without installing dependencies; local pytest passed 9/9. Both remote result manifests verified with `sha256sum -c` (69 entries each). No failure file was created. Full checkpoint/mask and per-layer hashes are in each output directory.

The full isolated outputs and logs were copied to `experiments/post_training_nm_benchmark/artifacts/geo_rank_sparsegpt_seed0_20261103/remote_copy/`. All 138 manifest entries were checked again locally with zero missing or mismatched files. The remote and local manifest hashes agree: APBR-L2 `51d2f9ecfd83d1445a6241c725fc363a07db162d4402df2ce6575d340a756071`; APBR-VPR `df6e93d6c3bd5f1d7aef170fa344711fd14efa48ef716054636e24a8ed6fb1ad`. No GPU compute process remained after completion.

## Research judgment

This pilot does **not** support replacing the native SparseGPT selector with APBR-normalized per-coordinate weights. It establishes that the implementation changes the mask while preserving the native OBS path and that the new scoring exponent alone does not reliably improve Seed 0 task accuracy. The prior `exp(-s)` Radius-Guided experiment and the present `exp(-2s)` experiment are distinct; neither should be presented as evidence that APBR geometry improves native SparseGPT. Dynamic Group-OBS and six-pair Geo-Select are separate mechanisms and their results must not be pooled with this per-coordinate variant.
