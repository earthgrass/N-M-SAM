# SAM dynamic Group-OBS control on RTX 5090

This is an exploratory Seed 0 comparison with the same frozen 1,024 CIFAR-100 Train calibration indices, 19 eligible convolutional layers, 2:4, 1% relative damping, blocksize 128, zero recovery epochs, and a fixed one-time Final Test per new checkpoint. The official 50k/10k archive, parent checkpoint hashes, 19-layer mask contract, and source/config identities were checked. No Dense training was repeated.

## Same-GPU task performance

| Dense parent | Post-training selector and compensation | Test accuracy | Test CE |
| --- | --- | ---: | ---: |
| Dense-SAM | Native SparseGPT (historical fixed endpoint) | 77.47% | 0.85026 |
| Dense-SAM | Dynamic Group-OBS | **77.93%** | **0.82836** |
| APBR-L2 | Dynamic Group-OBS | 77.62% | 0.83630 |
| APBR-L2 | Geo-Select + standard group OBS | 77.73% | 0.83548 |
| APBR-L2 | Geo-Full | 77.68% | 0.83587 |
| Original APBR-VPR | Dynamic Group-OBS | 77.85% | 0.82978 |
| Original APBR-VPR | Geo-Select + standard group OBS | 77.80% | 0.82819 |
| Original APBR-VPR | Geo-Full | 77.83% | 0.82676 |

All **new** task metrics in this table were run on the RTX 5090. The native SparseGPT task metric is the previously audited Seed 0 result from RTX 4080 SUPER; a same-GPU native **cost-only** run was performed without Test access. Therefore no new native accuracy is claimed here.

On the same APBR-L2 parent, Geo-Select exceeds unscaled dynamic Group-OBS by **0.11 percentage points**. On the original APBR-VPR parent, Geo-Select is **0.05 points lower** and Geo-Full is **0.02 points lower** than unscaled dynamic Group-OBS. The best new Seed 0 accuracy is SAM + dynamic Group-OBS at 77.93%. These differences are small, the dense parents differ, and one seed is insufficient for a stability claim. The current evidence does **not** establish that APBR geometry improves one-shot accuracy under this six-pair operator.

## Same-GPU cost comparison

The native SparseGPT cost-only run and the SAM dynamic Group-OBS run used the same parent, GPU, data and adapter settings. GPU synchronization brackets each layer. The quoted sums cover the 19 pruning/compensation calls and exclude calibration, checkpoint writing and Test.

| SAM parent method | Calibration | 19-layer pruning + compensation | Relative to native |
| --- | ---: | ---: | ---: |
| Native SparseGPT, FP32, dynamic per-coordinate selector and sequential OBS | 0.697 s | **4.333 s** | 1.00× |
| Dynamic Group-OBS, FP64 conditional group factor and six-pair selection | 0.725 s | **5.846 s** | 1.35× |

The observed Group-OBS implementation takes **1.513 s more**, approximately **34.9%** more than native SparseGPT on this GPU. Precision differs as specified by the respective implementations, so the difference cannot be attributed solely to six-pair scoring. APBR-L2 Geo-Select took 5.566 s and original APBR-VPR Geo-Select took 5.624 s for the same 19-layer phase, but those have different dense parents and are not the controlled SAM timing pair.

## Mechanism boundary

Native SparseGPT chooses two coordinates per quartet from individual per-coordinate scores at group entry, then compensates sequentially column by column. Dynamic Group-OBS enumerates six deletion pairs and applies a joint four-column update. Geo-Select changes the six-pair selection signal from \(w\) to \(v=w\exp(-s)\), while compensating physical \(w\) with the group operator. Geo-Full selects and compensates in \(v\) coordinates. Thus the current Geo-Select result **does not** test the distinct idea of replacing only the per-coordinate score in original SparseGPT. That idea is left for separate mathematical specification; no such new run was started.

## Evidence

The authoritative server run root is `/root/autodl-tmp/sam_dynamic_group_20261102/artifacts/`. The SAM dynamic output, APBR-L2 and VPR same-GPU outputs, their logs, and native GPT cost-only output are transferred to `experiments/post_training_nm_benchmark/artifacts/sam_geo_dynamic_rtx5090_20261102_remote_copy/`. Each method's new result is linked to a frozen checkpoint hash and a one-time fixed Test record. Source and numerical gate code are kept in the independent `geo_dynamic_obs` adapter and experiment scripts.

Local transfer verification on 2026-11-03: all SHA256 manifest entries matched the downloaded files: 111 APBR-L2, 111 original APBR-VPR, 27 SAM dynamic Group-OBS, and 3 native SparseGPT timing files (252 entries total). An initial Windows long-path transfer omitted six VPR intermediate shortcut files; those files were copied through a short mapped path and the full manifest was rechecked with zero missing or mismatched entries. The source result directories on the server were not deleted.

The separate Radius-Guided SparseGPT adapter already tests a related per-coordinate modification of the *native* SparseGPT selector: it multiplies the squared native score by `exp(-s)` while retaining native sequential OBS. Replacing the weight with `v=w exp(-s)` in that squared score would instead multiply by `exp(-2s)`; that exact proposal remains untested. In the historical original APBR-VPR Seed 0 evaluation, the `exp(-s)` variant reached 76.86% accuracy, compared with 76.99% for original GPT on the same APBR parent. Neither result is the six-pair Geo-Select experiment above.
