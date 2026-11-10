# Unified APBR-GPT: Shared-Q Seed 0 Screen

## Outcome

The authorized Seed 0 full-network Shared-Q screen completed all 33 registered endpoints. It used the frozen APBR-SAM Epoch-200 parent, the frozen Original GPT mask where applicable, 19 eligible convolutional layers, strict 2:4 sparsity, and the registered lambda grid. No Test sample, Recovery, BN-only pass, gradient update, Dense retraining, or mask tuning was used.

The main Shared APBR-Q gate is negative. For the fixed Original GPT mask, every positive lambda had equal or lower train-side diagnostic accuracy and higher CE than Isotropic Ridge. APBR-Q also produced lower layer reconstruction energy than Isotropic Ridge in only 3--6 of 19 layers, depending on lambda. This is not stable mechanism evidence.

| lambda | Shared APBR-Q Acc. | Isotropic Acc. | APBR-Q CE | Isotropic CE | CE delta (APBR - Iso) |
|---:|---:|---:|---:|---:|---:|
| 1e-4 | 99.9023% | 99.9023% | 0.042903 | 0.042902 | +0.000002 |
| 1e-3 | 99.9023% | 99.9023% | 0.043235 | 0.043231 | +0.000004 |
| 1e-2 | 99.9023% | 99.9023% | 0.046539 | 0.046520 | +0.000018 |
| 1e-1 | 99.7070% | 99.9023% | 0.081117 | 0.080771 | +0.000347 |
| 1 | 92.9688% | 93.0664% | 0.505977 | 0.500965 | +0.005011 |

The diagnostic set is a fixed 1,024-image subset disjoint from the 1,024 calibration images but drawn from the official training set. Its near-saturated accuracy limits task-level discrimination; CE and the paired layer diagnostics therefore matter, but they do not justify a Final-Test claim.

## Unified cross-ablation

`HH` (Original H selection + H compensation) reached 99.9023% / 0.042820 on the train-side diagnostic set. `QH` reached the same accuracy and slightly lower CE at lambda 1e-2 (0.042331) and 1e-1 (0.042363). This is a small exploratory signal, not a passed gate: it is measured on a saturated training-side diagnostic set and lacks a matched isotropic-selection control. `HQ` and `QQ` did not show a consistent advantage.

All methods were reconstructed independently from the same Dense parent. Cross-ablation compensation never inherited weights from a preceding endpoint.

## Geometry diagnostics

Output-row aggregation preserved input-coordinate order and did not average input coordinates or renormalize 2:4 blocks. Across 19 layers, q_shared had mean CV 0.00915 (range 0.00073--0.03919) and mean relative L2 distance from an isotropic vector of 0.01729. Thus q_shared is not exactly constant, but is close to isotropic in most layers. Its mean Spearman correlation with the Hessian diagonal was 0.695 (layer range 0.429--0.935), indicating substantial overlap with curvature already represented by H.

Exact-to-Shared aggregation relative RMS error averaged 0.0150 (range 0.00135--0.08684). Shared-Q is therefore a distinct approximation and is not reported as Exact Row-wise Q.

## Cost and decision

Per-endpoint pruning/compensation time was 3.31--4.63 s for most H/HQ/QQ endpoints and 7.54--7.83 s for Q-selection followed by an independent H fixed-mask compensation. Peak CUDA allocated/reserved memory was 1.316/1.484 GiB. The complete 33-endpoint pruning screen accumulated 143.06 s of pruning time, excluding evaluation and artifact transfer.

Following the registered gate, Seed 1/2 expansion is paused. No lambda was selected using Final Test, and no APBR-GPT Final Test was run. The negative Shared-Q result and the limited QH diagnostic signal are both retained.
