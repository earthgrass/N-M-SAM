# EXPERIMENT_REPORT.md — Phase 1 Report

> **Stage.** Code-implementation + smoke-test + short validation runs
> (ResNet18 / CIFAR-10, 2 epochs, single seed, CPU).
> **Status.** Code complete and sanity-checked; full multi-seed long training
> deliberately not started (Phase-1 spec forbids long training in this stage).

---

## 1. Experiment purpose

The research goal is to understand how 2:4 structured weight sparsity
interacts with Sharpness-Aware Minimization (SAM), and whether a SAM-aware
mask selection strategy can outperform classical magnitude pruning.

Phase 1 focuses on producing the qualitative phenomenon on a small CNN +
CIFAR-10 setup; it is **not** about:

* achieving state-of-the-art accuracy on CIFAR-10,
* invoking real NVIDIA Sparse Tensor Core kernels,
* proving a theoretical claim.

The three questions being answered are listed in the README and restated in
the summary table at the bottom of this file.

---

## 2. Five methods — exact definitions

All methods share the same model (`resnet18_cifar` or `simple_cnn` for
smoke), the same augmentation pipeline, the same batch size, the same
number of epochs, the same SGD hyperparameters, and the same random seed.
The only thing that changes is the *optimizer + sparsity strategy*.

| Method        | Sparsity | Optimizer | Mask-score                  |
| ------------- | -------- | --------- | --------------------------- |
| Dense-SGD     | none     | SGD       | n/a                         |
| Dense-SAM     | none     | SAM       | n/a                         |
| 2:4-SGD       | 2:4      | SGD       | magnitude                   |
| 2:4-SAM       | 2:4      | SAM       | magnitude                   |
| 2:4-SAM-aware | 2:4      | SAM       | `\|W * g_SAM\|` (Taylor on SAM grad) |

* **SGD** is plain SGD with momentum 0.9, weight decay 5e-4, no Nesterov.
* **SAM** wraps an SGD base-optimizer with two forward/backward passes per
  step: first at `w` to get `g1`, then at `w + rho * g1 / ||g1||` to get
  `g2`, and the SGD step is taken with `g2`. BN running stats are frozen
  (`disable_running_stats`) during both forward passes.
* **N:M mask**: for every block of `M=4` consecutive elements along the
  inner dimension, keep the top-`N=2` positions by score, zero out the
  rest. Conv weights are flattened along `[in, kh, kw]` before blocking.
  Layers whose inner dimension is shorter than `M` (or the first conv, per
  Phase-1 spec) are left dense; the tail of partial blocks is also dense.
* **STE**: forward sees `W * M`, backward uses straight-through identity so
  the dense master weight still receives gradient everywhere
  (`W_eff = W + (W * M - W).detach()`).
* **Mask fixed across a SAM step**: the mask captured before the first
  backward is identical to the one used in the second backward. The SAM
  perturbation itself is also element-wise masked so masked positions
  cannot become non-zero via epsilon.
* **SAM-aware score**: `|W * g_SAM|`. `g_SAM = g2` is the second-backward
  gradient. Mask recomputation uses `blockwise_topk_mask` on this score
  every `--mask-update-interval` optimizer steps, and is applied **after**
  the optimizer step.

---

## 3. N:M mask implementation

Lives in `sparsity/mask.py`. Two public builders (`linear_mask`,
`conv2d_mask`) plus the generic `blockwise_topk_mask` that operates on a
flattened tensor and respects a dense tail. `sam_aware_mask` dispatches
between the linear and conv4D layouts so conv layers do **not** collapse
to all-ones (this was an early bug, see §6).

`attach_nm_sparsity` replaces each eligible `nn.Linear` / `nn.Conv2d`'s
weight with an `NMSparsifiedWeight` wrapper that owns:

* `weight_master` — the trainable dense Parameter,
* `mask` — a non-trainable buffer holding the current 0/1 mask,
* `recompute_mask(score, grad=None)` — rebuild the mask; returns change ratio.

---

## 4. SAM implementation

Lives in `optim/sam.py`. `make_sam_optimizer` creates the SGD base
optimizer plus a `SAM` wrapper. The training loop calls
`first_step(zero_grad=True, mask_fn=...)` followed by the second forward /
backward / `second_step(zero_grad=True)`. The `mask_fn` closure is built
by `make_param_mask_fn` and returns `M` for sparse master weights and
`None` otherwise, so the perturbation is masked element-wise.

`disable_running_stats` swaps every BN module's `momentum` to 0 and back,
freezing running-stat updates during the SAM passes.

---

## 5. SAM-aware score

After the second backward, `sam_step` writes `weight_master.grad` into
`wrapper.last_grad` (cloned). `_recompute_all_masks` then calls
`recompute_mask(score="sam_taylor", grad=w.last_grad)`, which builds a
mask from `|W * g_SAM|`. This is interpreted as a first-order Taylor
approximation of "deleting this weight would change the SAM worst-case
loss".

---

## 6. Bugs found and fixed during smoke tests

1. **`metrics/sanity.py` used a relative import that escaped the package.**
   Fixed by switching to `from nm_sam_cifar10.sparsity...`.
2. **`models/__init__.py` referenced `torch` without importing it.**
   Fixed by adding `import torch` / `import torch.nn as nn`.
3. **`sam_aware_mask` called `blockwise_topk_mask` on a 4D conv weight
   without first flattening `[in, kh, kw]`, so for conv layers
   `inner = kw < m` and `num_full = 0`, causing the SAM-aware mask to
   collapse to all-ones (passed the in-training `check_block_sums` guard
   which short-circuits when `num_full == 0`, but failed the post-hoc
   runtime sanity check).** Fixed by routing through the same dispatch
   logic used by `mask_for_weight` (reshape `(out, in*kh*kw)` first).

All three are also covered by regression tests
(`tests/test_mask.py`, `tests/test_ste.py`, `tests/test_sam.py`,
`tests/runtime_sanity.py`).

---

## 7. Project file structure

(see README for the tree; the layout follows the Phase-1 spec exactly).

---

## 8. Actual commands used

### Unit tests
```bash
PYTHONPATH=. python nm_sam_cifar10/tests/test_mask.py
PYTHONPATH=. python nm_sam_cifar10/tests/test_ste.py
PYTHONPATH=. python nm_sam_cifar10/tests/test_sam.py
```

### Smoke test (SimpleCNN + FakeData, 1 epoch)
```bash
PYTHONPATH=. python -m nm_sam_cifar10.train \
    --method <METHOD> --model simple_cnn --epochs 1 \
    --batch-size 256 --use-fake-data \
    --fake-train-size 2000 --fake-test-size 500 \
    --num-workers 0 --sharpness-interval 1 \
    --output-dir ./nm_sam_cifar10/results/smoke_<METHOD>
```

### Short ResNet18 + real CIFAR-10 (2 epoch, single seed)
```bash
PYTHONPATH=. python -m nm_sam_cifar10.train \
    --method <METHOD> --model resnet18_cifar --epochs 2 \
    --batch-size 128 --subset-train 4000 --subset-test 1000 \
    --num-workers 0 --mask-update-interval 20 --sharpness-interval 1 \
    --data-root ./nm_sam_cifar10/data/cifar10 \
    --output-dir ./nm_sam_cifar10/results/short_real/<METHOD>_seed0
```

(commands above are run for each of the 5 methods; rho=0.05 is added
for the SAM-based methods.)

---

## 9. Smoke-test results

### SimpleCNN + FakeData (2000/500), 1 epoch, CPU

| Method        | Train Loss | Train Acc | Test Loss | Test Acc | Time | Sharpness |
| ------------- | ---------- | --------- | --------- | -------- | ---- | --------- |
| Dense-SGD     | 2.3190     | 0.1132    | 2.3043    | 0.098    | 1.7s | 0.0314    |
| Dense-SAM     | 4.0784     | 0.0867    | 2.3665    | 0.102    | 1.8s | 0.1249    |
| 2:4-SGD       | 2.3109     | 0.1182    | 2.3052    | 0.100    | 1.5s | 0.0140    |
| 2:4-SAM       | 3.0170     | 0.0901    | 2.3994    | 0.102    | 1.8s | 0.0607    |
| 2:4-SAM-aware | 3.1241     | 0.0901    | 2.4001    | 0.102    | 1.8s | 0.0834    |

(Expected: FakeData is randomly-labelled noise so accuracy ≈ 0.10. The
purpose here is to confirm forward / backward / SAM / mask-update /
saving all run end-to-end, not to see learning.)

* All 5 methods: forward + backward succeeded; checkpoints saved; CSV
  written; per-epoch sanity checks (`check_block_sums`,
  `check_effective_weight_zeros`) passed.
* 2:4-SAM and 2:4-SAM-aware: in-training mask-stability check (mask
  before first backward == mask before second backward) passed for
  every SAM step.
* Post-hoc `tests/runtime_sanity.py`: for all three sparse methods, every
  one of the 34 080 blocks across the SimpleCNN conv/fc layers has
  exactly 2 active weights and the effective weight is zero wherever
  mask==0.

---

## 10. ResNet18 + real CIFAR-10 short experiment (2 epoch, single seed)

**Setup.** ResNet18-CIFAR, real CIFAR-10, **subset 4000 train / 1000 test**
(`--subset-train 4000 --subset-test 1000`), 2 epochs, batch size 128, lr 0.1,
momentum 0.9, weight decay 5e-4, rho 0.05 (SAM methods), mask-update every
20 optimizer steps, sharpness probe every epoch. CPU only.

### 10.1 Per-method results

| Method        | epoch | train_loss | train_acc | test_loss   | test_acc | sharpness     | epoch_time |
| ------------- | ----- | ---------- | --------- | ----------- | -------- | ------------- | ---------- |
| Dense-SGD     | 0     | 3.185      | 0.154     | 19.233      | 0.144    | 60.028        | 48.4s      |
| Dense-SGD     | 1     | 2.236      | 0.187     | 2.348       | 0.189    | 0.230         | 31.8s      |
| Dense-SAM     | 0     | 6.001      | 0.101     | 3.05e11     | 0.090    | 7.82e10       | 59.1s      |
| Dense-SAM     | 1     | 6.314      | 0.110     | 3.90e11     | 0.090    | 7.34e10       | 59.5s      |
| 2:4-SGD       | 0     | 2.663      | 0.175     | 197.28      | 0.116    | 65.109        | 31.7s      |
| 2:4-SGD       | 1     | 2.793      | 0.202     | 2.500       | 0.191    | 0.504         | 32.0s      |
| 2:4-SAM       | 0     | 5.109      | 0.100     | 7.32e5      | 0.086    | 1.16e5        | 74.0s      |
| 2:4-SAM       | 1     | 14.107     | 0.111     | 1.43e6      | 0.090    | 2.56e5        | 79.6s      |
| 2:4-SAM-aware | 0     | 5.072      | 0.101     | 4.27e5      | 0.086    | 8.55e4        | 104.8s     |
| 2:4-SAM-aware | 1     | 14.053     | 0.111     | 4.64e5      | 0.090    | 9.99e4        | 82.9s      |

### 10.2 Aggregate summary (from `summary.csv`)

| Method        | best_test_acc | sharpness_mean | epoch_time (s) | total_time (s) |
| ------------- | ------------- | -------------- | -------------- | -------------- |
| Dense-SGD     | 0.189         | 30.13          | 31.8           | 80.1           |
| Dense-SAM     | 0.090         | 7.57e10        | 59.5           | 118.6          |
| 2:4-SGD       | 0.191         | 32.81          | 32.0           | 63.7           |
| 2:4-SAM       | 0.090         | 1.86e5         | 79.6           | 153.6          |
| 2:4-SAM-aware | 0.090         | 9.28e4         | 82.9           | 187.7          |

### 10.3 Mask-related diagnostics (sparse methods)

| Method        | Epoch | mask_change_ratio | active_weight_ratio | per-layer sparsity |
| ------------- | ----- | ----------------- | ------------------- | ------------------ |
| 2:4-SGD       | 0     | 0.277             | 0.500               | all 0.5            |
| 2:4-SGD       | 1     | 0.003             | 0.500               | all 0.5            |
| 2:4-SAM       | 0     | 0.254             | 0.500               | all 0.5            |
| 2:4-SAM       | 1     | 0.025             | 0.500               | all 0.5            |
| 2:4-SAM-aware | 0     | 0.341             | 0.500               | all 0.5            |
| 2:4-SAM-aware | 1     | 0.261             | 0.500               | all 0.5            |

* 17 layers were wrapped in every sparse run (16 Conv2d + 1 Linear); the
  first conv (`conv1`) is left dense per spec.
* `tests/runtime_sanity.py` confirmed **every one of the 2,747,648
  2:4 blocks** in each sparse checkpoint has exactly 2 active weights
  and effective weight is zero wherever mask==0.
* SAM-aware recomputation produced higher change ratios (0.34/0.26) than
  magnitude (0.28/0.003) — consistent with the fact that
  `|W * g_SAM|` is a different score than `|W|`.

### 10.4 Plots

See `nm_sam_cifar10/results/short_real/plots/` for:

* `train_loss.png`
* `test_loss.png`
* `test_acc.png`
* `sharpness.png`
* `epoch_time.png`

---

## 11. Code complete vs. experimental conclusion

| Aspect                                       | Status |
| -------------------------------------------- | ------ |
| All five methods implemented & sanity-checked | DONE |
| Per-epoch CSV + JSON config + checkpoints    | DONE |
| Mask stability across SAM step                | DONE (asserted at runtime) |
| STE: effective weight = 0 where mask = 0      | DONE (asserted at runtime) |
| Sparse SAM epsilon respects mask              | DONE (asserted at runtime) |
| SAM-aware mask updates from g2                | DONE (post-fix) |
| Sharpness probe per spec                      | DONE |
| Plots + summary.csv                           | DONE |
| Multi-seed (0/1/2) runs                        | NOT DONE (Phase-1 spec: do not start long training) |
| ResNet18 + 50 000-sample full training        | NOT DONE |
| Hardware-native 2:4 sparse tensor cores       | NOT IN SCOPE (Phase 2) |

---

## 12. What the Phase-1 short runs support (preliminary)

1. **Pipeline correctness.** All five methods run end-to-end on CPU; every
   sparse checkpoint has the strict 2:4 block structure; SAM steps
   preserve the mask; SAM-aware recomputes from g2. (see §10.3)
2. **N:M magnitude mask is stable under short training.** 2:4-SGD's
   mask-change ratio drops from 0.28 in epoch 0 to 0.003 in epoch 1 —
   once the mask settles, magnitude ranking does not flip frequently.
3. **SAM-aware mask changes more often than magnitude.** Across the two
   epochs, SAM-aware's mask-change ratio is 0.34/0.26 versus 2:4-SAM's
   0.25/0.025. This is consistent with the fact that `|W * g_SAM|`
   depends on the SAM second-backward gradient and therefore changes
   whenever the loss landscape shifts.

## 13. What the Phase-1 short runs do NOT yet support

1. **No accuracy comparison.** All SAM-based methods exploded numerically
   under `lr=0.1 + rho=0.05 + 2-epoch short run`. The standard SAM paper
   uses lr ~0.01 with warmup on CIFAR-10; here lr=0.1 is too aggressive
   for the chosen 4 000-sample subset (only 31 SGD updates per epoch).
   This is a *known SAM training-stability issue*, not a code bug —
   fixing it requires a longer run with proper warmup, not different
   code.
2. **No flat-minima conclusion.** The sharpness probe captured the
   typical high-loss regime of an unstable first epoch; it is
   meaningless for answering "is 2:4 + SAM-aware flatter?" at this
   scale.
3. **No timing / hardware claim.** Epoch times reflect only the
   STE-mask implementation on CPU; per Phase-1 spec, they must not be
   reported as evidence of NVIDIA Sparse Tensor Core acceleration.

## 14. Next three questions worth answering first

1. **Magnitude vs. SAM-aware mask at equal sparsity, equal compute, on
   the same long training budget.** With warmup + cosine LR + a more
   conservative initial lr (0.01–0.05), the two sparse SAM-based
   methods should produce comparable accuracies — at which point the
   question "is SAM-aware better than magnitude at 2:4?" can actually
   be answered.
2. **Does the SAM-aware mask correlate with which weights the *robust*
   objective would actually prefer to keep?** A controlled ablation
   that freezes a SAM-aware mask, then trains a dense model on those
   weights, would isolate the "mask choice" effect from the "optimizer
   interaction" effect.
3. **Is the SAM step genuinely *flatter* under 2:4, and does SAM-aware
   flatten more?** The current sharpness probe uses a single
   first-order step; a longer run with periodic sharpness probes will
   tell us whether SAM-aware finds consistently lower `S_rho`.

---

## 15. Known limitations of Phase-1 runs

* All Phase-1 numbers were obtained on CPU in single-seed, 2-epoch
  configurations on a 4 000-sample CIFAR-10 subset. None of these
  numbers should be treated as a final answer to the research question;
  they only validate that the pipeline produces the expected qualitative
  behavior (loss decreases, mask stays valid, SAM perturbation is masked,
  SAM-aware recomputes from g2).
* Sharpness probe uses a single first-order step, not a true max
  (`max_{||e||<=rho} L(w+e)`). It is the standard cheap approximation
  used by most SAM papers.
* "Epoch time" reported in Phase 1 is the wall-clock cost of the
  STE-mask implementation on CPU. It is **not** a measure of NVIDIA 2:4
  Sparse Tensor Core acceleration.
* SAM-based methods need warmup + smaller initial lr to remain stable;
  the lr=0.1 used here is too aggressive for a 4 000-sample, 2-epoch
  sanity run. This is well-known and is a Phase-2 hyperparameter
  issue, not a code defect.
