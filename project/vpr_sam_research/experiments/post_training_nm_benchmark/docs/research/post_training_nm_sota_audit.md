# Post-training N:M literature and implementation audit

Current authorized extension: OBC32-image technical smoke is now approved and
frozen separately from the still-unapproved formal budget. SNOWS has an additional
controlled acceptance-only patch with compatible statuses; it remains blocked
on actual CIFAR18 representations/Mask and Armijo data semantics. See the latest
PRE_EXECUTION_ACCEPTANCE and SNOWS_STEP_ACCEPTANCE reports. Prior source audit
and literature claims below are retained as historical evidence, not a claim
that no controlled patch now exists.

Audit date: 2026-10-25. This is a controlled benchmark preparation, not a SOTA
claim or completed reproduction. Primary papers establish method identity;
pinned official source establishes the executable path. Published numbers from
other models/datasets are not inserted into the local comparison table.
All five repositories and inspected file SHA256 values are locked in
`configs/upstream_lock.json`. Official code is fetched externally, not vendored.

## Candidate decisions

| Candidate | Primary source / native setting | Actual sparsity and optimization | Decision for this benchmark |
|---|---|---|---|
| OBC / ExactOBS | [NeurIPS 2022 paper](https://papers.nips.cc/paper_files/paper/2022/file/1caf09c9f4e6b0150b06a07e77f2710c-Paper-Conference.pdf), [official code](https://github.com/IST-DASLab/OBC/tree/9b7979bfc9ee20d87db553823a32ee9890beaa99). ImageNet CNNs and other compression tasks; locked loader has no CIFAR branch. | Native `nmprune` exists; removes weights with inverse-Hessian greedy scores and compensates surviving weights. Native Conv groups traverse input channels at each spatial kernel location. | CPU-tested coordinate adapter; CUDA pilot pending. Exact project CIFAR-ResNet18 + row-contiguous2:4 is a **controlled adaptation**, not reproduction of published numbers. |
| SNOWS | [ICLR 2025 paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/57e5c3cb4c1e56cd7c12569426121a31-Paper-Conference.pdf), [official code](https://github.com/mazumder-lab/SNOWS/tree/750497fe13454253056307956a56d581fedf1128). CNN examples include ResNet20/CIFAR10, ResNet50/CIFAR100; ViT code separate. | N:M entry exists. Nonlinear downstream representation reconstruction, Hessian-free/CG surviving-weight optimization; not plain Magnitude. Native Conv group order differs. | **Blocked at pinned code**, not silently replaced or patched. Module syntax, chunked HVP and zero-RHS CG diagnostics below. Four-stage current model also needs a tested architecture bridge. |
| CAP | [NeurIPS 2023 paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/5bd9fbb3a5a985f80c16ddd0ec1dfc43-Abstract-Conference.html), [official code](https://github.com/IST-DASLab/CAP/tree/f4cbe69663d4b2d3438754adb5ca7a477882caad). Empirical-Fisher correlation-aware pruning; recipes include fine-tuning. | Repo contains N:M handle/ViT recipe; presence of recipe does not establish an operational CIFAR18 one-shot path. | Conditional candidate **blocked by two native constructor errors**. Do not substitute the unstructured branch. |
| MOONSHOT | [arXiv 2604.13287](https://arxiv.org/abs/2604.13287), [TMLR accepted paper](https://openreview.net/pdf/7fc99052b62b8588735b28294d8386aa7c1efa53.pdf), [official code location](https://github.com/mazumder-lab/MOONSHOT/tree/3961cefeefd6d506b191ce6de876e2dc334ca9ed). Multi-objective vision/LLM pruning. | Paper-level candidate; no executable method in locked repository. | **Literature only**. Repository exists, but only README/LICENSE; README says code will be released. No fabricated implementation. |
| UniPTS | [CVPR 2024 paper](https://openaccess.thecvf.com/content/CVPR2024/html/Xie_UniPTS_A_Unified_Framework_for_Proficient_Post-Training_Sparsity_CVPR_2024_paper.html), [official code](https://github.com/xjjxmu/UniPTS/tree/7b29888cfce16032c21ca5d9dbe90b9cfd0f93fe). ResNet18/50 ImageNet configurations. | Limited-calibration optimization with dynamic sparsity/regrowth; `GetWeightSubnet` sorts flattened whole-layer magnitudes. No verified equivalent fixed row2:4 branch. | **Literature only / incompatible native N:M path**. Imposing new blockwise selection would change its mechanism and needs a separate decision. |

## Calibration and BN are separate variables

The user explicitly left the **common pruning calibration budget undecided**.
Samples, indices, index SHA256, rounds and augmentation are all pending;
the formal entry point rejects the config. No4096-image pruning budget is assumed.

Native defaults are documentation, not approved local settings:

* OBC `main_trueobs.py:26,57–60`:1024 distinct samples; default10 augmentation
  rounds for standard vision; `--noaug` changes default rounds to1.
  Native compression does not automatically invoke optional BN tuning.
  `postproc.py:97–119` exposes separate `--bnt`, reset stats/momentum.1,
  default100 batches. The benchmark default records native BN as disabled;
  enabling this extra branch requires its own recorded setting.
* SNOWS `CNN_pruning/prune_NM.py` parser `ngrads=128`; README example uses3000.
  An example is not a universal method budget. Architecture/representation
  optimization and BN semantics need validation before a formal comparison.
* CAP/UniPTS calibration/optimization budgets depend on recipes; no numeric
  common budget has been selected from them.

The existing4096 no-augmentation train IDs are copied solely for the already
frozen **common BN-only** control: one pass, batch128, momentum=None,32batches,
independent model copy. This is not evidence of approval for pruning calibration.
New recovery starts from native post-pruning weights **before** that BN control.

## Reproducible blockers, not GPU excuses

SNOWS pinned `CNN_pruning/prune/SNOWS.py` fails Python parsing: a function at
line913 has no body, producing an IndentationError at965. For diagnostics only,
unaltered complete function bodies were extracted and executed; that is not a
working full method. In `hessian_vector_product_chunks:411–413`, the second
mask assignment overwrites the lower chunk bound. For the analytic objective
0.5||w||²,200008 coordinates/100004 retained, default chunk100000 gives HVP
values1 and2 instead of all1 (max error1). Eight eligible current Conv layers
have >100000 retained coordinates. A zero right-hand side also makes native CG
nonfinite. These failures are retained; no tolerance was relaxed or formula fixed.

CAP `modifier_pruning_cap.py:328` sends n/m to `CAPHandle`, whose signature
does not accept them (TypeError). Direct `CAPNMHandle` uses `self.m` before
assignment at `pruning_handle.py:371–373` (AttributeError). On Windows its
recipe filename contains `:`, so relevant source was restored without that
filename and the recipe read directly from the immutable Git blob. This OS
checkout limitation does not explain or erase the algorithm constructor errors.

The initial audit applied no patch. The later GPU-smoke preparation now contains
an explicitly labelled isolated SNOWS kernel-repair candidate, authorized for
investigation by the user. It is not a full accepted solver. See
[controlled repair evidence and remaining behavioral blockers](SNOWS_CONTROLLED_REPAIR.md).
No newer advertised official revision was found in the fresh Git ref check.
Further full-solver behavioral changes require review and remain stopped.
Do not count diagnostic tests passing as these methods passing reproduction.

## 2026-10-25 calibration call-path clarification (not approval)

OBC nsamples1024 is the size of a seeded random ImageNet subset, not1024
batches. main_trueobs iterates that loader nrounds times; default10 revisits
the same IDs with newly drawn native RandomResizedCrop224/Flip views. With
noaug, resize/center-crop ImageNet inputs and default1round are used. Batch
defaults to128 for ImageNet. The locked loader supports ImageNet/COCO/SQuAD,
not CIFAR; the earlier audit's CIFAR-loader claim is corrected here.

SNOWS prune_NM.ngrads128 is passed to LayerPruner.nsamples. prune_NM then
constructs its own batch1 shuffled training loader, caches the first nsamples
images in xdata and copies xdata2. It is an image-count budget, not128optimizer
steps or128batches. README's N:M example overrides it to3000 for
ResNet50/CIFAR100/1:4/k_step30; its CIFAR10 example is unstructured and has a
different protocol. Thus128 and3000 control the same cache size but refer to
different declared configurations, not equivalent budgets. Crop/Flip is drawn
while caching; reconstruction reuses cached tensors, unlike OBC's repeated
augmented loader rounds. Native CIFAR normalization std(.2023,.1994,.2010)
differs from the existing Dense-SAM parent's SSAM normalization. Matching
the actual parent is a controlled preprocessing adaptation, not official-value
reproduction. No active calibration setting is approved by these findings.

## Cost and comparison boundaries

OBC native compensation is part of its method and must be retained. Its output
is called **native post-pruning**, never APBR-style unmodified Raw Hard.
Current19-layer adaptation accumulates one layer's Hessian at a time from the
unchanged dense teacher; this bounds memory but repeats calibration forwards.
No additional training or calibration optimizer is substituted.
For the largest4608-column layer, one FP64 H is about162MiB and a32-row
FP32 inverse batch about2.53GiB, before workspace/model/copies. CUDA peak,
wall time and stability remain unmeasured. A technical seed0 pilot is mandatory.

Main table: Accuracy only, mean±sampleSD over independent seeds0/1/2.
Preserve per-seed Loss/Final/epochs/provenance internally. Repeated use of the
official Test set for Best selection is disclosed; these are not unbiased
held-out estimates. Existing E0/E1 results are reused without retraining.
Different pretrained parents and APBR training cost are part of the full-pipeline
comparison and must be visible in the cost report. No sparse hardware speedup
is claimed merely because Mask legality passes.
