# APBR-GPT Group-level Joint OBS: Preflight Audit

Date: 2026-10-31 (Asia/Shanghai)  
Scope: repository and mathematical-design audit only. No GPU Pilot, training, pruning run, Test access, or historical artifact mutation was performed.

## 1. Repository identity and protection state

- Git root: `C:/Users/earthgrass73/Desktop/NM稀疏/APBR-post-training-benchmark`
- Branch: `research/post-training-nm-benchmark`
- HEAD: `55fe3e300dbbd09c1178a6316afebc9ecf367947`
- The worktree already contains many untracked research adapters, scripts, configurations, tests, and evidence directories belonging to prior work. They were not reset, cleaned, overwritten, or staged by this audit.
- Protected historical implementations `benchmark/model.py`, `benchmark/nm.py`, and `benchmark/obc.py` were read only. The frozen Original SparseGPT adapter was not edited.

## 2. Frozen Seed 0 parent and geometry identity

| Item | Evidence | Result |
|---|---|---|
| APBR Dense parent | `artifacts/apbr_unified_gpt_full_experiment/checkpoints/apbr_seed0_dense_final.pt` | Present |
| Parent SHA256 | `4332079becdb58fb50f0e68b8bd0dcc251db0184a6050facb524817520a51d82` | Verified |
| Protocol / method / seed | `cifar100_50k10k_final_v1` / `apbr_sam` / `0` | Verified |
| Fixed endpoint | Epoch 200, 78,200 steps | Verified |
| Test isolation | `test_accessed=false`; no Test loader during training | Verified from checkpoint identity |
| Geometry state | `stepper.geometry.method=vpr` | Present |
| Geometry mapping | 62 parameter entries, identical names and shapes to `model.named_parameters()` | Exact match |
| Geometry dtype / finite | FP64 and finite for every eligible weight | Verified |
| Dense weight source | The same checkpoint payload that contains the geometry state | Verified |

The stored geometry tensors are log-scales `s`. Existing APBR mask reconstruction reads them by exact parameter name and computes `abs(w) * exp(-s)`. No radius is relearned or reconstructed from another checkpoint.

## 3. Model, coordinate, and N:M scope audit

The actual model yields 19 eligible internal modules:

`layer1.0.conv1`, `layer1.0.conv2`, `layer1.1.conv1`, `layer1.1.conv2`, `layer2.0.conv1`, `layer2.0.conv2`, `layer2.0.shortcut.0`, `layer2.1.conv1`, `layer2.1.conv2`, `layer3.0.conv1`, `layer3.0.conv2`, `layer3.0.shortcut.0`, `layer3.1.conv1`, `layer3.1.conv2`, `layer4.0.conv1`, `layer4.0.conv2`, `layer4.0.shortcut.0`, `layer4.1.conv1`, and `layer4.1.conv2`.

`benchmark.nm.eligible_layers` enumerates Conv2d/Linear modules and excludes the first and last, thereby excluding the stem Conv2d and final Linear. BN affine parameters and bias are not modules in this set. The verified Original GPT mask has:

- 19 bool keep-mask tensors;
- 11,157,504 eligible positions;
- 5,578,752 deleted positions;
- exactly two retained coordinates in every complete output-row-contiguous group of four;
- dense incomplete tails (all current eligible widths are divisible by four);
- keys exactly equal to the 19 eligible weight names.

The mask file SHA256 is `aea9e43dfe90f0e5cfa137769648e13a72d5971362a8291e81e3d7d565a217e1`. Its embedded tensor hash in the native SparseGPT checkpoint is `24bfc8bbba216bd866077d1085850a79687ce44945d2af19a003dee86242fbfe`.

## 4. Conv2d and Hessian coordinate audit

`baseline_adapters/common/conv2d.py` establishes the exact local-linear map:

- OIHW weights are reshaped to `[Cout, Cin/G*kH*kW]`;
- `torch.nn.functional.unfold` produces `[Cin/G*kH*kW, N*L]` patches;
- both use channel-major, kernel-row, kernel-column order;
- grouped convolutions construct one independent patch matrix and Hessian per group.

`SparseGPTAdapter.add_batch` accumulates `2 X X^T`, and `hessians()` divides by the number of activation vectors. The formal runner uses the fixed 1,024 training images selected by NumPy PCG64 seed `20261027`, no augmentation, one pass, FP32 computation, relative damping `0.01`, and block size `128`. Its calibration ID SHA256 is `3b7da27d79762235c87f18571bc73224d6e6f2b3ba663b7d9a4a3de312f23157`.

## 5. Original SparseGPT factor and dynamic selection semantics

The frozen adapter performs:

1. dead-column handling;
2. `damp = 0.01 * mean(diag(H))` and diagonal addition;
3. lower Cholesky of the damped Hessian;
4. `Q = H_damped^{-1}` via `cholesky_inverse`;
5. upper Cholesky factor `U = chol(Q, upper=True)`, so `Q = U^T U`;
6. left-to-right dynamic 2:4 selection on the currently compensated block;
7. `error = (current - projected) / U_ii` and the frozen block-local/block-global propagation.

The native metric `w_current^2 / U_ii^2` is a sequential conditional score. It is not the same object as the proposed static pair cost computed from the full `Q_DD`. This distinction must remain explicit in names, reports, and comparisons.

Critical implementation constraint: `U[D,D]` is not `Q_DD`. The proposed selector must either retain `Q` returned by `cholesky_inverse` or reconstruct `Q = U^T U`, then index the resulting matrix. Directly indexing the factor would be a mathematical error.

## 6. Fixed-mask OBS replay path

`baseline_adapters/fixedmask_obs/adapter.py` already accepts an immutable external bool keep-mask and uses the same dead-column rule, damping, factor construction, column order, and error propagation as the Original SparseGPT adapter. It never derives or changes support. Historical evidence shows finite outputs, exact deleted zeros, and survivor updates for all 19 layers.

This is a useful base, but it is not yet sufficient proof of native replay identity. Before the new Pilot, the final native SparseGPT mask must be supplied to the external-mask provider in the same process/device/dtype. The output must reproduce the native SparseGPT weights, deleted coordinates, per-layer mask hashes, and compensation loss. On the identical execution path, bitwise equality is the target; any difference requires a stopped audit with max-absolute and relative-error diagnostics.

## 7. Historical evidence relevant to the new hypothesis

- The CNN SparseGPT adapter previously matched the pinned official Linear mask and compensation within the documented tolerance and passed the Conv2d Hessian/unfold checks.
- Fixed-mask OBS demonstrated that second-order compensation strongly affects accuracy and changes surviving weights without changing support.
- Radius-guided coordinate scoring and Shared APBR-Q did not provide stable task-level improvements.
- Shared-Q geometry was close to isotropic in most layers and correlated with Hessian diagonal information. This motivates testing geometry only as a tie-breaker among group-level near-optimal OBS candidates rather than modifying the compensation objective.

These are motivation and scope constraints, not positive evidence for the proposed method.

## 8. Source and configuration lock

| File | SHA256 |
|---|---|
| `baseline_adapters/sparsegpt/adapter.py` | `4b1462b1f09e6e3789406cb506fc4c47b47cfd159fa3d0e4fc013f45927a3f36` |
| `baseline_adapters/common/conv2d.py` | `e6a45097853a8613fabb013e320011f2c2d48d70f9e83a56c2552c94b9129c63` |
| `baseline_adapters/common/nm.py` | `d44f7fa7927f4a8d46adc113b2667188f4deda4a3cc07a97ed43b80d21a711f0` |
| `baseline_adapters/fixedmask_obs/adapter.py` | `d7cff3b33665811bad242ed8139b03dbbff493a665f63074b87aec266d3b88c3` |
| `baseline_adapters/cifar100_model.py` | `9dfd402b4aac66d8ed18062089a07f3779ba119c909859e031b15abd94ade889` |
| `benchmark/nm.py` | `92433dbfd3b768b28922b89954d76fde7c3a6fdd23e661622ac248526b44678f` |
| `benchmark/model.py` | `befe89e8801d0b3b3146b113b5ed815bc0d9acd62bba6ba998eaf62a5b6aad71` |
| `scripts/run_apbr_weights_original_sparsegpt_one.py` | `946e2ba3a4dcbdb5ae63e6ef5eaa938e76964d8bfd9be12867632d2f4222fc7d` |
| `scripts/run_apbr_seed0_fixedmask_obs.py` | `917dc9dfd1c5e4d33a887758203dd0ced9f82c13c6036a7b38ae1ec3d2e8747b` |
| `configs/cifar100_50k10k_final_v1.yaml` | `e4be5df7cf39b1ebd2730999cd6fd42458a5bcc0779e621894b1c588616879d6` |

Pinned upstream SparseGPT provenance remains commit `147d2159dc4f3e9f73e47b32c04d7b3708f44436`, as recorded in `BASELINE_SOURCE_AUDIT.md`.

## 9. Preflight decision

**DESIGN_PHASE_PASS.** The authentic Seed 0 APBR parent, matching FP64 geometry, frozen calibration protocol, native Original GPT checkpoint/mask, coordinate mapping, and external-mask OBS base are present and internally consistent.

This does not authorize a GPU Pilot. The next permitted step is an isolated implementation plus CPU numerical validation. GPU work remains gated on native-mask replay identity, direct `Q_DD` reconstruction tests, six-candidate correctness, deterministic tie behavior, and all required CPU tests.

