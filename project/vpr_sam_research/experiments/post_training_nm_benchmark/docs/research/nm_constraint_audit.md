# N:M hard gate — canonical contract frozen

Convention: `oihw_output_row_contiguous_v1`.

Conv `[O,I,kH,kW]` is flattened to `[O,I*kH*kW]` without permutation.
Linear `[O,I]` is already row-major. Within each output row, consecutive
4coordinates retain exactly2Mask positions. Dense tails remain unpruned.
Exclude first Conv2d, last Linear, all BN affine, biases and buffers.
Magnitude/guided ties keep lower coordinates. Original magnitude scores
are FP32 abs(w); original APBR scores are FP64 abs(w)*exp(-s), unchanged.

Actual model:19eligible convolutions,11,157,504eligible coordinates;
every row is divisible by4, so no eligible tail occurs in this benchmark.

## Native grouping is different

For `[1,4,3,3]`, the first canonical group is `[0,1,2,3]`.
OBC `prepare(columnslast=True)` permutes I,H,W into H,W,I and first groups
`[0,9,18,27]`. SNOWS's convolution helper creates that same channel-strided
group. They agree on1x1, not3x3. A native result must never just be relabelled
as canonical. Proof is saved in `artifacts/compatibility_probes.json`.

## Required adaptations

OBC: use original Conv2d Unfold and Hessian normalization, then present
unchanged flattened weights and the same Hessian through original TrueOBS's
Linear interface (identity permutation). Native greedy scores, compensation
and inverse updates stay byte-identical. CPU tests must establish Unfold
output/objective/Hessian equivalence, legality and compensation. This is a
controlled grouping/model-interface adaptation, not strict reproduction of
the paper's native-convention published numbers.

SNOWS: its documented arbitrary-Mask optimization could accept canonical
magnitude Masks from Dense-SAM. However its native3-stage ResNet20/ResNet50
helpers do not support this4-stage functional-ReLU/shortcut architecture.
A bridge would need exact forward/gradient/HVP equivalence and shortcut
coverage. Its current syntax/numerical gates fail; no full adapter is
silently repaired or approved in this stage.

CAP: flattened Fisher blocks can align with canonical groups when row and
Fisher dimensions divide4. Actual constructor/N:M entry failures currently
block execution. UniPTS's global percentage mask is not a verified N:M
pattern; MOONSHOT's pinned public repository contains no algorithm code.

Gate meaning: adapt compatible interfaces only; do not compare different
patterns, change existing APBR artifacts, or override failed numerical gates.
