# OBC / SNOWS compatibility contract

Canonical rule: output-row OIHW.reshape(out,-1), consecutive groups of4,
retain2, first/last Conv/Linear excluded, BN/bias excluded, dense tail.
There are19 eligible layers,11157504 coordinates; all row lengths divide4.
For Conv shape[1,4,3,3], canonical first group=[0,1,2,3], both official
native convolution layouts first group=[0,9,18,27]. A completed native Mask
cannot be relabelled as canonical.

## OBC implemented adapter

`benchmark/obc.py` dynamically executes hash-verified official `quant.py` and
`trueobs.py`. The greedy score, selection order, surviving-weight compensation,
inverse updates, damping and fallback are unchanged. Native Conv Unfold is
used to accumulate H. The original native Linear interface receives W.flatten(1)
and that same H, so its identity coordinate permutation produces canonical
groups. Native n means number **removed**; adapter calls m-Nkeep.

An argmin observer records explicit selections without changing the result.
This preserves a valid2:4Mask even when retained coordinates happen to equal0;
`weight != 0` would be an invalid way to recover that Mask. On CPU only,
CUDA timing synchronization is a no-op; no numerical op is replaced.
Grouped Conv and incomplete official buckets reject explicitly. The actual
model has neither, so the generic tail=dense contract is not silently violated.

Tests prove native Conv-H normalization/cast order, Conv-to-Linear forward
equivalence, identity-H selections, correlated-H compensation equal to the
conditional quadratic optimum, nonfinite rejection, explicit zero-value Mask,
scope exclusions, healthy layer-boundary resume and dense teacher immutability.
This validates a small CPU adapter, **not** full-size CUDA operation or accuracy.

## SNOWS withheld adapter

No full SNOWS adapter is advertised. Source audit detects parse failure,
incorrect native chunked HVP on a closed-form test and zero-RHS CG NaNs.
Native model helpers also target different stage/shortcut layouts; current
four-stage CIFAR-ResNet18 uses functional ReLU and `shortcut`, including
projection Conv in canonical scope. A bridge must preserve exact activations,
representation loss and gradients. Replacing it with Magnitude, dropping
second-order updates, disabling chunking just to avoid the counterexample, or
silently inserting a function body would not satisfy this task.

`tests/test_upstream_blockers.py` passing means the failures are reproducible.
It does not make SNOWS or CAP runnable. Those candidates remain stopped.
