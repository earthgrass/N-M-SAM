# v1 pilot invalidated

The runner used `weight_to_matrix(layer.weight.detach()).contiguous()`, which
retained a view of the live model parameter. Swapping the layer weight for
training-side diagnostics then changed the purported immutable Dense reference.
The first arm may have valid local calculations, but the four-arm comparison,
subsequent physical errors, and summary are **invalid**. The original v1 files
are preserved as failure evidence. The corrected run uses an independent
`clone()` and a new v2 output directory. No Test data was accessed.
