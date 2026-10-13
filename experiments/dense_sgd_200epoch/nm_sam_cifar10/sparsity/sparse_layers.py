"""Public sparse-layer API."""
from .parametrization import NMSparsifiedWeight, attach_nm_sparsity

__all__ = ["NMSparsifiedWeight", "attach_nm_sparsity"]
