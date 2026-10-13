"""CPU structural primitive smoke only; no Sparse-SAM training."""
import json
import torch
from nmsam.sparsity.layout import MaskLayout
from nmsam.sparsity.mask import build_magnitude_mask, enumerate_masks, validate_mask


if __name__ == "__main__":
    counts = {}
    for name, shape in [("linear", (3, 8)), ("conv", (2, 2, 2, 2))]:
        w = torch.arange(1, 1+torch.tensor(shape).prod().item(), dtype=torch.float32).reshape(shape)
        layout = MaskLayout(shape)
        mask = build_magnitude_mask(w)
        validate_mask(mask, layout)
        counts[name] = {"shape": shape, "groups": layout.groups(mask).sum(-1).tolist()}
    print(json.dumps({"candidate_count": len(enumerate_masks()), "valid": True,
                      "layers": counts, "sparse_training": "PLANNED"}, indent=2))

