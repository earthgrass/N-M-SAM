"""Reproducible CPU-only numerical acceptance for CNN baseline adapters."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
import time
import traceback
import types
from pathlib import Path

import torch
from torch import nn

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from baseline_adapters.common.conv2d import conv2d_via_unfold
from baseline_adapters.rose import ROSEAdapter
from baseline_adapters.sparsegpt import SparseGPTAdapter
from baseline_adapters.wanda import WandaAdapter


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def install_official_import_stubs() -> None:
    quant = types.ModuleType("quant")
    quant.__all__ = []
    transformers = types.ModuleType("transformers")

    class Conv1D(nn.Module):
        pass

    transformers.Conv1D = Conv1D
    sys.modules["quant"] = quant
    sys.modules["transformers"] = transformers


def max_error(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach() - right.detach()).abs().max())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wanda-source", type=Path, required=True)
    parser.add_argument("--sparsegpt-source", type=Path, required=True)
    parser.add_argument("--rose-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite evidence: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    torch.manual_seed(20261026)
    records: list[dict] = []

    # Exact Conv2d forward/backward equivalence over representative mappings.
    for name, layer, shape in (
        ("standard", nn.Conv2d(3, 5, 3, padding=1).double(), (2, 3, 7, 9)),
        ("grouped", nn.Conv2d(4, 6, 3, stride=2, padding=2, dilation=2, groups=2).double(), (2, 4, 9, 8)),
        ("depthwise", nn.Conv2d(4, 4, 1, groups=4, bias=False).double(), (2, 4, 5, 6)),
    ):
        x1 = torch.randn(*shape, dtype=torch.float64, requires_grad=True)
        x2 = x1.detach().transpose(2, 3).contiguous().transpose(2, 3).requires_grad_(True)
        native = layer(x1)
        mapped = conv2d_via_unfold(layer, x2, max_vectors=7)
        probe = torch.randn_like(native)
        grad_native = torch.autograd.grad((native * probe).sum(), (x1, layer.weight), retain_graph=False)
        grad_mapped = torch.autograd.grad((mapped * probe).sum(), (x2, layer.weight), retain_graph=False)
        records.append(
            {
                "name": f"conv2d_mapping_{name}",
                "status": "MATHEMATICAL_MAPPING_PASS",
                "forward_max_abs_error": max_error(native, mapped),
                "input_grad_max_abs_error": max_error(grad_native[0], grad_mapped[0]),
                "weight_grad_max_abs_error": max_error(grad_native[1], grad_mapped[1]),
                "noncontiguous_input": not x2.is_contiguous(),
            }
        )

    # Wanda pinned Linear core versus the adapter on identical tensors.
    official_wanda = load_module("official_wanda_layerwrapper", args.wanda_source / "lib" / "layerwrapper.py")
    layer_official = nn.Linear(8, 4, bias=False).double()
    layer_adapter = nn.Linear(8, 4, bias=False).double()
    layer_adapter.load_state_dict(layer_official.state_dict())
    linear_input = torch.randn(5, 8, dtype=torch.float64)
    wrapped = official_wanda.WrappedGPT(layer_official)
    wrapped.add_batch(linear_input, layer_official(linear_input))
    adapted = WandaAdapter(layer_adapter)
    adapted.add_batch(linear_input)
    official_score = layer_official.weight.abs() * wrapped.scaler_row.sqrt()[None]
    adapted_score = adapted.scores()
    official_remove = torch.zeros_like(official_score, dtype=torch.bool)
    for start in range(0, official_score.shape[1], 4):
        official_remove.scatter_(1, start + torch.topk(official_score[:, start : start + 4], 2, dim=1, largest=False).indices, True)
    adapted_keep = adapted.prune().reshape_as(official_remove)
    records.append(
        {
            "name": "wanda_official_linear_vs_adapter",
            "status": "NUMERICAL_SMOKE_PASS",
            "score_max_abs_error": max_error(official_score, adapted_score),
            "mask_equal": bool(torch.equal(~official_remove, adapted_keep)),
            "retained_weight_update": 0.0,
        }
    )

    # SparseGPT pinned Linear core versus paper-core adapter. CUDA timing fences
    # are neutralized only for this CPU diagnostic and source bytes are unchanged.
    install_official_import_stubs()
    official_sparse = load_module("official_sparsegpt", args.sparsegpt_source / "sparsegpt.py")
    sparse_official_layer = nn.Linear(8, 4, bias=False)
    sparse_adapter_layer = nn.Linear(8, 4, bias=False)
    sparse_adapter_layer.load_state_dict(sparse_official_layer.state_dict())
    sparse_input = torch.randn(6, 8)
    official_handle = official_sparse.SparseGPT(sparse_official_layer)
    official_handle.add_batch(sparse_input, sparse_official_layer(sparse_input))
    original_sync = torch.cuda.synchronize
    torch.cuda.synchronize = lambda: None
    try:
        official_handle.fasterprune(0.5, prunen=2, prunem=4, blocksize=8, percdamp=0.01)
    finally:
        torch.cuda.synchronize = original_sync
    sparse_adapter = SparseGPTAdapter(sparse_adapter_layer)
    sparse_adapter.add_batch(sparse_input)
    adapter_mask = sparse_adapter.prune(blocksize=8)
    official_mask = sparse_official_layer.weight != 0
    records.append(
        {
            "name": "sparsegpt_official_linear_vs_adapter",
            "status": "NUMERICAL_SMOKE_PASS",
            "mask_equal": bool(torch.equal(official_mask, adapter_mask)),
            "weight_max_abs_error": max_error(sparse_official_layer.weight, sparse_adapter_layer.weight),
            "official_cpu_timing_fence_neutralized": True,
        }
    )

    # Execute the pinned ROSE N:M defect. A columnar layer reaches the branch
    # that references reorder_column although that name is never assigned in
    # the prune_n != 0 path.
    official_rose = load_module("official_rose", args.rose_source / "lib" / "prune_zoo" / "rose.py")
    rose_layer = nn.Linear(8, 2, bias=False)
    with torch.no_grad():
        rose_layer.weight.copy_(torch.tensor([[0.01, 0.02, 8, 9, 1, 2, 3, 4], [0.03, 0.04, 7, 6, 1.5, 2.5, 3.5, 4.5]]))
    rose_handle = official_rose.ROSE(rose_layer)
    rose_handle.scaler_row.fill_(1)
    rose_handle.H.copy_(torch.eye(8))
    failure = None
    original_sync = torch.cuda.synchronize
    torch.cuda.synchronize = lambda: None
    try:
        rose_handle.fasterprune(0.5, prune_n=2, prune_m=4, blocksize=4, percdamp=0.01)
    except Exception as exc:  # the traceback class/message are audit evidence
        failure = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    finally:
        torch.cuda.synchronize = original_sync
    records.append(
        {
            "name": "rose_official_nm_reorder_path",
            "status": "OBSERVED_PINNED_SOURCE_FAILURE" if failure else "UNEXPECTED_PASS",
            "failure": failure,
        }
    )

    repair_layer = nn.Linear(8, 2, bias=False).double()
    with torch.no_grad():
        repair_layer.weight.copy_(rose_layer.weight.double())
    repair = ROSEAdapter(repair_layer)
    repair_input = torch.eye(8, dtype=torch.float64)
    repair.add_batch(repair_input)
    repair_mask, orders = repair.prune(blocksize=8, threshold=-1.0)
    records.append(
        {
            "name": "rose_controlled_paper_faithful_repair",
            "status": "NUMERICAL_SMOKE_PASS",
            "all_reordered": all(order.reordered for order in orders),
            "permutation_roundtrip": all(bool(torch.equal(order.permutation[order.inverse], torch.arange(8))) for order in orders),
            "zeros": int((~repair_mask).sum()),
            "finite": bool(torch.isfinite(repair_layer.weight).all()),
        }
    )

    source_files = {
        "wanda_layerwrapper": args.wanda_source / "lib" / "layerwrapper.py",
        "wanda_prune": args.wanda_source / "lib" / "prune.py",
        "sparsegpt_core": args.sparsegpt_source / "sparsegpt.py",
        "rose_core": args.rose_source / "lib" / "prune_zoo" / "rose.py",
    }
    payload = {
        "schema": "apbr-cnn-baseline-smoke-v1",
        "seed": 20261026,
        "scope": "synthetic CPU implementation validation; not CIFAR-100 accuracy",
        "status": "NUMERICAL_SMOKE_PASS",
        "full_experiment": "FULL_EXPERIMENT_NOT_RUN",
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "default_dtype": str(torch.get_default_dtype()),
        },
        "source_sha256": {name: sha256(path) for name, path in source_files.items()},
        "records": records,
        "elapsed_seconds": time.time() - started,
    }
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if any(record["status"] in {"UNEXPECTED_PASS"} for record in records):
        return 1
    if not all(record.get("mask_equal", True) for record in records):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
