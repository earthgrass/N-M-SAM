"""No hardware queries on the default path."""
import torch


def capabilities():
    return {"torch":torch.__version__,"cuda_build":torch.version.cuda,
            "cuda_runtime":"GPU_VALIDATION_PENDING","amp":"UNSUPPORTED_FAIL_CLOSED",
            "ddp":"UNSUPPORTED_FAIL_CLOSED","ddp_build":torch.distributed.is_available()}


def require_execution_backend(config):
    if config.amp or config.ddp:
        raise NotImplementedError("AMP/DDP not enabled: per-pass unscale/allreduce parity pending; use single-device FP32")
    if config.device=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: GPU_VALIDATION_PENDING")
