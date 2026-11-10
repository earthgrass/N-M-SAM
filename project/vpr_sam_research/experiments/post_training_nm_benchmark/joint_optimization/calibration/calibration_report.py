from __future__ import annotations

from pathlib import Path


def write_summary(path: Path, lambda_result: dict, eta_result: dict, *, source: str) -> None:
    status = "PENDING_REAL_ASSET" if source != "REAL_MODEL_TRAIN_BATCH" else "LOCAL_CALIBRATION_PASS"
    text = f"""# APBR-GPT Joint local calibration

- Status: `{status}`
- Calibration source: `{source}`
- Final Test accessed: `false`
- CUDA used: `false`

## lambda_p

- Value: `{lambda_result['lambda_p']:.17g}`
- CE eligible-gradient norm: `{lambda_result['ce_gradient_norm']:.17g}`
- Pruning eligible-gradient norm: `{lambda_result['prune_gradient_norm']:.17g}`
- Achieved ratio: `{lambda_result['achieved_ratio']:.17g}`
- Acceptance interval: `[{lambda_result['lower_bound']}, {lambda_result['upper_bound']}]`

## eta_s

- Value: `{eta_result['eta_s']:.17g}`
- Energy-update norm: `{eta_result['energy_update_norm']:.17g}`
- Independent pruning-update norm: `{eta_result['independent_prune_update_norm']:.17g}`
- Achieved ratio: `{eta_result['achieved_ratio']:.17g}`
- Energy/pruning cosine: `{eta_result['cosine_similarity']:.17g}`
- Combined projected update norm: `{eta_result['combined_update_norm']:.17g}`

These numerical values validate the calibration interfaces only. They are not
authorized formal-training constants because the historical Seed 0 initial
tensor state cannot be reconstructed under the current local PyTorch runtime.
They must be recalibrated once on the verified server model and fixed training
batch before formal training.
"""
    path.write_text(text, encoding="utf-8", newline="\n")
