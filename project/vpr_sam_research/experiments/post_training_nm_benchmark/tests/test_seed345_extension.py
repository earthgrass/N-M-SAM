from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_seed345_config_changes_only_extension_identity_fields():
    reference = yaml.safe_load((ROOT / "configs/cifar100_50k10k_final_v1.yaml").read_text())
    extension = yaml.safe_load((ROOT / "configs/cifar100_50k10k_final_v1_seed345_20261029.yaml").read_text())
    for key in ("dataset", "model", "training", "apbr_sam", "evaluation", "nm",
                "formal_calibration", "bn_only", "recovery"):
        assert extension[key] == reference[key]
    assert extension["protocol_id"] == reference["protocol_id"]
    assert extension["device"] == reference["device"]
    assert extension["seeds"] == [3, 4, 5]
    assert extension["gpu_uuid"] == "GPU-05cb4980-8f60-df90-148d-e162f0789327"
    assert extension["execution"]["automatic_retry"] is False


def test_read_only_geometry_epoch_diagnostics_matches_independent_values():
    module = load_script("run_cifar100_50k_formal_training_seed345.py")

    class Geometry:
        logs = [torch.tensor([1.0, -1.0], dtype=torch.float64),
                torch.tensor([2.0, -2.0], dtype=torch.float64)]
        d = 4

    before = [torch.tensor([0.5, -0.5], dtype=torch.float64),
              torch.tensor([1.5, -1.5], dtype=torch.float64)]
    saved = [value.clone() for value in Geometry.logs]
    result = module.geometry_epoch_diagnostics(Geometry, before)
    assert result["geometry_s_mean"] == 0.0
    assert abs(result["geometry_s_std_population"] - (2.5 ** 0.5)) < 1e-12
    assert result["geometry_s_min"] == -2.0
    assert result["geometry_s_max"] == 2.0
    assert result["geometry_epoch_net_update_rms"] == 0.5
    assert result["geometry_epoch_net_update_max_abs"] == 0.5
    assert result["geometry_all_finite"] is True
    assert all(torch.equal(left, right) for left, right in zip(saved, Geometry.logs))


def test_seed345_controller_matrix_is_frozen():
    module = load_script("run_cifar100_seed345_full_matrix.py")
    assert module.PROTOCOL == "cifar100_50k10k_final_v1"
    assert module.SEEDS == (3, 4, 5)
    assert module.SAM_METHODS == ("magnitude", "wanda", "sparsegpt", "rose")


def test_apbr_magnitude_seed345_wrapper_identity_and_seed_scope():
    wrapper = load_script("run_apbr_parent_magnitude_seed345.py")
    assert wrapper.implementation.ABLATION_ID == "apbr_full_baselines_seed345_20261029"
    source = (ROOT / "scripts/run_apbr_parent_magnitude_seed345.py").read_text(encoding="utf-8")
    assert 'kwargs["choices"] = (3, 4, 5)' in source


def test_seed3_postprocess_wrapper_changes_only_seed_scope():
    wrapper = load_script("run_cifar100_50k_postprocess_seed3.py")
    assert wrapper.PROTOCOL == "cifar100_50k10k_final_v1"
    source = (ROOT / "scripts/run_cifar100_50k_postprocess_seed3.py").read_text(
        encoding="utf-8"
    )
    assert "choices=(3,)" in source


def test_seed345_continuation_is_bound_to_audited_dense_boundary():
    module = load_script("run_cifar100_seed345_resume_after_seed3_dense.py")
    assert module.SEED3_DENSE_SHA256 == (
        "ef00f4d3819d4e5e5e6e72f2542bfa20578a10bcf0345de85f6256ae60d20efa"
    )
    source = (ROOT / "scripts/run_cifar100_seed345_resume_after_seed3_dense.py").read_text(
        encoding="utf-8"
    )
    assert 'if seed == 3 and method == "sam":' in source
    assert 'root / "seed3/apbr_sam"' in source
    assert 'root / "seed4"' in source


def test_seed345_training_cli_rejects_historical_seed(monkeypatch, tmp_path):
    module = load_script("run_cifar100_50k_formal_training_seed345.py")
    monkeypatch.setattr(
        "sys.argv",
        ["train", "--dataset-root", str(tmp_path), "--vpr-src", str(tmp_path),
         "--config", str(tmp_path / "missing.yaml"), "--output-dir", str(tmp_path / "out"),
         "--method", "sam", "--seed", "0"],
    )
    try:
        module.main()
    except SystemExit as error:
        assert error.code == 2
    else:
        raise AssertionError("Seed 0 must be rejected by the Seed 3/4/5 training entry")
