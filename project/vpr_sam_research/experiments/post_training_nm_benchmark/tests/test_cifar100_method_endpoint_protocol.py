from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "cifar100_method_specific_endpoint_v1.yaml"
POSTPROCESS = ROOT / "scripts" / "run_cifar100_method_endpoint_postprocess.py"
QUEUE = ROOT / "scripts" / "run_cifar100_method_endpoint_queue.py"
SUMMARY = ROOT / "scripts" / "summarize_cifar100_method_endpoints.py"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_endpoint_config_changes_reporting_not_execution():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["comparison_protocol_id"] == "cifar100_method_specific_endpoint_v1"
    assert config["status"] == "FROZEN"
    for method in ("magnitude", "wanda", "sparsegpt", "rose"):
        assert config["methods"][method]["primary_endpoint"] == "raw_hard"
        assert config["methods"][method]["executed_recovery_epochs"] == 20
        assert config["methods"][method]["recovery_role"] == "supplemental"
    assert config["methods"]["sparsegpt"]["native_flow"] == "dynamic_nm_mask_with_obs_compensation"
    assert "sparsegpt_compensation" in config["methods"]["rose"]["native_flow"]
    assert config["methods"]["apbr_sam"]["primary_endpoint"] == "recovery_final_epoch_20"
    assert config["methods"]["apbr_sam"]["executed_recovery_epochs"] == 20
    assert config["methods"]["apbr_sam"]["recovery_role"] == "primary"
    assert config["execution"]["mode"] == "reporting_overlay_on_existing_full_recovery_queue"
    assert config["execution"]["existing_full_queue_continues"] is True
    assert config["execution"]["baseline_recovery_seed1_seed2"] is True
    assert config["execution"]["alternate_endpoint_controller_started"] is False


def test_postprocess_reuses_frozen_pruning_and_only_recovers_apbr():
    source = POSTPROCESS.read_text(encoding="utf-8")
    assert "masks, report = legacy.prune" in source
    assert 'if args.method in BASELINES:' in source
    assert "write_complete_baseline" in source
    assert "recover_apbr" in source
    assert '"recovery_started": False' in source
    assert '"primary_endpoint": "recovery_final_epoch_20"' in source
    assert "Reporting-only protocol does not authorize alternate endpoint execution" in source


def test_queue_main_target_count_and_seed0_preservation():
    module = load(QUEUE, "method_endpoint_queue")
    assert module.METHODS == ("magnitude", "wanda", "sparsegpt", "rose", "apbr_sam")
    source = QUEUE.read_text(encoding="utf-8")
    assert "completed = [audit_seed0(root)]" in source
    assert 'tests.mkdir(parents=True, exist_ok=False)' in source
    assert 'targets.append(("apbr_sam_recovery_final"' in source
    assert "Reporting-only protocol forbids launching the alternate endpoint queue" in source


def test_sample_statistics_use_ddof_one_semantics():
    module = load(SUMMARY, "method_endpoint_summary")
    mean, std = module.mean_std([1.0, 2.0, 3.0])
    assert mean == 2.0
    assert std == pytest.approx(1.0)
    mean, std = module.mean_std([1.0])
    assert mean == 1.0 and std is None


def test_reporting_output_refuses_overwrite(tmp_path):
    module = load(SUMMARY, "method_endpoint_summary_overwrite")
    output = tmp_path / "existing"
    output.mkdir()
    (output / "evidence.txt").write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        module.write_outputs({}, output)

