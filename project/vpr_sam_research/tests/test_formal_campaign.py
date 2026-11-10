"""Registration and fail-closed campaign checks; never launches CUDA training."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("formal_campaign", ROOT / "scripts/formal_main200.py")
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


def test_registered_six_configs_and_budgets():
    configs = campaign.configs(ROOT)
    assert set(configs) == {(s, m) for s in (0, 1, 2) for m in ("sam", "vpr")}
    assert campaign.STEPS == 70400
    for c in configs.values():
        assert c["device"] == "cuda:0" and c["threads"] == 4 and c["deterministic"]
        assert c["data"]["validation_size"] == 5000 and c["data"]["workers"] == 2
        assert c["data"]["augment"] and c["data"]["cutout"] == 0 and not c["data"]["download"]
        assert c["optimizer"] == dict(method=c["optimizer"]["method"], lr=.1, momentum=.9,
                                      weight_decay=.0005, nesterov=False, rho=.05)
        assert c["geometry"]["gamma"] == 5.482145661475941e-8
        assert c["geometry"]["dtype"] == "float64" and c["geometry"]["warmup_steps"] == 0
        assert c["geometry"]["volume_atol"] == 1e-10 and c["geometry"]["volume_rtol"] == 1e-12
        assert c["training"]["epochs"] == 200 and c["training"]["batch_size"] == 128
        assert c["training"]["scheduler"] == "cosine" and c["training"]["eta_min"] == 0
        assert c["pruning"] == dict(n=2, m=4, scope="exclude_first_last", score="magnitude", tail="dense")
        assert c["recovery"] == dict(epochs=20, method="sgd", lr=.005, momentum=.9,
                                     weight_decay=.0005, rho=.05, scheduler="cosine", eta_min=0.)
        assert c["evaluation"] == dict(batch_size=256, probe_rho=.02, probe_batches=10)


def paired_states():
    state = {"epoch": 200, "next_batch": 0, "stepper": {"steps": 70400},
             "epoch_rows": [None] * 200, "step_rows": [None] * 70400,
             "initialization_hash": "init", "order_hash": "order",
             "manifest": {"fingerprint": "data"}, "source_hashes": {"code": "hash"},
             "environment": {"device": "approved"}}
    expected = {"initialization_hash": "init", "order_hash": "order", "dataset_fingerprint": "data"}
    return [state, copy.deepcopy(state)], expected


def test_completed_pair_positive():
    states, expected = paired_states()
    campaign.check_pair(states, expected)


@pytest.mark.parametrize("field,value", [("epoch", 199), ("next_batch", 1),
                         ("initialization_hash", "wrong"), ("order_hash", "wrong"),
                         ("environment", {"device": "cpu"}), ("source_hashes", {"code": "wrong"})])
def test_completed_pair_rejects_unfair_or_incomplete(field, value):
    states, expected = paired_states()
    states[1][field] = value
    with pytest.raises(ValueError):
        campaign.check_pair(states, expected)


def test_gpu_uuid_and_foreign_device_rejected(monkeypatch):
    monkeypatch.setattr(campaign.subprocess, "check_output", lambda *a, **k:
                        "0, NVIDIA GeForce RTX 5090, GPU-approved, 32607, 2, 0\n")
    assert campaign.gpu_inventory("GPU-approved")[0] == "0"
    with pytest.raises(RuntimeError, match="UUID mismatch"):
        campaign.gpu_inventory("GPU-other")


def test_frozen_rejects_dirty_tree_or_changed_source(monkeypatch):
    monkeypatch.setattr(campaign, "git", lambda root, *args: "commit" if args[0] == "rev-parse" else "dirty")
    with pytest.raises(RuntimeError, match="clean Git"):
        campaign.frozen(ROOT, "commit")
    monkeypatch.setattr(campaign, "git", lambda root, *args: "commit" if args[0] == "rev-parse" else "")
    monkeypatch.setattr(campaign, "source_hashes", lambda root: {"file": "new"})
    with pytest.raises(RuntimeError, match="hashes changed"):
        campaign.frozen(ROOT, "commit", {"file": "old"})


@pytest.mark.parametrize("exit_code", [0, 9])
def test_controller_serial_order_and_failure_stops_queue(tmp_path, monkeypatch, exit_code):
    registered = campaign.configs(ROOT)
    calls = []
    monkeypatch.setattr(campaign, "configs", lambda root: registered)
    monkeypatch.setattr(campaign, "frozen", lambda *a: {"source": "frozen"})
    monkeypatch.setattr(campaign, "gpu_inventory", lambda uuid: ["approved"])
    monkeypatch.setattr(campaign.signal, "signal", lambda *a: None)
    monkeypatch.setattr(campaign.shutil, "disk_usage", lambda root: SimpleNamespace(free=30 * 1024**3))

    class Process:
        def __init__(self, args, **kwargs):
            assert kwargs["start_new_session"]
            assert kwargs["env"]["VPR_FORMAL_APPROVED"] == "YES"
            calls.append(args)
            self.pid = 123

        def poll(self):
            return exit_code

    monkeypatch.setattr(campaign.subprocess, "Popen", Process)
    if exit_code:
        with pytest.raises(RuntimeError, match="failed with exit 9"):
            campaign.run_campaign(tmp_path, "commit", "approved")
        assert len(calls) == 1 and "--gate-only" in calls[0]
        failure = json.loads((tmp_path / campaign.CAMPAIGN / "control/failure.json").read_text(encoding="utf-8"))
        assert failure["automatic_retry"] is False
    else:
        campaign.run_campaign(tmp_path, "commit", "approved")
        trains = [args for args in calls if "train" in args]
        assert [Path(args[args.index("--config") + 1]).stem for args in trains] == [
            "sam_seed0", "vpr_seed0", "sam_seed1", "vpr_seed1", "sam_seed2", "vpr_seed2"]
        assert len([args for args in calls if "recover" in args]) == 6
        assert all("--resume" not in args and "--stop-after" not in args for args in calls)
        assert all("test" not in args for args in calls)
        for seed in (0, 1, 2):
            train_indices = [i for i, args in enumerate(calls) if "train" in args
                             and any(f"seed{seed}.json" in a for a in args)]
            audit_index = next(i for i, args in enumerate(calls)
                               if "--audit-pair" in args and str(seed) == args[-1])
            assert max(train_indices) < audit_index
        status = json.loads((tmp_path / campaign.CAMPAIGN / "control/status.json").read_text(encoding="utf-8"))
        assert status["state"] == "complete"
        with pytest.raises(FileExistsError):
            campaign.run_campaign(tmp_path, "commit", "approved")
