from pathlib import Path

import scripts.run_group_joint_obs_seed0_full as runner


def test_exactly_eight_frozen_methods():
    assert len(runner.METHODS) == 8
    assert len(set(runner.METHODS)) == 8
    assert runner.METHODS[0] == "original_sparsegpt"
    assert "group_obs" in runner.METHODS
    assert sum(name.startswith("apbr_tau_") for name in runner.METHODS) == 3
    assert sum(name.startswith("shuffled_") for name in runner.METHODS) == 3


def test_test_pickle_is_opened_only_after_pretest_freeze():
    source = Path(runner.__file__).read_text(encoding="utf-8")
    freeze = source.index('"frozen_pretest_manifest.json"')
    test_open = source.index('load_pickle(extracted / "test", 10_000)')
    test_loader = source.index("test_loader = DataLoader")
    assert freeze < test_open < test_loader


def test_runner_has_no_training_or_recovery_path():
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert "optimizer.step" not in source
    assert "loss.backward" not in source
    assert '"recovery_epochs": 0' in source
    assert '"gradient_training": False' in source
    assert '"bn_calibration": False' in source
