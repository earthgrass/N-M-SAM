"""Crash after publishing epoch last, before any later validation improvement."""
import copy
import pytest
from vpr_sam.training import engine
from vpr_sam.training.engine import train, read_checkpoint


@pytest.mark.parametrize("method", ["sam", "vpr"])
def test_epoch_last_has_durable_best_before_publication(config, isolated_root, monkeypatch, method):
    from test_server_diagnostics import diagnostics
    c = copy.deepcopy(config); c["optimizer"]["method"] = method
    c["training"]["epochs"] = 2
    # Equal validation scores ensure no future epoch can repair a missing best.
    monkeypatch.setattr(engine, "evaluate", lambda *a, **k: {"accuracy":.5,"loss":1.,"count":4})
    c["output"] = "runs/continuous"
    train(c, isolated_root)
    expected, _ = read_checkpoint(c["output"]+"/final.pt", isolated_root)
    split = copy.deepcopy(c); split["output"] = "runs/interrupted"
    save = engine.atomic_save
    class Interruption(Exception): pass
    def interrupt_after_last(path, payload):
        save(path, payload)
        if path.name == "last.pt" and payload["epoch"] == 1 and payload["next_batch"] == 0:
            assert (path.parent/"best_validation.pt").is_file()
            raise Interruption()
    with monkeypatch.context() as scoped:
        scoped.setattr(engine, "atomic_save", interrupt_after_last)
        with pytest.raises(Interruption): train(split, isolated_root)
    train(split, isolated_root, resume=split["output"]+"/last.pt")
    actual, _ = read_checkpoint(split["output"]+"/final.pt", isolated_root)
    for field in ("model","stepper","scheduler","rng","order_hash","epoch","next_batch","epoch_rows"):
        assert diagnostics.identical(expected[field], actual[field]), field
    best, _ = read_checkpoint(split["output"]+"/best_validation.pt", isolated_root)
    assert best["epoch"] == 1
