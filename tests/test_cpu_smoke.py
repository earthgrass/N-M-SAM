from nmsam.experiment.runner import run_dense_cpu


def test_cpu_smoke_reproducible_actual_metrics(tmp_path):
    one = run_dense_cpu("configs/dense_sam_official_toy.yaml", tmp_path)
    two = run_dense_cpu("configs/dense_sam_official_toy.yaml", tmp_path)
    assert one["metrics"] == two["metrics"]
    assert len(one["metrics"]) == 3
    assert all(m["forward_passes"] == 2 and m["restoration_error"] == 0
               for m in one["metrics"])
    assert len(list(tmp_path.glob("dense_*.json"))) == 2

