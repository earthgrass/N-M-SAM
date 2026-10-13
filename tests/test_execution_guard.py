from pathlib import Path
import ast


def test_experiment_scripts_have_explicit_execute_guard():
    root=Path(__file__).resolve().parents[1]/"scripts"
    for name in ("smoke_dense_cpu.py","smoke_cifar_cpu.py","smoke_sparse_cpu.py",
                 "smoke_effective_cpu.py","run_cifar_baselines.py"):
        source=(root/name).read_text(encoding="utf-8");ast.parse(source)
        assert '"--execute"' in source and '"dry_run":True' in source.replace(" ","")
