from pathlib import Path
import ast


def test_sam_kernel_independent_of_diagnostics_and_sparsity():
    root=Path(__file__).resolve().parents[1]/"src/nmsam/sam"
    for path in root.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node,ast.ImportFrom):
                assert not (node.module or "").startswith(("nmsam.diagnostics","nmsam.sparsity"))
