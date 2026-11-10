"""Create an isolated source-only server bundle with per-file SHA256."""
from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path

RESEARCH = Path(__file__).resolve().parents[3]
PROJECT = RESEARCH / "experiments/post_training_nm_benchmark"
OUT = PROJECT / "artifacts/dense_sam_apbr_l2_seed1to4_20261103_transfer"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    files = [RESEARCH / "AGENTS.md"]
    files += list((RESEARCH / "src/vpr_sam").rglob("*.py"))
    files += [PROJECT / rel for rel in (
        "baseline_adapters/__init__.py", "baseline_adapters/cifar100_model.py",
        "benchmark/__init__.py", "benchmark/model.py",
        "configs/cifar100_50k10k_final_v1_sam_seed245_rtx5090_20261103.yaml",
        "configs/cifar100_50k10k_apbr_fixed_l2_seed1to4_rtx5090_20261103.yaml",
        "configs/seed1_sam_reused_identity_20261103.json",
        "scripts/run_cifar100_50k_sam_seed245_training.py",
        "scripts/run_cifar100_apbr_fixed_l2_seed1to4_training.py",
        "scripts/run_dense_sam_apbr_l2_seed1to4_queue.py",
    )]
    files = sorted(set(files))
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
    lines = [f"{sha(path)}  {path.relative_to(RESEARCH).as_posix()}" for path in files]
    manifest = "\n".join(lines) + "\n"
    (OUT / "SOURCE_SHA256SUMS.txt").write_text(manifest, encoding="utf-8")
    bundle = OUT / "dense_sam_apbr_l2_seed1to4_source_v1.tar.gz"
    with tarfile.open(bundle, "w:gz") as archive:
        for path in files:
            archive.add(path, arcname=(Path("vpr_sam_research") / path.relative_to(RESEARCH)).as_posix())
        data = manifest.encode()
        info = tarfile.TarInfo("vpr_sam_research/SOURCE_SHA256SUMS.txt")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    (OUT / "BUNDLE_SHA256.txt").write_text(f"{sha(bundle)}  {bundle.name}\n", encoding="utf-8")
    print(f"bundle={bundle} files={len(files)} bytes={bundle.stat().st_size} sha256={sha(bundle)}")


if __name__ == "__main__":
    main()
