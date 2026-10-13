"""Read-only local CIFAR Python TAR, bounded extraction and official file hashes."""
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
import hashlib
import ntpath
import shutil
import tarfile
from torchvision.datasets import CIFAR10

MAX_BYTES = 512 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_members(archive: tarfile.TarFile) -> list[tarfile.TarInfo]:
    members = archive.getmembers()
    seen, total = set(), 0
    if len(members) > 100:
        raise ValueError("too many archive members")
    for member in members:
        name = member.name
        path = PurePosixPath(name)
        if (path.is_absolute() or ntpath.isabs(name) or "\\" in name or ":" in name
                or ".." in path.parts or not path.parts
                or path.parts[0] != "cifar-10-batches-py"):
            raise ValueError(f"unsafe or unsupported member: {name}")
        if name in seen or not (member.isfile() or member.isdir()):
            raise ValueError("duplicate/link/special archive entry rejected")
        seen.add(name)
        total += member.size
        if total > MAX_BYTES or member.size < 0:
            raise ValueError("archive exceeds extraction budget")
    required = {"cifar-10-batches-py/" + name for name, _ in CIFAR10.train_list + CIFAR10.test_list}
    required.add("cifar-10-batches-py/batches.meta")
    if not required.issubset(seen):
        raise ValueError("not complete CIFAR-10 Python format")
    return members


def verify_python_files(root: Path) -> None:
    """Verify known official MD5 before torchvision ever unpickles a file."""
    expected = CIFAR10.train_list + CIFAR10.test_list + [(CIFAR10.meta["filename"], CIFAR10.meta["md5"])]
    for name, md5 in expected:
        path = root / CIFAR10.base_folder / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("missing/link file in cache")
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        if digest != md5:
            raise ValueError(f"corrupt or nonofficial CIFAR file: {name}")


def prepare_archive(archive_path: str | Path, dataset_cache_dir: str | Path) -> tuple[Path, str]:
    path = Path(archive_path).resolve(strict=True)
    cache = Path(dataset_cache_dir).resolve()
    repo = Path(__file__).resolve().parents[3]
    if cache == repo or repo in cache.parents:
        raise ValueError("dataset cache must be outside repository")
    archive_hash = sha256(path)
    target = cache / archive_hash
    if target.exists():
        verify_python_files(target)
        return target, archive_hash
    cache.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(path, "r:*") as tar, TemporaryDirectory(dir=cache, prefix="cifar-stage-") as tmp:
            members = checked_members(tar)
            staging = Path(tmp)
            for member in members:
                destination = (staging / member.name).resolve()
                if staging.resolve() not in destination.parents:
                    raise ValueError("extraction escapes staging")
                if member.isdir():
                    destination.mkdir(parents=True, exist_ok=True)
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    source = tar.extractfile(member)
                    if source is None:
                        raise ValueError("unreadable member")
                    with source, destination.open("wb") as output:
                        shutil.copyfileobj(source, output)
            verify_python_files(staging)
            staging.rename(target)
    except (tarfile.TarError, EOFError, OSError) as error:
        raise ValueError(f"invalid/truncated CIFAR archive: {type(error).__name__}") from error
    return target, archive_hash
