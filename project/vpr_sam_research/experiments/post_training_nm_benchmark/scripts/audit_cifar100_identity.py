"""Download and audit the official CIFAR-100 Python archive.

The downloader uses fixed HTTP byte ranges because the Toronto host is very
slow for a single connection.  Identity is accepted only after the published
MD5 matches; pickle files are never opened before that check succeeds.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import os
import pickle
import tarfile
import time
import urllib.request
from collections import Counter
from pathlib import Path, PurePosixPath

import numpy as np

SOURCE_URL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
EFFECTIVE_URL = "https://cave.cs.toronto.edu/kriz/cifar-100-python.tar.gz"
EXPECTED_BYTES = 169_001_437
PUBLISHED_MD5 = "eb9058c3a382ffc7106e4002c42a8d85"


def digest(path: Path, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_ranges(target: Path, workers: int) -> list[dict[str, object]]:
    if target.exists():
        if target.stat().st_size != EXPECTED_BYTES:
            raise RuntimeError(f"Existing archive has wrong size: {target.stat().st_size}")
        return []
    part = target.with_suffix(target.suffix + ".part")
    if part.exists():
        raise RuntimeError(f"Refusing to overwrite incomplete range file: {part}")
    with part.open("wb") as stream:
        stream.truncate(EXPECTED_BYTES)
    width = (EXPECTED_BYTES + workers - 1) // workers
    ranges = [(start, min(start + width, EXPECTED_BYTES) - 1) for start in range(0, EXPECTED_BYTES, width)]

    def fetch(bounds: tuple[int, int]) -> dict[str, object]:
        start, end = bounds
        last_error: BaseException | None = None
        for attempt in range(1, 5):
            try:
                request = urllib.request.Request(
                    EFFECTIVE_URL,
                    headers={"Range": f"bytes={start}-{end}", "User-Agent": "APBR-CIFAR100-identity-audit/1"},
                )
                with urllib.request.urlopen(request, timeout=180) as response:
                    status = response.status
                    content_range = response.headers.get("Content-Range")
                    if status != 206 or content_range != f"bytes {start}-{end}/{EXPECTED_BYTES}":
                        raise RuntimeError(f"Invalid range response {status=} {content_range=}")
                    written = 0
                    with part.open("r+b", buffering=0) as output:
                        output.seek(start)
                        while True:
                            block = response.read(1024 * 1024)
                            if not block:
                                break
                            output.write(block)
                            written += len(block)
                    if written != end - start + 1:
                        raise RuntimeError(f"Short range {start}-{end}: {written}")
                    return {"start": start, "end": end, "bytes": written, "status": status, "attempt": attempt}
            except BaseException as exc:
                last_error = exc
                if attempt < 4:
                    time.sleep(attempt)
        raise RuntimeError(f"Range {start}-{end} failed after 4 fixed attempts") from last_error

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            receipts = list(pool.map(fetch, ranges))
    except BaseException:
        # Preserve the partial artifact for diagnosis.  It must never be mistaken
        # for the final archive because it retains the .part suffix.
        raise
    os.replace(part, target)
    return receipts


def safe_extract(archive: Path, destination: Path) -> list[str]:
    if destination.exists():
        raise RuntimeError(f"Refusing to overwrite extraction directory: {destination}")
    destination.mkdir(parents=True)
    names: list[str] = []
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            pure = PurePosixPath(member.name)
            if pure.is_absolute() or ".." in pure.parts:
                raise RuntimeError(f"Unsafe archive member: {member.name}")
            names.append(member.name)
            target = destination.joinpath(*pure.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise RuntimeError(f"Unsupported archive member type: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            source = bundle.extractfile(member)
            if source is None:
                raise RuntimeError(f"Cannot read archive member: {member.name}")
            with source, target.open("xb") as output:
                while block := source.read(1024 * 1024):
                    output.write(block)
    return names


def load_pickle(path: Path) -> dict[bytes, object]:
    with path.open("rb") as stream:
        value = pickle.load(stream, encoding="bytes")
    if not isinstance(value, dict):
        raise RuntimeError(f"Unexpected pickle root in {path}")
    return value


def row_hashes(data: np.ndarray) -> list[str]:
    return [hashlib.sha256(memoryview(row)).hexdigest() for row in data]


def duplicate_summary(hashes: list[str], labels: np.ndarray) -> tuple[int, int]:
    grouped: dict[str, list[int]] = {}
    for row_hash, label in zip(hashes, labels.tolist(), strict=True):
        grouped.setdefault(row_hash, []).append(int(label))
    duplicate_groups = [values for values in grouped.values() if len(values) > 1]
    conflicting_groups = sum(len(set(values)) > 1 for values in duplicate_groups)
    return len(duplicate_groups), conflicting_groups


def label_sets_by_hash(hashes: list[str], labels: np.ndarray) -> dict[str, set[int]]:
    grouped: dict[str, set[int]] = {}
    for row_hash, label in zip(hashes, labels.tolist(), strict=True):
        grouped.setdefault(row_hash, set()).add(int(label))
    return grouped


def channel_stats(data: np.ndarray) -> tuple[list[float], list[float]]:
    pixels = data.reshape(data.shape[0], 3, 1024)
    count = data.shape[0] * 1024
    sums = np.zeros(3, dtype=np.float64)
    squares = np.zeros(3, dtype=np.float64)
    for start in range(0, data.shape[0], 1000):
        block = pixels[start : start + 1000].astype(np.float64) / 255.0
        sums += block.sum(axis=(0, 2))
        squares += np.square(block).sum(axis=(0, 2))
    mean = sums / count
    std = np.sqrt(squares / count - np.square(mean))
    return mean.tolist(), std.tolist()


def array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    return hashlib.sha256(memoryview(value)).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise RuntimeError(f"Refusing to overwrite audit output: {args.output}")
    archive = args.data_dir / "cifar-100-python.tar.gz"
    started = dt.datetime.now(dt.timezone.utc)
    archive_existed_before_audit = archive.exists()
    ranges = download_ranges(archive, args.workers)
    if archive.stat().st_size != EXPECTED_BYTES:
        raise RuntimeError("Archive byte count mismatch")
    md5 = digest(archive, "md5")
    sha256 = digest(archive, "sha256")
    if md5 != PUBLISHED_MD5:
        raise RuntimeError(f"Published MD5 mismatch: {md5}")
    extracted = args.data_dir / "extracted"
    members = safe_extract(archive, extracted) if not extracted.exists() else []
    root = extracted / "cifar-100-python"
    expected_files = {"train", "test", "meta", "file.txt~"}
    actual_files = {p.name for p in root.iterdir() if p.is_file()}
    if actual_files != expected_files:
        raise RuntimeError(f"Unexpected extracted files: {sorted(actual_files)}")
    train = load_pickle(root / "train")
    test = load_pickle(root / "test")
    meta = load_pickle(root / "meta")
    train_data = np.asarray(train[b"data"])
    test_data = np.asarray(test[b"data"])
    train_labels = np.asarray(train[b"fine_labels"], dtype=np.int64)
    test_labels = np.asarray(test[b"fine_labels"], dtype=np.int64)
    train_coarse = np.asarray(train[b"coarse_labels"], dtype=np.int64)
    test_coarse = np.asarray(test[b"coarse_labels"], dtype=np.int64)
    checks = {
        "train_shape": train_data.shape == (50_000, 3_072),
        "test_shape": test_data.shape == (10_000, 3_072),
        "train_dtype_uint8": train_data.dtype == np.uint8,
        "test_dtype_uint8": test_data.dtype == np.uint8,
        "fine_label_range": train_labels.min() == 0 and train_labels.max() == 99 and test_labels.min() == 0 and test_labels.max() == 99,
        "coarse_label_range": train_coarse.min() == 0 and train_coarse.max() == 19 and test_coarse.min() == 0 and test_coarse.max() == 19,
        "fine_names_100": len(meta[b"fine_label_names"]) == 100,
        "coarse_names_20": len(meta[b"coarse_label_names"]) == 20,
        "train_balanced_500": set(Counter(train_labels.tolist()).values()) == {500},
        "test_balanced_100": set(Counter(test_labels.tolist()).values()) == {100},
    }
    checks = {name: bool(value) for name, value in checks.items()}
    if not all(checks.values()):
        raise RuntimeError(f"Dataset semantic checks failed: {checks}")
    train_hashes = row_hashes(train_data)
    test_hashes = row_hashes(test_data)
    train_counts = Counter(train_hashes)
    test_counts = Counter(test_hashes)
    train_duplicate_groups, train_label_conflicts = duplicate_summary(train_hashes, train_labels)
    test_duplicate_groups, test_label_conflicts = duplicate_summary(test_hashes, test_labels)
    cross_hashes = set(train_counts) & set(test_counts)
    train_labels_by_hash = label_sets_by_hash(train_hashes, train_labels)
    test_labels_by_hash = label_sets_by_hash(test_hashes, test_labels)
    cross_different_label_sets = sum(train_labels_by_hash[value] != test_labels_by_hash[value] for value in cross_hashes)
    cross_no_shared_label = sum(not (train_labels_by_hash[value] & test_labels_by_hash[value]) for value in cross_hashes)
    mean, std = channel_stats(train_data)
    rng = np.random.default_rng(20_261_026)
    fixed_ids = np.sort(rng.choice(50_000, size=32, replace=False)).tolist()
    finished = dt.datetime.now(dt.timezone.utc)
    result = {
        "status": "DATA_IDENTITY_PASS",
        "source": {
            "official_landing_page": "https://www.cs.toronto.edu/~kriz/cifar.html",
            "requested_url": SOURCE_URL,
            "effective_url": EFFECTIVE_URL,
            "published_md5": PUBLISHED_MD5,
            "audit_started_utc": started.isoformat(),
            "audit_finished_utc": finished.isoformat(),
            "archive_existed_before_audit": archive_existed_before_audit,
            "acquisition": "provided by user" if archive_existed_before_audit else "range download by audit script",
            "external_download_time_utc": None if archive_existed_before_audit else {"started": started.isoformat(), "finished": finished.isoformat()},
            "archive_filesystem_last_write_utc": dt.datetime.fromtimestamp(archive.stat().st_mtime, dt.timezone.utc).isoformat(),
            "range_workers": args.workers,
            "range_receipts": ranges,
        },
        "archive": {"path": str(archive.resolve()), "bytes": archive.stat().st_size, "md5": md5, "sha256": sha256},
        "archive_members": members or sorted(str(p.relative_to(extracted)).replace("\\", "/") for p in extracted.rglob("*")),
        "extracted_files": {
            name: {"bytes": (root / name).stat().st_size, "sha256": digest(root / name, "sha256")}
            for name in sorted(actual_files)
        },
        "checks": checks,
        "labels": {
            "fine_names": [value.decode("utf-8") for value in meta[b"fine_label_names"]],
            "coarse_names": [value.decode("utf-8") for value in meta[b"coarse_label_names"]],
        },
        "train": {
            "count": 50_000, "raw_shape": list(train_data.shape), "image_shape": [50_000, 3, 32, 32], "dtype": str(train_data.dtype),
            "fine_label_distribution": {str(i): int(v) for i, v in sorted(Counter(train_labels.tolist()).items())},
            "coarse_label_distribution": {str(i): int(v) for i, v in sorted(Counter(train_coarse.tolist()).items())},
            "data_sha256": array_sha256(train_data), "fine_labels_sha256": array_sha256(train_labels),
        },
        "test": {
            "count": 10_000, "raw_shape": list(test_data.shape), "image_shape": [10_000, 3, 32, 32], "dtype": str(test_data.dtype),
            "fine_label_distribution": {str(i): int(v) for i, v in sorted(Counter(test_labels.tolist()).items())},
            "coarse_label_distribution": {str(i): int(v) for i, v in sorted(Counter(test_coarse.tolist()).items())},
            "data_sha256": array_sha256(test_data), "fine_labels_sha256": array_sha256(test_labels),
        },
        "duplicates": {
            "train_repeated_rows": int(sum(v - 1 for v in train_counts.values() if v > 1)),
            "train_duplicate_groups": train_duplicate_groups,
            "train_duplicate_groups_with_fine_label_conflict": train_label_conflicts,
            "test_repeated_rows": int(sum(v - 1 for v in test_counts.values() if v > 1)),
            "test_duplicate_groups": test_duplicate_groups,
            "test_duplicate_groups_with_fine_label_conflict": test_label_conflicts,
            "cross_split_unique_overlap": len(cross_hashes),
            "cross_split_hashes_with_different_fine_label_sets": cross_different_label_sets,
            "cross_split_hashes_with_no_shared_fine_label": cross_no_shared_label,
            "definition": "exact equality of all 3072 uint8 channel-major pixel bytes; labels are not part of the key",
        },
        "preprocessing_for_gpu_smoke": {
            "augmentation": "none", "conversion": "uint8 / 255.0 to float32", "layout": "CHW",
            "normalization_mean_train_only": mean, "normalization_std_train_only_population": std,
        },
        "fixed_gpu_smoke_train_sample_ids": fixed_ids,
        "fixed_sample_selection_seed": 20_261_026,
        "test_set_used_for_selection_or_tuning": False,
    }
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "archive_sha256": sha256, "fixed_ids": fixed_ids}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
