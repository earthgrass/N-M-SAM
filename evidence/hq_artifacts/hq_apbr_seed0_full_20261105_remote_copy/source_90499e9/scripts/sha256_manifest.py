"""Write once or verify the independent research workspace SHA256 manifest."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "SHA256SUMS.txt"
IGNORED_DIRS = {".git", ".pytest_cache", ".venv", "__pycache__", "artifacts"}


def candidate_paths() -> list[Path]:
    return sorted(
        (path for path in ROOT.rglob("*")
         if path.is_file() and path != MANIFEST
         and not any(part in IGNORED_DIRS for part in path.relative_to(ROOT).parts)
         and path.suffix != ".pyc"),
        key=lambda path: path.relative_to(ROOT).as_posix(),
    )


def calculate() -> str:
    return "".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(ROOT).as_posix()}\n"
        for path in candidate_paths()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true")
    action.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    expected = calculate()
    if args.write:
        if MANIFEST.exists():
            raise FileExistsError(MANIFEST)
        MANIFEST.write_text(expected, encoding="utf-8", newline="\n")
        print(f"Wrote {len(candidate_paths())} file hashes")
    else:
        actual = MANIFEST.read_text(encoding="utf-8")
        if actual != expected:
            raise AssertionError("Manifest content or workspace files changed")
        print(f"Verified {len(candidate_paths())} file hashes")


if __name__ == "__main__":
    main()
