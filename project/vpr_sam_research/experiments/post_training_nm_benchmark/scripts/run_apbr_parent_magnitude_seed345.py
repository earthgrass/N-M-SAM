"""Seed 3/4/5 wrapper for the frozen APBR-parent magnitude ablation.

The implementation is loaded from the audited Seed 0/1/2 entry point. This
wrapper changes only the experiment identity and accepted seed set before
delegating to the unchanged pruning/recovery implementation.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_apbr_parent_magnitude_ablation as implementation


implementation.ABLATION_ID = "apbr_full_baselines_seed345_20261029"


def main() -> int:
    original_argument_parser = implementation.argparse.ArgumentParser

    class Seed345ArgumentParser(original_argument_parser):
        def add_argument(self, *args, **kwargs):
            if args and args[0] == "--seed":
                kwargs["choices"] = (3, 4, 5)
            return super().add_argument(*args, **kwargs)

    implementation.argparse.ArgumentParser = Seed345ArgumentParser
    try:
        return implementation.main()
    finally:
        implementation.argparse.ArgumentParser = original_argument_parser


if __name__ == "__main__":
    raise SystemExit(main())
