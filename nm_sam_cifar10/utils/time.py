"""Timing helpers used by the training loop and metrics."""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def timer() -> Iterator[list]:
    """Context manager that records elapsed wall-clock time in seconds.

    Usage:
        with timer() as t:
            do_work()
        elapsed = t[0]
    """
    elapsed = [0.0]
    start = time.perf_counter()
    try:
        yield elapsed
    finally:
        elapsed[0] = time.perf_counter() - start
