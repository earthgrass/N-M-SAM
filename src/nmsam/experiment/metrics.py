"""Serializable metrics; task loss kept distinct from optimizer loss."""
from dataclasses import asdict
from nmsam.core.step_types import SAMStepReport


def step_metrics(report: SAMStepReport, step: int) -> dict:
    if report.trace is not None:
        raise ValueError("tensor traces are not experiment metrics")
    metrics = asdict(report)
    metrics.pop("trace")
    return {"step": step, **metrics}

