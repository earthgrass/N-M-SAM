"""Generate v2 selectors from the aggregate manuscript table."""
from __future__ import annotations

import json
from pathlib import Path

root = Path(__file__).resolve().parent
summary = json.loads((root / "aggregate_results.json").read_text(encoding="utf-8"))
entries = []
for method, record in summary.items():
    escaped = method.replace("~", "~0").replace("/", "~1")
    for field in ("runs", "accuracy_mean", "accuracy_sample_std"):
        entries.append({
            "value": record[field],
            "source": "evidence/aggregate_results.json",
            "selector": {"kind": "json-pointer", "pointer": f"/{escaped}/{field}"},
            "representations": ["raw"] if field == "runs" else ["percent"],
            "tolerance": 0.006,
            "note": f"{method}: {field}; table values rounded to two decimal places",
        })
payload = {"schema_version": "ai-research-writing/numeric-evidence-v2", "entries": entries}
constants = json.loads((root / "experiment_constants.json").read_text(encoding="utf-8"))
for key in ("dataset_classes", "resnet_depth", "nm_group_size", "nm_keep_count"):
    entries.append({
        "value": constants[key],
        "source": "evidence/experiment_constants.json",
        "selector": {"kind": "json-pointer", "pointer": f"/{key}"},
        "representations": ["raw"],
        "note": "Frozen dataset/model/N:M protocol constant",
    })
root.parent.joinpath("numeric_evidence.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(f"Registered {len(entries)} manuscript table values")
