"""Inspect future GPU artifacts without turning observations into novelty claims."""
from pathlib import Path
import json
from nmsam.diagnostics.candidate_ranking import validate_labels


def report_records(records):
    lines=["# Generated experiment readout","","Status: GPU_VALIDATION_PENDING until authorized artifacts are supplied.",
           "No automatic claim of improved generalization, exact network max or scientific novelty.",""]
    for record in records:
        required={"run_id","git_sha","config_sha256","config","seed"}
        if not required.issubset(record):raise ValueError("artifact provenance incomplete")
        lines.extend([f"## Run {record['run_id']}","",f"Source: `{record['git_sha']}`; config `{record['config_sha256']}`.",""])
        for run in record.get("runs",[]):
            if not {"archive_sha256","split_sha256"}.issubset(run["data"]):raise ValueError("missing data fingerprints")
            evaluation=run["validation"]
            if evaluation["objective"]!="task_CE" or evaluation["state_mode"]!="eval_fixed_buffers":
                raise ValueError("incompatible baseline evaluation")
            lines.append(f"- {run['method']}, seed {run['seed']}: validation CE={evaluation['task_CE']}, accuracy={evaluation['accuracy']}; updates={run['updates']}.")
        rows=record.get("rows",[])
        if rows:
            validate_labels(rows)
            if not {"archive_sha256","split_sha256"}.issubset(record.get("data",{})):raise ValueError("missing probe data fingerprints")
            lines.append(f"Candidate rows: {len(rows)}; fixed state `{rows[0]['state_mode']}`. Network inner solver values remain proxies.")
        lines.append("")
    lines.extend(["## Interpretation gates","","Hypothesis -> implementation -> unit validation -> observations -> counterexamples -> limitations -> next authorization.",
        "A: compare residual-informed ranking vs observed direction; fail if benefit is unstable or too costly.",
        "B: compare DeltaL, DeltaS and DeltaF; fail if robust proxy does not improve candidate-target agreement.",
        "C: compare to matched-backward recovery; fail if compensation benefit disappears at matched budget.",
        "Formal accuracy/generalization claims require separate multi-seed evidence and uncertainty."])
    return "\n".join(lines)+"\n"


def summarize_directory(directory,output):
    paths=sorted(Path(directory).glob("gpu_*.json"))
    records=[json.loads(path.read_text(encoding="utf-8")) for path in paths]
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    Path(output).write_text(report_records(records),encoding="utf-8")
    return len(records)
