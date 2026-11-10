"""Frozen sequential 200-epoch campaign. No tuning, warm starts or automatic retries."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

from vpr_sam.config import comparison_protocol, load_config, method_protocol, validate
from vpr_sam.repro import atomic_json, file_hash, source_hashes

SEEDS = (0, 1, 2)
METHODS = ("sam", "vpr")
CAMPAIGN = "runs/main200_seeds012_v1"
CONFIGS = "configs/approved_main200"
STEPS = 200 * math.ceil(45000 / 128)
MIN_START_BYTES = 20 * 1024**3
MIN_RUN_BYTES = 4 * 1024**3


def configs(root):
    result = {}
    for seed in SEEDS:
        for method in METHODS:
            c = load_config(root / CONFIGS / f"{method}_seed{seed}.json")
            validate(c, formal_allowed=True)
            if c["seed"] != seed or c["optimizer"]["method"] != method:
                raise ValueError("Campaign seed/method mismatch")
            if c["output"] != f"{CAMPAIGN}/{method}_seed{seed}":
                raise ValueError("Campaign output mismatch")
            result[seed, method] = c
        if comparison_protocol(result[seed, "sam"]) != comparison_protocol(result[seed, "vpr"]):
            raise ValueError("SAM/VPR common protocol mismatch")
    for method in METHODS:
        if any(method_protocol(result[s, method]) != method_protocol(result[0, method]) for s in SEEDS):
            raise ValueError("Across-seed method protocol mismatch")
    return result


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def frozen(root, commit, hashes=None):
    if git(root, "rev-parse", "HEAD") != commit or git(root, "status", "--porcelain"):
        raise RuntimeError("Campaign requires the approved clean Git commit")
    current = source_hashes(root)
    if hashes is not None and current != hashes:
        raise RuntimeError("Frozen source/config/script hashes changed")
    return current


def gpu_inventory(uuid):
    raw = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True)
    rows = [[v.strip() for v in line.split(",")] for line in raw.strip().splitlines()]
    if len(rows) != 1 or rows[0][0] != "0" or rows[0][2] != uuid or "RTX 5090" not in rows[0][1]:
        raise RuntimeError("Approved single physical RTX 5090/UUID mismatch")
    return rows[0]


def check_pair(states, expected):
    for state in states:
        if state["epoch"] != 200 or state["next_batch"] != 0 or state["stepper"]["steps"] != STEPS:
            raise ValueError("Exact 200-epoch budget not completed")
        if len(state["epoch_rows"]) != 200 or len(state["step_rows"]) != STEPS:
            raise ValueError("Incomplete epoch/step history")
        for key in ("initialization_hash", "order_hash"):
            if state[key] != expected[key]:
                raise ValueError(f"Preregistered {key} mismatch")
        if state["manifest"]["fingerprint"] != expected["dataset_fingerprint"]:
            raise ValueError("Preregistered dataset/split mismatch")
    for key in ("initialization_hash", "order_hash", "manifest", "source_hashes", "environment"):
        if states[0][key] != states[1][key]:
            raise ValueError(f"Dense pair mismatch: {key}")


def gate(root, commit, uuid):
    import torch
    from vpr_sam.data import build_data
    from vpr_sam.models import build_model
    from vpr_sam.repro import environment, seed_all, seed_for, tensor_hash

    hashes = frozen(root, commit)
    cs = configs(root)
    if shutil.disk_usage(root).free < MIN_START_BYTES:
        raise RuntimeError("Less than 20 GiB free before campaign")
    inventory = gpu_inventory(uuid)
    active = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
    if active:
        raise RuntimeError("GPU already has compute processes; refusing to compete")
    if not Path(sys.prefix).resolve().is_relative_to(root):
        raise RuntimeError("Python environment must belong to this standalone project")
    import vpr_sam
    if not Path(vpr_sam.__file__).resolve().is_relative_to(root):
        raise RuntimeError("Imported source is outside the project")
    torch.set_num_threads(4)
    seed_all(0, True)
    env = environment(cs[0, "sam"])
    if env["execution"]["gpu_uuid"].removeprefix("GPU-") != uuid.removeprefix("GPU-"):
        raise RuntimeError("Logical cuda:0 does not resolve to the approved UUID")
    x = torch.ones(32, 32, device="cuda:0")
    if float((x @ x)[0, 0]) != 32:
        raise RuntimeError("CUDA kernel failed")
    del x
    identities = {}
    for seed in SEEDS:
        c = cs[seed, "sam"]
        for method in METHODS:
            if (root / cs[seed, method]["output"]).exists():
                raise FileExistsError("Fresh campaign refuses existing run directories")
        seed_all(seed_for(seed, "initialization"), True)
        datasets, manifest = build_data(c, root)
        if len(datasets["train"]) != 45000 or len(datasets["validation"]) != 5000:
            raise ValueError("CIFAR split sizes differ")
        model = build_model(c["model"])
        initialization = tensor_hash(model.state_dict())
        d = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if d != 11173962 or any(p.dtype != torch.float32 for p in model.parameters()):
            raise ValueError("Unexpected model dimension/dtype")
        del model
        order_hash = hashlib.sha256(b"vpr-order-v1").hexdigest()
        prefix = {}
        for epoch in range(200):
            gen = torch.Generator().manual_seed(seed_for(seed, "order", epoch))
            order = torch.randperm(45000, generator=gen).numpy()
            ids = datasets["train"].indices[order]
            for offset in range(0, 45000, 128):
                order_hash = hashlib.sha256(bytes.fromhex(order_hash) + ids[offset:offset+128].tobytes()).hexdigest()
            if epoch == 19:
                prefix["recovery_order_hash"] = order_hash
        identities[str(seed)] = {
            "initialization_hash": initialization, "order_hash": order_hash, **prefix,
            "dataset_fingerprint": manifest["fingerprint"], "d": d,
            "probe_validation_ids": manifest["validation_indices"][:2560],
            "probe_rule": "first 10 non-shuffled validation batches, batch size 256, rho 0.02"}
        atomic_json(root / CAMPAIGN / "control" / f"data_manifest_seed{seed}.json", manifest)
    result = {"commit": commit, "source_hashes": hashes, "environment": env,
              "approved_gpu_uuid": uuid, "gpu_inventory": inventory,
              "identities": identities, "free_bytes": shutil.disk_usage(root).free,
              "config_sha256": {f"{m}_seed{s}": file_hash(root / CONFIGS / f"{m}_seed{s}.json")
                                for s in SEEDS for m in METHODS},
              "time": time.time(), "authorized_seeds": list(SEEDS), "steps_per_run": STEPS}
    atomic_json(root / CAMPAIGN / "control" / "gates.json", result)
    print("PASS one-time gates: six configs, clean commit, UUID, full data, initialization/order/probe identities", flush=True)


def audit_pair(root, seed):
    import torch
    from vpr_sam.training.engine import read_checkpoint
    from vpr_sam.models import build_model
    from vpr_sam.optim.geometry import Geometry

    gates = json.loads((root / CAMPAIGN / "control/gates.json").read_text(encoding="utf-8"))
    expected = gates["identities"][str(seed)]
    states = []
    records = []
    torch.set_num_threads(4)
    for method in METHODS:
        path = root / CAMPAIGN / f"{method}_seed{seed}/final.pt"
        state, _ = read_checkpoint(path, root)
        summary = json.loads((path.parent / "summary.json").read_text(encoding="utf-8"))
        if not summary["complete"] or summary["checkpoint_sha256"]["final"] != file_hash(path):
            raise ValueError("Final checkpoint publication/summary mismatch")
        if state["stage"] != "train" or state["config"]["optimizer"]["method"] != method:
            raise ValueError("Unexpected dense checkpoint stage/method")
        if state["config"] != configs(root)[seed, method] or state["parent"] is not None:
            raise ValueError("Fresh-run config/parent mismatch")
        if state["scheduler"]["last_epoch"] != 200 or state["scheduler"]["T_max"] != 200:
            raise ValueError("Cosine schedule horizon/completion mismatch")
        geometry = Geometry(build_model("resnet18").named_parameters(), state["config"]["geometry"], method)
        geometry.load_state_dict(state["stepper"]["geometry"])
        diagnostics = geometry.diagnostics()
        if method == "vpr" and geometry.updates + geometry.zero_skips != STEPS:
            raise ValueError("Geometry update count mismatch")
        states.append(state)
        records.append({"method": method, "sha256": file_hash(path), "summary": summary,
                        "geometry": diagnostics})
    check_pair(states, expected)
    atomic_json(root / CAMPAIGN / "control" / f"pair_seed{seed}_audit.json", records)


def audit_recovery(root, seed, method):
    from vpr_sam.config import recovery_config
    from vpr_sam.training.engine import read_checkpoint
    run = root / CAMPAIGN / f"{method}_seed{seed}"
    hard = run / "pruning/n2_m4_exclude_first_last_magnitude_dense_final/pruned.pt"
    final = hard.parent / "recovery/final.pt"
    state, _ = read_checkpoint(final, root)
    gates = json.loads((root / CAMPAIGN / "control/gates.json").read_text(encoding="utf-8"))
    expected = gates["identities"][str(seed)]
    if (state["stage"] != "recovery" or state["epoch"] != 20 or state["next_batch"] != 0
            or state["stepper"]["steps"] != 7040 or len(state["step_rows"]) != 7040
            or len(state["epoch_rows"]) != 20 or state["scheduler"]["last_epoch"] != 20
            or state["scheduler"]["T_max"] != 20):
        raise ValueError("Exact recovery budget/scheduler mismatch")
    if state["config"] != recovery_config(configs(root)[seed, method]):
        raise ValueError("Recovery optimizer/protocol mismatch")
    if (state["order_hash"] != expected["recovery_order_hash"]
            or state["initialization_hash"] != expected["initialization_hash"]
            or state["manifest"]["fingerprint"] != expected["dataset_fingerprint"]
            or state["parent"]["sha256"] != file_hash(hard) or state["masks"] is None):
        raise ValueError("Recovery order/initialization/data/mask/parent mismatch")
    atomic_json(root / CAMPAIGN / "control" / f"{method}_seed{seed}_recovery_audit.json",
                {"checkpoint_sha256": file_hash(final), "order_hash": state["order_hash"],
                 "epochs": 20, "steps": 7040, "parent": state["parent"], "schema": state["schema"]})


def run_campaign(root, commit, uuid):
    control = root / CAMPAIGN / "control"
    control.mkdir(parents=True, exist_ok=True)
    lock = control / "campaign.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(descriptor, str(os.getpid()).encode())
    os.close(descriptor)
    hashes = frozen(root, commit)
    cs = configs(root)
    env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONHASHSEED="0", VPR_FORMAL_APPROVED="YES")
    active = None

    def launch(name, args, run=None):
        nonlocal active
        frozen(root, commit, hashes)
        gpu_inventory(uuid)
        if shutil.disk_usage(root).free < MIN_RUN_BYTES:
            raise RuntimeError("Disk reserve below 4 GiB")
        status = {"state": "running", "stage": name, "time": time.time(), "controller_pid": os.getpid()}
        atomic_json(control / "status.json", status)
        print(f"START {name}", flush=True)
        with (control / f"{name}.log").open("x", encoding="utf-8") as log:
            active = subprocess.Popen([sys.executable, *args], cwd=root, env=env,
                                      stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            status["child_pid"] = active.pid
            atomic_json(control / "status.json", status)
            while True:
                code = active.poll()
                if code is not None:
                    active = None
                    if code != 0:
                        raise RuntimeError(f"Stage {name} failed with exit {code}; see stage log")
                    break
                telemetry = {**status, "time": time.time(), "free_bytes": shutil.disk_usage(root).free,
                             "gpu": gpu_inventory(uuid)}
                if run is not None and (run / "epochs.json").exists():
                    rows = json.loads((run / "epochs.json").read_text(encoding="utf-8"))
                    telemetry["epochs_completed"] = len(rows)
                    telemetry["latest_epoch"] = rows[-1] if rows else None
                    steps_file = run / "steps.json"
                    if steps_file.exists():
                        steps = json.loads(steps_file.read_text(encoding="utf-8"))
                        telemetry["latest_step"] = steps[-1] if steps else None
                        early = [s for s in steps if s["epoch"] < 10]
                        telemetry["first10_monitor"] = {
                            "observed_steps": len(early),
                            "max_abs_delta_s": max((s["max_abs_delta_s"] for s in early), default=0),
                            "max_abs_log_volume": max((abs(s["log_volume"]) for s in early), default=0)}
                with (control / "telemetry.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(telemetry, allow_nan=False) + "\n")
                    stream.flush()
                atomic_json(control / "latest_telemetry.json", telemetry)
                if telemetry["free_bytes"] < MIN_RUN_BYTES:
                    raise RuntimeError("Disk reserve fell below 4 GiB; stopping active stage")
                time.sleep(15)
        frozen(root, commit, hashes)
        print(f"PASS {name}", flush=True)

    def interrupted(signum, frame):
        raise RuntimeError(f"Controller received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        launch("gates", ["scripts/formal_main200.py", "--gate-only", "--commit", commit, "--gpu-uuid", uuid])
        all_runs = []
        for seed in SEEDS:
            for method in METHODS:
                c = cs[seed, method]
                launch(f"{method}_seed{seed}_train", ["-m", "vpr_sam", "train", "--config",
                       f"{CONFIGS}/{method}_seed{seed}.json", "--allow-formal"], root / c["output"])
            launch(f"seed{seed}_pair_audit", ["scripts/formal_main200.py", "--audit-pair", str(seed)])
            for method in METHODS:
                run = cs[seed, method]["output"]
                final = f"{run}/final.pt"
                pruned = f"{run}/pruning/n2_m4_exclude_first_last_magnitude_dense_final/pruned.pt"
                recovery = str(Path(pruned).parent / "recovery/final.pt")
                launch(f"{method}_seed{seed}_prune", ["-m", "vpr_sam", "prune", "--checkpoint", final])
                launch(f"{method}_seed{seed}_recover", ["-m", "vpr_sam", "recover", "--checkpoint", pruned,
                       "--allow-formal"], root / Path(recovery).parent)
                launch(f"{method}_seed{seed}_recovery_audit", ["scripts/formal_main200.py",
                       "--audit-recovery", str(seed), "--method", method])
                for label, path in (("dense", final), ("hard", pruned), ("recovered", recovery)):
                    launch(f"{method}_seed{seed}_probe_{label}", ["-m", "vpr_sam", "evaluate",
                           "--checkpoint", path, "--split", "validation"])
                all_runs.append(run)
            output = f"{CAMPAIGN}/analysis_through_seed{seed}"
            launch(f"summary_seed{seed}", ["-m", "vpr_sam", "summarize", "--runs", *all_runs,
                   "--output", output, "--checkpoint", "final", "--recovery-checkpoint", "final"])
            launch(f"plot_seed{seed}", ["-m", "vpr_sam", "plot", "--summary", f"{output}/summary.json",
                   "--output", f"{output}/figures"])
        atomic_json(control / "status.json", {"state": "complete", "commit": commit, "time": time.time()})
    except BaseException as exc:
        if active is not None and active.poll() is None:
            os.killpg(active.pid, signal.SIGTERM)
            try:
                active.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(active.pid, signal.SIGKILL)
                active.wait()
        atomic_json(control / "failure.json", {"type": type(exc).__name__, "message": str(exc),
                    "traceback": traceback.format_exc(), "time": time.time(), "automatic_retry": False})
        atomic_json(control / "status.json", {"state": "failed", "time": time.time(), "reason": str(exc)})
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--gate-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--audit-pair", type=int, choices=SEEDS)
    mode.add_argument("--audit-recovery", type=int, choices=SEEDS)
    p.add_argument("--method", choices=METHODS)
    p.add_argument("--commit")
    p.add_argument("--gpu-uuid")
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.audit_recovery is not None:
        if args.method is None:
            p.error("Recovery audit requires method")
        audit_recovery(root, args.audit_recovery, args.method)
    elif args.audit_pair is not None:
        audit_pair(root, args.audit_pair)
    else:
        if not args.commit or not args.gpu_uuid:
            p.error("Explicit approved commit and physical GPU UUID required")
        if args.gate_only:
            gate(root, args.commit, args.gpu_uuid)
        else:
            if os.environ.get("VPR_FORMAL_APPROVED") != "YES":
                p.error("Explicit VPR_FORMAL_APPROVED=YES required")
            run_campaign(root, args.commit, args.gpu_uuid)


if __name__ == "__main__":
    main()
