"""Strict audited artifact contract; V1 artifacts are not silently migrated."""
from __future__ import annotations
import hashlib
import json

from .repro import environment, file_hash, source_hashes, tensor_hash, validate_execution_record

ARTIFACT_SCHEMA = 2
ENVIRONMENT_KEYS = ("python", "platform", "packages", "cuda_runtime", "gpu", "cuda_capabilities", "deterministic", "execution")


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def model_identity(state):
    return {"architecture": state["config"]["model"], "state_sha256": tensor_hash(state["model"]),
            "tensors": {name: {"shape": list(t.shape), "dtype": str(t.dtype)}
                        for name, t in sorted(state["model"].items())}}


def seal_artifact(state):
    state["schema"] = ARTIFACT_SCHEMA
    state["model_identity"] = model_identity(state)
    state["config_sha256"] = json_hash(state["config"])
    state["masks_sha256"] = tensor_hash(state["masks"]) if state.get("masks") is not None else None
    return state


def verify_artifact(state, root, manifest=None, *, check_environment=False):
    if state.get("schema") != ARTIFACT_SCHEMA:
        raise ValueError("Audited artifact schema 2 required; V1 is retained without automatic migration")
    if state.get("source_hashes") != source_hashes(root):
        raise ValueError("Artifact source/config/scripts identity mismatch; postprocessing refused")
    if state.get("config_sha256") != json_hash(state["config"]):
        raise ValueError("Artifact config identity mismatch")
    validate_execution_record(state["config"], state.get("environment", {}).get("execution"))
    if state.get("model_identity") != model_identity(state):
        raise ValueError("Artifact model identity mismatch")
    masks_hash = tensor_hash(state["masks"]) if state.get("masks") is not None else None
    if state.get("masks_sha256") != masks_hash:
        raise ValueError("Artifact mask identity mismatch")
    if manifest is not None and state["manifest"] != manifest:
        raise ValueError("Artifact dataset/split identity mismatch")
    if check_environment:
        current = environment(state["config"])
        for key in ENVIRONMENT_KEYS:
            if state["environment"][key] != current[key]:
                raise ValueError(f"Artifact environment mismatch: {key}")


def artifact_reference(state, path):
    """Bind JSON evidence to a specific checkpoint and its declared identities."""
    return {"checkpoint_sha256": file_hash(path), "stage": state["stage"],
            "model_identity": state["model_identity"], "config_sha256": state["config_sha256"],
            "masks_sha256": state["masks_sha256"], "source_hashes": state["source_hashes"],
            "environment": state["environment"], "dataset_fingerprint": state["manifest"]["fingerprint"]}


def verify_evaluation(result, state, path, split):
    if result.get("split") != split or result.get("provenance") != artifact_reference(state, path):
        raise ValueError("Evaluation provenance/split mismatch")
    if result.get("checkpoint_sha256") != file_hash(path):
        raise ValueError("Evaluation/checkpoint identity mismatch")
