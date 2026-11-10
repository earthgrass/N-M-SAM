"""Read-only audit of saved APBR and native GPT masks on each Dense parent."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
OUT = ROOT / "docs/research/apbr_gpt_mask_compensation_match_20261103"


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha(values: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


PAIRS = {
    "apbr_vpr_seed0": {
        "parent": ART / "cifar100_50k10k_final_v1/seed0_remote_copy/apbr_sam/final_epoch_200.pt",
        "raw": ART / "cifar100_50k10k_final_v1/seed0_remote_copy/methods/apbr_sam/raw_hard.pt",
        "gpt": ART / "apbr_weights_original_sparsegpt_seed012/remote_seed01_copy/artifacts/seed0/apbr_sam_original_sparsegpt/mask.pt",
        "status": ART / "apbr_weights_original_sparsegpt_seed012/remote_seed01_copy/artifacts/seed0/apbr_sam_original_sparsegpt/status.json",
    },
    "apbr_l2_seed0": {
        "parent": ART / "apbr_fixed_l2_seed0_20261102/remote_final_copy/artifacts/run_v1/apbr_fixed_l2/final_epoch_200.pt",
        "raw": ART / "apbr_fixed_l2_seed0_20261102/remote_final_copy/artifacts/run_v1/geometry/raw_hard.pt",
        "gpt": ART / "apbr_fixed_l2_seed0_20261102/remote_final_copy/artifacts/run_v1/original_sparsegpt/mask.pt",
        "status": ART / "apbr_fixed_l2_seed0_20261102/remote_final_copy/artifacts/run_v1/original_sparsegpt/status.json",
    },
}


def audit_pair(paths: dict[str, Path]) -> dict:
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    hashes = {key: file_sha(path) for key, path in paths.items()}
    status = json.loads(paths["status"].read_text(encoding="utf-8"))
    raw = torch.load(paths["raw"], map_location="cpu", weights_only=False)
    gpt = torch.load(paths["gpt"], map_location="cpu", weights_only=False)
    parent = torch.load(paths["parent"], map_location="cpu", weights_only=False)
    apbr = raw["masks"]
    if hashes["parent"] != raw["parent"]["sha256"] or hashes["parent"] != status["parent_sha256"]:
        raise ValueError("Dense parent identity mismatch")
    if hashes["gpt"] != status["mask_file_sha256"]:
        raise ValueError("GPT mask file SHA256 mismatch")
    if tensor_sha(apbr) != raw["mask_sha256"] or tensor_sha(gpt) != status["mask_tensor_sha256"]:
        raise ValueError("Mask tensor SHA256 mismatch")
    if set(apbr) != set(gpt) or len(apbr) != 19:
        raise ValueError("Eligible-layer mismatch")
    if not all(torch.equal(raw["model"][key], parent["model"][key]) for key in parent["model"] if key not in apbr):
        raise ValueError("Raw Hard changed an excluded weight or BN buffer")
    layers = []
    for key in sorted(apbr):
        a, b = apbr[key], gpt[key]
        w = parent["model"][key]
        if a.dtype != torch.bool or b.dtype != torch.bool or a.shape != b.shape or a.shape != w.shape:
            raise ValueError(f"Mask shape/dtype mismatch at {key}")
        if not torch.equal(raw["model"][key], w * a):
            raise ValueError(f"APBR Raw Hard is not exact Mask-only projection at {key}")
        af, bf = a.reshape(a.shape[0], -1), b.reshape(b.shape[0], -1)
        n = af.shape[1] // 4 * 4
        for mask in (af, bf):
            if not bool((mask[:, :n].reshape(-1, 4).sum(dim=1) == 2).all()) or not bool(mask[:, n:].all()):
                raise ValueError(f"Non-2:4 or non-dense tail at {key}")
        same_groups = (af[:, :n].reshape(-1, 4) == bf[:, :n].reshape(-1, 4)).all(dim=1)
        layers.append({
            "layer": key,
            "positions": a.numel(),
            "groups": same_groups.numel(),
            "same_positions": int((a == b).sum()),
            "same_groups": int(same_groups.sum()),
            "apbr_deleted": int((~a).sum()),
            "gpt_deleted": int((~b).sum()),
        })
    totals = {key: sum(row[key] for row in layers) for key in ("positions", "groups", "same_positions", "same_groups", "apbr_deleted", "gpt_deleted")}
    if (totals["positions"], totals["apbr_deleted"], totals["gpt_deleted"]) != (11157504, 5578752, 5578752):
        raise ValueError("Frozen 2:4 counts mismatch")
    totals["same_position_fraction"] = totals["same_positions"] / totals["positions"]
    totals["same_group_fraction"] = totals["same_groups"] / totals["groups"]
    return {
        "status": "LOCAL_MASK_IDENTITY_PASS",
        "protocol_id": raw["protocol_id"],
        "parent_sha256": hashes["parent"],
        "apbr_mask_tensor_sha256": tensor_sha(apbr),
        "gpt_mask_tensor_sha256": tensor_sha(gpt),
        "file_sha256": hashes,
        "totals": totals,
        "layers": layers,
        "test_accessed": False,
        "fixed_mask_obs_full_network_run": False,
    }


def main() -> None:
    results = {name: audit_pair(paths) for name, paths in PAIRS.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    evidence = OUT / "mask_identity_overlap.json"
    evidence.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# APBR 与原生 GPT Mask 本地审计", "", "两组对照各自使用完全相同的 Dense 父模型；本审计只读取历史文件，不访问数据集或 Test。", "", "| 父模型 | 相同位置 | 相同 2:4 组 |", "|---|---:|---:|"]
    for name, result in results.items():
        t = result["totals"]
        lines.append(f"| {name} | {t['same_positions']:,}/{t['positions']:,} ({t['same_position_fraction']:.2%}) | {t['same_groups']:,}/{t['groups']:,} ({t['same_group_fraction']:.2%}) |")
    lines += [
        "", "所有 19 层均通过严格 2:4、Dense 父模型 SHA256、Mask 张量 SHA256、GPT Mask 文件 SHA256 和 APBR Raw Hard 精确投影检查。",
        "每套 Mask 都删除 5,578,752 个位置。",
        "", "本地 CPU 数值门禁：`python -m pytest -q tests/test_fixedmask_obs.py --basetemp=artifacts/pytest-temp-apbr-gpt-mask-local`，6 passed。",
        "覆盖外部固定 Mask 不变、删除位置保持零、幸存权重得到补偿、解析二次解、局部重建误差和 Conv2d unfold Hessian。",
        "", "已有历史 Test 结果仅作诊断：原版 APBR 同一 Dense 父模型上的 APBR Mask-only 为 59.22%，GPT Mask-only 为 48.66%；APBR Mask + Fixed-Mask OBS 为 76.37%。",
        "GPT Native 采用动态选 Mask 与原生顺序补偿，不能直接与 APBR Fixed-Mask OBS 当作同求解器的 Mask 质量对照。",
        "", "| 同一 Dense 父模型 | APBR Mask-only | GPT Mask-only | APBR Mask + Fixed-Mask OBS | GPT Mask + Fixed-Mask OBS | GPT Native（参考） |",
        "|---|---:|---:|---:|---:|---:|",
        "| 原版 APBR Seed 0 | 59.22% | 48.66% | 76.37% | **未运行** | 76.99% |",
        "| APBR-L2 Seed 0 | 56.32% | 未核验 | 未运行 | **未运行** | 77.12% |",
        "", "`GPT Native` 是原生动态选 Mask 与补偿的历史结果，不能填充 `GPT Mask + Fixed-Mask OBS` 列。",
        "本机 PyTorch 为 `2.13.0+cpu`，`torch.cuda.is_available() == False`；尚无可用 GPU 服务器，因此不能对未运行列报告准确率。",
        "", "这些重合率不能单独证明 Mask 与 OBS 的匹配程度。完整同父模型、同固定 Mask OBS 求解器对照尚未运行，需 GPU。",
        "", f"机器可读逐层结果：`{evidence.name}`。", "",
    ]
    (OUT / "LOCAL_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    manifest = OUT / "SHA256SUMS.txt"
    manifest.write_text("".join(f"{file_sha(path)}  {path.name}\n" for path in (evidence, OUT / "LOCAL_AUDIT.md")), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
