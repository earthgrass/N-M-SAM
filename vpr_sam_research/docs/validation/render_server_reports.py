"""Render public reports from local raw acceptance evidence (no training).

Usage: python docs/validation/render_server_reports.py --raw artifacts/server_session/raw/server_acceptance
Raw UUIDs stay local. Public JSON is evidence, never a replacement checkpoint.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from vpr_sam.repro import source_hashes

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT/"docs"
PUBLIC = DOCS/"validation/server_acceptance"
BASELINE = "ee57bc68b9a2f5f71dffb067f67416a9461f8db1"
MEASURED = "bf4f123e01a9b0819376a4def7d2e82e5c56975d"


def read(path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def redact(value):
    if isinstance(value, dict):
        return {k:redact(v) for k,v in value.items()}
    if isinstance(value, list): return [redact(v) for v in value]
    if isinstance(value, str):
        value = re.sub(r"GPU-[a-fA-F0-9-]{30,}", lambda m:"GPU-REDACTED-"+hashlib.sha256(m[0][4:].encode()).hexdigest()[:12], value)
        return re.sub(r"\b[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\b",
                      lambda m:"UUID-REDACTED-"+hashlib.sha256(m[0].encode()).hexdigest()[:12], value)
    return value


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")


def write_csv(path, rows):
    if not rows: return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--raw", type=Path, required=True)
    raw = parser.parse_args().raw
    PUBLIC.mkdir(parents=True, exist_ok=True)
    facts = {name:read(raw/f"{name}.json", {}) for name in ("kernels","math","data","smoke","gamma_estimate")}
    for name in ("kernels", "math", "data", "smoke"):
        if facts[name].get("status") == "PASS":
            assert facts[name]["source_hashes"] == source_hashes(ROOT), f"Different source/config set: {name}"
    final_state = read(raw/"final_state.json", {})
    assert final_state.get("commit") == MEASURED and final_state.get("working_tree") == ""
    assert final_state.get("source_hashes") == source_hashes(ROOT)
    for name in ("kernels","math","data","smoke","real_resume","real_resume_geometry","real_pruning","recovery_resume","underflow_audit","package_sources","hardware_raw","device_properties_raw","final_state","epoch_boundary_failure_9b6c02d","early_synthetic","geometry_microbenchmark_deterministic_raw"):
        data = read(raw/f"{name}.json")
        if data is not None: write_json(PUBLIC/f"{name}.json", {"public_redacted_evidence":True, "evidence":redact(data)})
    for path in raw.glob("*failure.json"):
        write_json(PUBLIC/path.name, redact(read(path)))
    if (raw/"pytest.xml").exists(): (PUBLIC/"server_pytest.xml").write_bytes((raw/"pytest.xml").read_bytes())
    if (raw/"resources_raw.txt").exists():
        (PUBLIC/"resources_redacted.txt").write_text(redact((raw/"resources_raw.txt").read_text()), encoding="utf-8")
    for path in (raw/"workflow_evidence").rglob("*.json"):
        write_json(PUBLIC/"workflow_evidence"/path.relative_to(raw/"workflow_evidence"),
                   {"public_redacted_evidence":True,"evidence":redact(read(path))})
    hardware = read(raw/"hardware_raw.json", {})
    core = {name:hashlib.sha256((ROOT/"src/vpr_sam/optim"/name).read_bytes()).hexdigest() for name in ("geometry.py","step.py")}
    assert core == {"geometry.py":"8e0e6d42bcf34a7a8c3b79dc4b76081e9d8c0d6313b183ce0f59109bd39058b5",
                    "step.py":"3c022f2815d577ef3d94ffeed40cb93c06e2359a2734c857a06f1a0c1f738ac2"}
    performance = []
    for folder in sorted(raw.glob("performance_*")):
        result = read(folder/"result.json")
        failure = read(folder/"failure.json")
        if result:
            rows = result["records"][2:]
            performance.append({"method":result["config"]["optimizer"]["method"], "dtype":result["config"]["geometry"]["dtype"],
                "status":"PASS", "warmup_discarded":2, "measured_steps":len(rows), "gamma":result["config"]["geometry"]["gamma"],
                "step_seconds_instrumented":statistics.mean(r["step_seconds_instrumented"] for r in rows),
                "forward_seconds":statistics.mean(r["forward_seconds"] for r in rows),
                "backward_loss_seconds":statistics.mean(r["backward_and_loss_seconds"] for r in rows),
                "perturbation_seconds":statistics.mean(r["perturbation_seconds"] for r in rows),
                "feedback_seconds":statistics.mean(r.get("feedback_seconds",0.)+r.get("commit_seconds",0.) for r in rows),
                "data_loader_seconds":statistics.mean(r["data_loader_seconds"] for r in rows),
                "peak_cuda_bytes":result["peak_cuda_bytes"], "checkpoint_seconds":result["checkpoint_seconds"],
                "checkpoint_bytes":result["checkpoint_bytes"], "max_abs_volume":max(abs(r["log_volume"]) for r in rows),
                "max_ellipsoid_error":max(abs(r["ellipsoid_norm"]-.05) for r in rows)})
        elif failure:
            performance.append({"method":failure["config"]["optimizer"]["method"],"dtype":failure["config"]["geometry"]["dtype"],
                                "status":"FAIL", "failure":failure["message"],"successful_steps":failure["step"]})
            write_json(PUBLIC/f"{folder.name}_failure.json", failure)
    unobserved = read(raw/"uninstrumented_performance.json", {})
    plain_table = ["|无逐步观测器的原生engine|step ms|峰值 GiB|schema2 checkpoint MiB|状态捕获+seal+写盘 s|", "|---|---:|---:|---:|---:|"]
    for row in unobserved.get("results", []):
        captured = sum(sum(values) for values in row["checkpoint_component_seconds"].values())
        plain_table.append(f"|{row['method']}|{row['mean_step_seconds']*1000:.2f}|{row['peak_cuda_bytes']/2**30:.3f}|{row['checkpoint_bytes']/2**20:.1f}|{captured:.3f}|")
    perf_public = {"measured_source_commit":MEASURED,"instrumented":performance,"uninstrumented":redact(unobserved),
                   "instrumented_scope":"synchronized wall; backward includes cross entropy; observer overhead is included in total",
                   "microbenchmark":"auxiliary only; existing microbenchmark tolerance is NOT the main geometry tolerance"}
    write_json(DOCS/"GPU_PERFORMANCE.json", perf_public)
    table = ["|方法/状态|step ms（含观测）|前向 ms|反向+loss ms|扰动 ms|反馈+commit ms|峰值 GiB|保存 s / MiB|", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for p in performance:
        label = f"{p['method']} {p['dtype']} {p['status']}"
        if p["status"] != "PASS": table.append(f"|{label}|—|—|—|—|—|—|—|"); continue
        table.append(f"|{label}|{p['step_seconds_instrumented']*1000:.2f}|{p['forward_seconds']*1000:.2f}|{p['backward_loss_seconds']*1000:.2f}|{p['perturbation_seconds']*1000:.2f}|{p['feedback_seconds']*1000:.2f}|{p['peak_cuda_bytes']/2**30:.3f}|{p['checkpoint_seconds']:.3f} / {p['checkpoint_bytes']/2**20:.1f}|")
    (DOCS/"GPU_PERFORMANCE.md").write_text("# RTX 5090 性能与精度验收\n\n"+"\n".join(table)+
        "\n\n"+"\n".join(plain_table)+
        "\n\n真实 CIFAR-10 / ResNet18，batch128，FP32模型，固定gamma=1e-7（仅性能夹具），4个intra-op线程、2个workers；前2步预热不计入均值。每个精度/方法在独立进程中测量，避免前一个模型分配污染峰值。\n\n"
        "SAM不维护log-radius状态，表中float64只是共同配置的占位标签；两方法模型均为FP32，SAM扰动仍使用原有FP64归约。\n\n"
        "观测总时间包含同步、hooks及诊断归约；反向时间包含loss计算，不能作为纯CUDA kernel时间。DataLoader数字是主进程等待时间，不能解释为全部CPU增强成本。原生engine对照同初始化/顺序，VPR步时开销约60%；checkpoint组件计时包含state捕获、seal及序列化，不覆盖save闭包的全部JSON与字典构造时间。诊断checkpoint不是schema2训练checkpoint，保存时间/大小与原生产物分列。长程step_rows历史增长会增加checkpoint成本，12步结果不能代表末期文件大小。\n\n"
        "精度优先推荐FP64 log-radius；任何FP32体积门禁失败均停止该配置，不放宽容差。FP64、FP32状态本体分别为d×8、d×4字节；峰值使用torch.cuda.max_memory_allocated，包含模型、梯度、SGD momentum、激活及临时FP64张量，但不包含全部驱动/context/NVML显存。辅助单向量microbenchmark采用原有独立夹具容差，不能替代真实训练门禁。\n\n"
        +"\n".join(f"- {p['method']}/{p['dtype']}: {p.get('failure','volume max='+str(p.get('max_abs_volume'))+', ellipse max error='+str(p.get('max_ellipsoid_error')))}" for p in performance)+
        "\n\n未验证独占调度或长时间热稳定性，短测不外推为200 epoch资源保证。机器可读数据见 GPU_PERFORMANCE.json。\n", encoding="utf-8")
    estimate = facts["gamma_estimate"]
    real_resume = read(raw/"real_resume.json", {})
    longer = read(raw/"real_resume_geometry.json", {})
    underflow = read(raw/"underflow_audit.json", {})
    recovery_resume = read(raw/"recovery_resume.json", {})
    real_pruning = read(raw/"real_pruning.json", {})
    for name, data in (("real_resume",real_resume),("real_pruning",real_pruning),("underflow_audit",underflow),("recovery_resume",recovery_resume)):
        if data.get("status") == "PASS": assert data["source_hashes"] == source_hashes(ROOT), name
    candidates, traces = [], []
    for folder in sorted(raw.glob("vpr_candidate_*")):
        result = read(folder/"result.json"); failure = read(folder/"failure.json")
        if result:
            rows = result["records"]
            candidates.append({"tag":folder.name,"status":"PASS","gamma":result["config"]["geometry"]["gamma"],"steps":len(rows),
                "initialization_hash":result["initialization_hash"],"max_delta":max(r["max_abs_delta_s"] for r in rows),
                "min_s":min(r["log_radius_min"] for r in rows),"max_s":max(r["log_radius_max"] for r in rows),
                "final_rms":rows[-1]["log_radius_rms"],"max_volume":max(abs(r["log_volume"]) for r in rows),
                "max_volume_bound_ratio":max(abs(r["log_volume"])/r["volume_bound"] for r in rows),
                "max_ellipse_error":max(abs(r["ellipsoid_norm"]-.05) for r in rows),
                "underflow_coordinates_max":max(r["underflow_coordinates"] for r in rows)})
            keys = ("step","epoch","max_abs_delta_s","log_radius_min","log_radius_max","log_radius_rms","log_volume","volume_bound",
                    "eps_norm","ellipsoid_norm","g1_norm","g2_norm","center_loss","bn_running_var_min","bn_running_var_max","underflow_coordinates","ids_sha256")
            traces.extend({"candidate":folder.name,"gamma":result["config"]["geometry"]["gamma"], **{k:r[k] for k in keys}} for r in rows)
            write_json(PUBLIC/f"{folder.name}_layers.json", [{"step":r["step"],"layer_log_volume":r["layer_log_volume"],"layer_budget":r["layer_budget"]} for r in rows])
        elif failure:
            candidates.append({"tag":folder.name,"status":"FAIL","gamma":failure["config"]["geometry"]["gamma"],"steps":failure["step"],"failure":failure["message"]})
            write_json(PUBLIC/f"{folder.name}_failure.json", failure)
    sam = read(raw/"sam_g1_300/result.json", {})
    energy = sam.get("energy", [])
    if energy:
        write_csv(PUBLIC/"g1_energy.csv", [{k:r[k] for k in ("step","d","zero","nonzero_fraction","g1_norm","c_max","max_abs_one_minus_c")} for r in energy])
        write_json(PUBLIC/"g1_layer_energy.json", [{"step":r["step"],"layer_energy":r["layer_energy"]} for r in energy])
    write_csv(PUBLIC/"candidate_trajectories.csv", traces)
    matched = all([r["ids_sha256"] for r in read(raw/c["tag"]/"result.json",{}).get("records",[])] ==
                  [r["ids_sha256"] for r in sam.get("records",[])][:c["steps"]]
                  and c.get("initialization_hash") == sam.get("initialization_hash") for c in candidates if c["status"] == "PASS")
    recommended = next((c["gamma"] for c in candidates if c["tag"] == "vpr_candidate_0" and c["status"] == "PASS"), None)
    if recommended is not None and not matched: raise AssertionError("Calibration initialization/order mismatch")
    gamma_public = {"estimate":estimate,"candidates":candidates,"paired_initialization_and_order":matched,"recommended_short_run_candidate":recommended,
                    "formal_approval":False,"independent_seeds":1,"validation_or_test_used_for_selection":False}
    write_json(PUBLIC/"gamma_summary.json", gamma_public)
    if energy and traces:
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        axes[0,0].plot([r["step"] for r in energy],[r["c_max"] for r in energy]); axes[0,0].set(title="SAM g1 concentration",ylabel="d max(p)",xlabel="step")
        for c in candidates:
            rows = [r for r in traces if r["candidate"] == c["tag"]]
            if not rows: continue
            x = [r["step"] for r in rows]; label = f"gamma={c['gamma']:.3g}"
            axes[0,1].plot(x,[r["max_abs_delta_s"] for r in rows],label=label)
            axes[1,0].plot(x,[r["log_radius_min"] for r in rows],label=label)
            axes[1,1].plot(x,[r["log_volume"] for r in rows],label=label)
        for ax,title,y in ((axes[0,1],"Maximum geometry update","max |delta s|"),(axes[1,0],"Minimum log radius","min s"),(axes[1,1],"Global volume drift","sum s")):
            ax.set(title=title,xlabel="step",ylabel=y); ax.legend(fontsize=8)
        fig.suptitle("RTX 5090 / real CIFAR-10 / ResNet18 — one diagnostic seed")
        fig.tight_layout(); fig.savefig(PUBLIC/"gamma_trajectories.png",dpi=180); fig.savefig(PUBLIC/"gamma_trajectories.pdf"); plt.close(fig)
    theory = """# 固定 Gamma 数值校准

冻结定义没有变化。固定非零梯度下，p=softmax(4s+2log|g|)。固定点要求p_i=1/d，因此s_i=C−(1/2)log|g_i|；sum(s)=0给出C=(1/(2d))sum(log|g|)，即a_i=(prod|g_j|)^(1/(2d))|g_i|^(−1/2)。

Jacobian为I−4gamma d(diag(p)−ppᵀ)。在均匀固定点为I−4gamma(I−11ᵀ/d)，零和子空间特征值1−4gamma；局部渐近稳定需要0<gamma<1/2。整体缩放方向特征值1，但不属于受约束零和空间。

这只适用于固定、有限且各坐标非零的梯度与精确算术、固定d的固定点邻域。零梯度坐标p=0，每步s增加gamma，通常没有上述有限固定点；全零扰动按核心定义跳过。随机梯度使目标固定点移动，不能推导同一全局收敛或长期稳定性。高维集中使单步最大变化接近gamma(d−1)，局部区间不约束这种远离固定点的大跳变。FP32扰动舍入、log-radius存储与体积归约容差须另行验收。

## 实测与提名

采样源为真实CIFAR-10/ResNet18 Dense-SAM的g1；grad=None计零但保留其坐标计入全局d。300步、一个seed，非独立重复；不用SAM原日志中的零占位d_p_max。公共rho0.05、batch128、SGD lr0.1/momentum0.9/weight decay0.0005、Cutout0；所有候选从同seed随机初始化重新开始，不继承SAM或其他候选权重。短程预算在首个epoch内完成，cosine尚未跨epoch更新；不能代表200 epoch的LR轨迹。

"""
    table = ["|候选|gamma|成功步|最大单步变化|最小s / 最大s|最终RMS|最大体积误差|状态|","|---|---:|---:|---:|---:|---:|---:|---|"]
    for c in candidates:
        table.append(f"|{c['tag']}|{c['gamma']:.8g}|{c['steps']}|{c.get('max_delta','—')}|{c.get('min_s','—')} / {c.get('max_s','—')}|{c.get('final_rms','—')}|{c.get('max_volume','—')}|{c['status']}|")
    details = f"d={estimate.get('d','NOT TESTED')}；c_max分布：`{json.dumps(estimate.get('c_max',{}))}`。\n\n"+f"非零比例：`{json.dumps(estimate.get('nonzero_fraction',{}))}`；全零步数={estimate.get('zero_steps','NOT TESTED')}；梯度范数：`{json.dumps(estimate.get('gradient_norm',{}))}`。\n\n"+f"gamma0=0.05/Q99(max_i|1−c_i|)={estimate.get('gamma0','NOT TESTED')}；分母：`{json.dumps(estimate.get('denominator',{}))}`；P95/max敏感性：`{json.dumps(estimate.get('robustness',{}))}`。0.05只是诊断尺度，不是安全定理。\n\n"
    (DOCS/"GAMMA_CALIBRATION.md").write_text(theory+details+"\n".join(table)+f"\n\n同初始化、同数据顺序核验：{matched}。保守提名固定gamma={recommended}（0.5gamma0）；不按准确率排序。最小候选已有可见几何变化，且最大单步变化约0.057，相比中间候选0.077及最大候选0.215留出更多余量；0.05不是硬阈值，三个候选都没有长期安全保证。SAM上的Q99也不能约束VPR改变轨迹后的集中度，仍需GPT/用户审批。\n\n"
        "数值异常见对应failure.json；失败对象不能保存有效checkpoint，不进行裁剪、重置或自动换gamma。FP32精度失败与FP64候选分开记录。\n\n"
        "![几何轨迹](validation/server_acceptance/gamma_trajectories.png)\n\n"
        "CSV包含逐步c_max、范数、单步变化、s极值/RMS、体积误差/批准界限、L2与椭球范数、BN方差及下溢坐标；JSON包含所有参数张量的层能量与预算迁移，不能把局部层和当逐层守恒。\n\n"
        f"原生100步内观测到极少数非零梯度对应FP32扰动为零（各候选每步最多3/1/2个坐标）。这不等于整次扰动下溢；专项复核按独立FP64公式逐坐标检查半个最小FP32 subnormal阈值：共{len(underflow.get('events',[]))}个张量/步事件，均低于该阈值，单事件丢失能量比例最高{max((r['missing_epsilon_energy_fraction'] for r in underflow.get('events',[])),default=0.)}。三条轨迹模型/优化器/几何逐位重放，见underflow_audit.json。保留这一有限精度现象，不描述为完全无下溢。复核重放不增加独立seed数。\n\n"
        f"推荐候选另在真实数据恢复验收中延长到{longer.get('steps','NOT TESTED')}步（跨一次epoch）：最大单步变化{longer.get('max_delta')}、s范围[{longer.get('min_s')}, {longer.get('max_s')}]、最终RMS={longer.get('final_rms')}、最大体积误差{longer.get('max_abs_volume')}、实际扰动L2范围[{longer.get('eps_l2_min')}, {longer.get('eps_l2_max')}]。这是smoke的2-epoch cosine horizon，非200-epoch调度；验证指标仅用于验收状态一致性，不用于提名gamma。\n\n"
        "这是早期数值可行性证据，不是长期稳定性、精度优势或200 epoch收敛保证。保守gamma在固定梯度、均匀固定点邻域的收敛因子非常接近1；数值安全不能保证反馈足够强或带来科研收益。尚无跨seed确认，也没有长程漂移证据；不运行0.1/0.2/0.3/0.4整段搜索，不实现动态gamma，不使用真实test或validation准确率调参。200 epoch仍须完整协议审批。\n",encoding="utf-8")
    data = facts["data"].get("result", {})
    gate = {"source":"PASS", "core":"PASS", "hardware":facts["kernels"].get("status","NOT TESTED"),
            "cuda_smoke":"PASS" if facts["smoke"].get("status")==real_pruning.get("status")=="PASS" else "FAIL / NOT TESTED","real_data":facts["data"].get("status","NOT TESTED"),
            "exact_resume":"PASS" if all(d.get("status")=="PASS" for d in (facts["smoke"],real_resume,recovery_resume)) else "FAIL / NOT TESTED",
            "gamma":"PASS" if len(candidates)==3 and all(c["status"]=="PASS" for c in candidates) and matched and underflow.get("status")=="PASS" else "FAIL / NOT TESTED",
            "protocol":"PENDING","formal_training":"NOT STARTED"}
    write_json(PUBLIC/"audit_gate.json", {"baseline":BASELINE,"measured_source_commit":MEASURED,"core_sha256":core,"gates":gate,"gamma":gamma_public})
    env = hardware.get("environment", {})
    (DOCS/"SERVER_ACCEPTANCE.md").write_text(f"# RTX 5090 服务器验收\n\n审计基线 `{BASELINE}`；实际测量代码 `{MEASURED}`，服务器工作树干净。后续报告提交仅增加文档/证据，不改变已冻结source_hashes。通过Git bundle传输并核对SHA；不依赖服务器直连GitHub成功。PR #1保持draft、未合并。\n\n"
        "首次在9b6c02d的epoch中断验收失败：last已发布而首次best尚未写盘，随后分数不改善会在最终汇总时找不到best。bf4f123调整为先持久化best再发布last，增加两方法固定验证分数的崩溃回归测试。冻结数学文件未改动。旧失败产物留在独立旧目录；新提交使用全新工作目录、全新训练产物重新验收，未放宽来源或精确恢复校验。\n\n"
        f"系统：Ubuntu22.04.5，内核5.15.0-78；Xeon Gold6459C，主机可见128逻辑CPU、754GiB RAM（容器可用CPU配额另见原始证据）；新项目位于50GiB XFS数据盘。驱动595.71.05；单张NVIDIA GeForce RTX5090，32607MiB、CC12.0。CUDA逻辑cuda:0的UUID与nvidia-smi物理清单匹配；原始UUID仅本地保留，公开证据使用一致脱敏标识。CUDA_VISIBLE_DEVICES未设置，无CPU回退。\n\n"
        f"环境：`{json.dumps(env,ensure_ascii=False)}`。Python解释器与包安装路径均记录。torch/torchvision从服务器预装官方wheel库复制到隔离.venv，未修改系统Python；核心库RECORD逐文件校验通过，未复制也未使用外部torchrun/torchfrtrace入口。其他依赖保持requirements锁定版本，网络慢时使用明确记录的镜像来源。pip freeze和wheel源/哈希见公开证据。完整复制环境含非研究依赖，正式部署建议使用锁定wheel离线重建精简环境后重新验收。\n\n"
        "真实CUDA核验包括Tensor、矩阵乘法、Conv/BN前反向、SGD、FP64 log/exp/reduction，开启确定性，关闭AMP/TF32/compile。非单位椭球与独立一步参考、参数恢复、base恰好更新一次、同batch双前向及BN仅更新一次已通过。preflight最小修复使用指定cuda:N及其UUID，不再硬编码物理GPU0。\n\n"
        f"数据：`{json.dumps(data,ensure_ascii=False)}`。官方服务器单连接过慢，使用同一官方地址Range分段下载，整包MD5=c58f30108f718f92721af3b95e74349a，随后经torchvision逐文件完整性检查。训练45000/验证5000，无交集；完整Manifest在服务器新目录内，公开报告给出指纹。测试10000张仅做完整性/载入，不执行测试集指标或选择gamma。\n\n"
        "恢复验收覆盖两方法、epoch内与已完成scheduler的epoch边界、workers=2；比较参数/buffers、momentum、scheduler、几何、RNG、顺序hash、步数及去除时间字段的历史。真实45000张训练划分的连续/batch中断/epoch中断三条轨迹各限定353步，完整跨一次epoch；两方法全部逐位一致。固定mask恢复训练也通过GPU精确续训。完整清单见smoke.json、real_resume.json和recovery_resume.json。故意注入的BoundaryStop在各中断目录留下failure.json，属于验收夹具；旧9b6c02d的FileNotFoundError则是已整改的真实失败，两者不得混淆。\n\n"
        "真实数据还完成两方法3步稠密smoke的last.pt→1:3 magnitude硬剪枝→统一1 epoch SGD恢复（各352步）→验证集评估，并逐位复核恢复续训。每步验证mask合法与参数不回生，保留schema2父checkpoint哈希。该last不是正式final checkpoint；1:3、恢复预算和准确率均不作为正式协议或Gamma选择依据。详见real_pruning.json。\n\n"
        "本地84项测试通过；服务器84项通过，保留Python3.12多线程fork警告，未关闭确定性或忽略测试。短程workers测试不能证明所有系统负载下绝无fork死锁。容器CPU配额为16核、内存上限92GiB；主机可见资源不是配额。首次GPU利用率100%但无可见计算进程，之后空闲回落0%；不据此声称独占主机或排除宿主干扰。线程记录为intra-op4、inter-op64，正式分配需批准。\n\n"
        "作业使用nohup、标准输入/dev/null、独立日志和PID，SSH断开不依赖交互shell；不设置正式批准变量，不运行run_one.sh。原始产物位于独立项目artifacts/server_acceptance与runs的新命名空间，不读取历史checkpoint。\n\n"
        +"\n".join(f"- {k}: {v}" for k,v in gate.items())+"\n\nFP32精度结果和短期/长期边界见GPU_PERFORMANCE与GAMMA_CALIBRATION。服务器使用detached HEAD固定测量提交；其源码与PR工作分支一致，未修改分支或合并PR。\n\n"
        "安装方案参见[PyTorch官方历史版本](https://pytorch.org/get-started/previous-versions/)；驱动兼容性参见[NVIDIA CUDA兼容性](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)。本轮兼容性结论同时有真实kernel证据。整包MD5与划分来源参见[官方CIFAR页面](https://cave.cs.toronto.edu/kriz/cifar.html)。\n",encoding="utf-8")
    names = {"source":"源码版本与分支一致","core":"数学核心未改动","hardware":"5090硬件验收","cuda_smoke":"CUDA Smoke","real_data":"真实CIFAR-10","exact_resume":"精确断点恢复","gamma":"固定Gamma短程校准","protocol":"完整协议冻结","formal_training":"200 epoch正式训练"}
    (DOCS/"AUDIT_GATE.md").write_text("# 实验审批门禁\n\n|门禁|状态|证据|\n|---|---|---|\n"+"\n".join(f"|{names[k]}|{v}|validation/server_acceptance/audit_gate.json；相关专题报告|" for k,v in gate.items())+
        f"\n\n测量代码 `{MEASURED}`，父审计基线 `{BASELINE}`。\n\n"+"\n".join(f"- {name}: SHA256 `{digest}`" for name,digest in core.items())+
        f"\n\n短程固定gamma提名：{recommended}；仅FP64有限预算证据。FP32门禁失败不以放宽容差处理。\n\n"
        "可以提交GPT审查本轮证据与待批准协议；这不构成正式实验执行许可。未批准项：验证划分/seed列表、SGD/LR/增强、rho/gamma/warm-up/精度、N:M/scope/tail、恢复预算/优化器、checkpoint选择、probe、日志频率与GPU资源。正式JSON中的null保持阻断。\n",encoding="utf-8")
    write_json(PUBLIC/"evidence_manifest.json", {"measured_source_commit":MEASURED,"public_files_sha256":{
        str(p.relative_to(DOCS)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(PUBLIC.rglob("*")) if p.is_file() and p.name != "evidence_manifest.json" and "__pycache__" not in p.parts},
        "renderer_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reports_sha256":{name:hashlib.sha256((DOCS/name).read_bytes()).hexdigest() for name in
                         ("SERVER_ACCEPTANCE.md","GPU_PERFORMANCE.md","GPU_PERFORMANCE.json","GAMMA_CALIBRATION.md","FORMAL_PROTOCOL_DRAFT.md","AUDIT_GATE.md")}})


if __name__ == "__main__": main()
