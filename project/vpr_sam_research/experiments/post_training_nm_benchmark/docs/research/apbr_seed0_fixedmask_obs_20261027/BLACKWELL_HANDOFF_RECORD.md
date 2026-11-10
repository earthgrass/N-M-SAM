# Seed 1/2 Blackwell Handoff Record

记录时间：2026-10-27 13:18 CST。

用户明确批准舍弃旧 RTX 4080 SUPER 上正在运行的 Seed 1 APBR partial，并在新 Blackwell GPU 从头训练 Seed 1 APBR。旧 partial 在 `completed_epochs=12`、`steps=4692` 时停止；`last.pt` SHA256 为 `f7c6cf3b104f7447322de27c725be7ccfefdc1b4c63589e12edaf39f5f1fb882d`。文件保留，未删除、未覆盖，不进入正式结果。

Seed 1 Dense-SAM Final 从旧机迁移到新机，迁移前后固定 SHA256：

`e1909b70c0574a17c0ba1e5558a738848e943a67eed2c2bc2bb150bd3c96da7b`

新队列：

- GPU：NVIDIA RTX PRO 6000 Blackwell Server Edition。
- UUID：`GPU-65be6845-be69-8ba1-72f8-e34688ce53c5`。
- 输出根：`artifacts/cifar100_50k10k_final_v1_blackwell_handoff/`。
- 配置：`configs/cifar100_50k10k_final_v1_blackwell_handoff.yaml`。
- 控制器：`scripts/run_cifar100_blackwell_handoff_queue.py`。
- 控制器 PID：2535；首个 Seed 1 APBR 子进程 PID：2562。
- 启动前 Seed 1 Dense checkpoint SHA、status、protocol、epoch、steps、seed、method 和 Test 隔离均验证通过。
- 冻结的数据、模型、优化器、scheduler、APBR 几何、N:M、校准、BN-only、Recovery-20 和 Test 协议与原配置逐项一致。
- 硬件 UUID、配置哈希和新训练 checkpoint 身份会如实反映 Blackwell handoff；不伪装为单机全程运行。

新队列顺序：Seed 1 APBR 200 epochs → Seed 1 五方法原生剪枝/BN-only/Recovery-20 → 12 个固定 Final Test；随后 Seed 2 Dense 200 → APBR 200 → 五方法完整后处理 → 12 个固定 Final Test。无自动重试。
