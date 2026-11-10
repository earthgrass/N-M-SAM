# APBR 与原生 GPT Mask 本地审计

两组对照各自使用完全相同的 Dense 父模型；本审计只读取历史文件，不访问数据集或 Test。

| 父模型 | 相同位置 | 相同 2:4 组 |
|---|---:|---:|
| apbr_vpr_seed0 | 8,773,346/11,157,504 (78.63%) | 1,658,684/2,789,376 (59.46%) |
| apbr_l2_seed0 | 8,761,406/11,157,504 (78.52%) | 1,652,593/2,789,376 (59.25%) |

所有 19 层均通过严格 2:4、Dense 父模型 SHA256、Mask 张量 SHA256、GPT Mask 文件 SHA256 和 APBR Raw Hard 精确投影检查。
每套 Mask 都删除 5,578,752 个位置。

本地 CPU 数值门禁：`python -m pytest -q tests/test_fixedmask_obs.py --basetemp=artifacts/pytest-temp-apbr-gpt-mask-local`，6 passed。
覆盖外部固定 Mask 不变、删除位置保持零、幸存权重得到补偿、解析二次解、局部重建误差和 Conv2d unfold Hessian。

已有历史 Test 结果仅作诊断：原版 APBR 同一 Dense 父模型上的 APBR Mask-only 为 59.22%，GPT Mask-only 为 48.66%；APBR Mask + Fixed-Mask OBS 为 76.37%。
GPT Native 采用动态选 Mask 与原生顺序补偿，不能直接与 APBR Fixed-Mask OBS 当作同求解器的 Mask 质量对照。

| 同一 Dense 父模型 | APBR Mask-only | GPT Mask-only | APBR Mask + Fixed-Mask OBS | GPT Mask + Fixed-Mask OBS | GPT Native（参考） |
|---|---:|---:|---:|---:|---:|
| 原版 APBR Seed 0 | 59.22% | 48.66% | 76.37% | **未运行** | 76.99% |
| APBR-L2 Seed 0 | 56.32% | 未核验 | 未运行 | **未运行** | 77.12% |

`GPT Native` 是原生动态选 Mask 与补偿的历史结果，不能填充 `GPT Mask + Fixed-Mask OBS` 列。
本机 PyTorch 为 `2.13.0+cpu`，`torch.cuda.is_available() == False`；尚无可用 GPU 服务器，因此不能对未运行列报告准确率。

这些重合率不能单独证明 Mask 与 OBS 的匹配程度。完整同父模型、同固定 Mask OBS 求解器对照尚未运行，需 GPU。

机器可读逐层结果：`mask_identity_overlap.json`。
