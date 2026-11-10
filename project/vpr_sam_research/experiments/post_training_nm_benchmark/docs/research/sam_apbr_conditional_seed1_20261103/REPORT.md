# Seed 1: SAM 条件贪心 OBS 与 APBR 几何近优 OBS

## 冻结设置与执行

复用已经完成的 Seed 1 SAM 和原版 APBR-VPR Epoch-200 Dense Final；两者均为官方 CIFAR-100 50k Train / 10k Test、同一 100 类 ResNet18、同一正式配置、同一固定 1,024 张 Train 校准 ID。对 19 个内部 Conv2d 执行严格 2:4，无 Dense 重训、BN-only 或 Recovery。SAM 使用纯物理条件贪心选择和物理联合 OBS 补偿；APBR 使用此前冻结的 τ=1% 近优窗口，在窗口内以 APBR 几何分数裁决，仍做同一物理联合 OBS 补偿。两组 Mask 与 checkpoint 均在打开 Test 前冻结并计算 SHA256，分别进行一次固定 Test。

本机仅有 PyTorch `2.13.0+cpu`，CUDA 不可用。以下为 CPU 探索性结果，不是 GPU 成本测量。

| Seed 1 完整路径 | Dense Final | 剪枝及补偿后 Test | 相对本身 Dense | Test CE |
|---|---:|---:|---:|---:|
| SAM + 条件贪心 OBS | 78.97% | **77.77%** | −1.20 pp | 0.829210 |
| APBR-VPR + 几何近优条件贪心 OBS | 79.08% | 77.43% | −1.65 pp | 0.843766 |

APBR 的 Seed 1 Dense Final 高 0.11 pp，但剪枝补偿后比 SAM 低 **0.34 pp**。这一结果与 Seed 0 的几乎并列不同，未支持 APBR 几何路径优于 SAM 条件贪心的假设。

| 探索性 Seed | SAM + 条件贪心 | APBR + 几何近优条件贪心 | APBR−SAM |
|---|---:|---:|---:|
| 0 | 77.84% | 77.83% | −0.01 pp |
| 1 | 77.77% | 77.43% | −0.34 pp |

两 Seed 的 APBR−SAM 平均差为 −0.175 pp。样本数只有两个，且跨父比较同时改变 Dense 训练和 Mask 选择，不能把差异单独归因于 APBR 几何。Seed 0 同父消融中真实几何 77.83%、打乱几何 77.82%，两者也几乎相同。不要依据这些 Test 结果调整 τ 后把新结果当作预注册验证。

## 证据门禁

- Seed 1 两个 Dense 父 checkpoint SHA256 分别为 `e1909b70c0574a17c0ba1e5558a738848e943a67eed2c2bc2bb150bd3c96da7b`、`6571e30ed783508f28bc378aa3b22a739bf1ddbbb790db3d98c08858e126e10b`；均通过 Epoch 200、78,200 steps、`test_accessed=false` 与正式评估记录核验。
- 独立产物审计通过：每组 19 层 bool keep-mask、11,157,504 eligible 坐标、5,578,752 删除位、严格 2:4、删除权重精确为零、全部有限；BN 和排除层与各自父模型 bitwise 相同。
- 两个 checkpoint 的 SHA256 分别为 `c9a657636f42d11aafd5654b1c01028841482a69036e1951f240c9334b464c1d`、`de237a1d579e628d1d6998156f0c616f71c7ff2bae6b9da670661694c3f8c28d`。两份 Mask、源码、配置、数据及校准 ID 哈希见 `preflight.json`、`pre_test_freeze.json`、`results.json` 和 `audit.json`。
- 算法相关 CPU 数值测试此前共 **35 passed**；本轮没有修改适配器或历史正式文件，没有自动重试失败任务。

机器可读产物位于 `artifacts/sam_apbr_conditional_seed1_cpu_20261103_v1/`。本报告和产物清单见 `SHA256SUMS.txt`。
