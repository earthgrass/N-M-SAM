# RTX5090 部署与实验执行计划

## 环境设计

推荐 Linux x86_64、Python3.12 的独立虚拟环境。安装脚本固定 torch2.8.0、
torchvision0.23.0、CUDA12.8 wheels；其余直接依赖见 requirements-research.txt。
这是一套待服务器验证的目标环境，不等于已经在服务器通过测试。
执行安装后保存完整 pip freeze，作为实际依赖锁；不宣称仅固定顶层包就锁定所有传递依赖。

依据：

- PyTorch 官方历史版本安装命令：https://pytorch.org/get-started/previous-versions/
- NVIDIA 官方 compute capability 列表（RTX5090=12.0）：https://developer.nvidia.com/cuda/gpus

驱动必须支持所用 CUDA runtime。preflight 检查 GPU 名称、compute capability、
实际 CUDA 矩阵乘 kernel 和 PyTorch build 信息；安装与版本号本身不证明训练兼容。

## 迁移

只需复制 vpr_sam_research 目录的源码、配置、脚本与文档，不复制父工程。
不要复制本地 .venv、artifacts、runs 或历史数据/checkpoint；数据由本项目单独显式准备。
从 GitHub VPR-SAM 分支检出后进入本目录也可。

```bash
bash scripts/setup_5090.sh
source .venv/bin/activate
python -m pytest -q --basetemp=artifacts/pytest-temp --junitxml=artifacts/pytest.xml
python scripts/smoke_suite.py --run-id server_cpu_acceptance --include-resnet
# 只做几何计算的 CUDA 性能与精度评估，仍不是正式训练
python scripts/benchmark_geometry.py --device cuda --dimensions 100000 1000000 --steps 20 --output artifacts/geometry_cuda.json
# 必须另行获批并在目标机器执行；真实 ResNet18 step time/显存及训练、恢复的严格断点对照
python scripts/device_acceptance.py --device cuda --model resnet18 --run-id approved_cuda_acceptance
```

CPU smoke 不能代替 GPU 训练路径。经批准再复制 smoke 配置到新文件，将 device 改为 cuda、
保持 purpose=smoke 和极小预算，验证两次反向、显存、确定性及 GPU 断点恢复。
GPU smoke 配置和结果须保存在本项目内；不要拿 CPU 配置加载 GPU checkpoint。
device_acceptance.py 使用显式合成数据、两个极短的SAM/VPR流程，分别比较连续和断点恢复状态，
并比较固定Mask恢复训练的连续/断点状态。输出每步时间、峰值CUDA显存、d_p_max与最大几何更新。
CPU脚本路径验收使用 --device cpu --model tiny，不代表 CUDA 或真实 CIFAR-10 通过。

## 正式实验门槛

1. GPT 审计源代码和数学定义，解决发现的问题。
2. 共同冻结正式配置、seed、数据划分、调参预算、N:M 与剪枝层范围、恢复和模型选择规则。
3. 本地测试通过；服务器依赖、CUDA kernel、真实数据、GPU smoke 和断点恢复通过。
4. 使用微基准及 ResNet18 GPU smoke 决定 log-radius 精度及资源需求。
5. 明确 GPU 分配、单任务预算、磁盘空间和停止条件，再授权单任务正式训练。
6. 同 seed 的所有方法使用同一共同协议，分别生成独立随机初始化训练产物。
7. 完整训练后先核验初始化 hash、数据顺序、实际预算、几何数值和 checkpoint 来源。
8. 按同一 checkpoint 选择规则执行硬剪枝和统一恢复；最终一次冻结 test 评估。
9. 汇总独立 seed、配对差值与不确定性；必要消融经讨论后安排。

没有在本地交付阶段执行上述服务器命令；不存在后台任务或自动提交。
FP64 几何使用较多全局归约与 log/exp，消费 GPU 的性能和临时显存必须实测，不能根据两次反向推断开销很小。
