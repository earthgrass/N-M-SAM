# RTX 5090 服务器验收

审计基线 `ee57bc68b9a2f5f71dffb067f67416a9461f8db1`；实际测量代码 `bf4f123e01a9b0819376a4def7d2e82e5c56975d`，服务器工作树干净。后续报告提交仅增加文档/证据，不改变已冻结source_hashes。通过Git bundle传输并核对SHA；不依赖服务器直连GitHub成功。PR #1保持draft、未合并。

首次在9b6c02d的epoch中断验收失败：last已发布而首次best尚未写盘，随后分数不改善会在最终汇总时找不到best。bf4f123调整为先持久化best再发布last，增加两方法固定验证分数的崩溃回归测试。冻结数学文件未改动。旧失败产物留在独立旧目录；新提交使用全新工作目录、全新训练产物重新验收，未放宽来源或精确恢复校验。

系统：Ubuntu22.04.5，内核5.15.0-78；Xeon Gold6459C，主机可见128逻辑CPU、754GiB RAM（容器可用CPU配额另见原始证据）；新项目位于50GiB XFS数据盘。驱动595.71.05；单张NVIDIA GeForce RTX5090，32607MiB、CC12.0。CUDA逻辑cuda:0的UUID与nvidia-smi物理清单匹配；原始UUID仅本地保留，公开证据使用一致脱敏标识。CUDA_VISIBLE_DEVICES未设置，无CPU回退。

环境：`{"python": "3.12.3", "platform": "Linux-5.15.0-78-generic-x86_64-with-glibc2.35", "packages": {"torch": "2.8.0+cu128", "torchvision": "0.23.0+cu128", "numpy": "2.2.6", "matplotlib": "3.10.3", "Pillow": "11.2.1"}, "cuda_runtime": "12.8", "gpu": "NVIDIA GeForce RTX 5090", "cuda_capabilities": ["sm_70", "sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"], "deterministic": true}`。Python解释器与包安装路径均记录。torch/torchvision从服务器预装官方wheel库复制到隔离.venv，未修改系统Python；核心库RECORD逐文件校验通过，未复制也未使用外部torchrun/torchfrtrace入口。其他依赖保持requirements锁定版本，网络慢时使用明确记录的镜像来源。pip freeze和wheel源/哈希见公开证据。完整复制环境含非研究依赖，正式部署建议使用锁定wheel离线重建精简环境后重新验收。

真实CUDA核验包括Tensor、矩阵乘法、Conv/BN前反向、SGD、FP64 log/exp/reduction，开启确定性，关闭AMP/TF32/compile。非单位椭球与独立一步参考、参数恢复、base恰好更新一次、同batch双前向及BN仅更新一次已通过。preflight最小修复使用指定cuda:N及其UUID，不再硬编码物理GPU0。

数据：`{"identity": "official-cifar10-md5-verified", "fingerprint": "ebf1209d6a63601ff62571e67b38127a245e579006a11cdcecc6f73309e652ef", "train": 45000, "validation": 5000, "test": 10000, "root": "/root/autodl-tmp/vpr_acceptance_20261023/acceptance_fixed/vpr_sam_research/data/cifar10", "augmentation": {"enabled": true, "cutout": 0}, "test_used_for_selection": false}`。官方服务器单连接过慢，使用同一官方地址Range分段下载，整包MD5=c58f30108f718f92721af3b95e74349a，随后经torchvision逐文件完整性检查。训练45000/验证5000，无交集；完整Manifest在服务器新目录内，公开报告给出指纹。测试10000张仅做完整性/载入，不执行测试集指标或选择gamma。

恢复验收覆盖两方法、epoch内与已完成scheduler的epoch边界、workers=2；比较参数/buffers、momentum、scheduler、几何、RNG、顺序hash、步数及去除时间字段的历史。真实45000张训练划分的连续/batch中断/epoch中断三条轨迹各限定353步，完整跨一次epoch；两方法全部逐位一致。固定mask恢复训练也通过GPU精确续训。完整清单见smoke.json、real_resume.json和recovery_resume.json。故意注入的BoundaryStop在各中断目录留下failure.json，属于验收夹具；旧9b6c02d的FileNotFoundError则是已整改的真实失败，两者不得混淆。

真实数据还完成两方法3步稠密smoke的last.pt→1:3 magnitude硬剪枝→统一1 epoch SGD恢复（各352步）→验证集评估，并逐位复核恢复续训。每步验证mask合法与参数不回生，保留schema2父checkpoint哈希。该last不是正式final checkpoint；1:3、恢复预算和准确率均不作为正式协议或Gamma选择依据。详见real_pruning.json。

本地84项测试通过；服务器84项通过，保留Python3.12多线程fork警告，未关闭确定性或忽略测试。短程workers测试不能证明所有系统负载下绝无fork死锁。容器CPU配额为16核、内存上限92GiB；主机可见资源不是配额。首次GPU利用率100%但无可见计算进程，之后空闲回落0%；不据此声称独占主机或排除宿主干扰。线程记录为intra-op4、inter-op64，正式分配需批准。

作业使用nohup、标准输入/dev/null、独立日志和PID，SSH断开不依赖交互shell；不设置正式批准变量，不运行run_one.sh。原始产物位于独立项目artifacts/server_acceptance与runs的新命名空间，不读取历史checkpoint。

- source: PASS
- core: PASS
- hardware: PASS
- cuda_smoke: PASS
- real_data: PASS
- exact_resume: PASS
- gamma: PASS
- protocol: PENDING
- formal_training: NOT STARTED

FP32精度结果和短期/长期边界见GPU_PERFORMANCE与GAMMA_CALIBRATION。服务器使用detached HEAD固定测量提交；其源码与PR工作分支一致，未修改分支或合并PR。

安装方案参见[PyTorch官方历史版本](https://pytorch.org/get-started/previous-versions/)；驱动兼容性参见[NVIDIA CUDA兼容性](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)。本轮兼容性结论同时有真实kernel证据。整包MD5与划分来源参见[官方CIFAR页面](https://cave.cs.toronto.edu/kriz/cifar.html)。
