# Block-wise 工程计划与已实现范围

独立 `research/blockwise-prototype-20261024` 分支的 prototypes/blockwise.py
继承冻结 Geometry 的全局 fixed_l2 扰动路径；只替换 prepare 的局部反馈。
CandidateStepper 复用冻结 Stepper 的两次 forward、BN、参数恢复和外层 SGD。
它硬性拒绝 formal purpose 与非 CPU 设备，没有正式训练入口。
历史 geometry.py、step.py 和 training/engine.py 不修改。

参数映射、n/m、gamma_B、eligible 名称、原 Global 状态及计数写入独立版本 checkpoint；
恢复时拒绝协议不一致，校验完整块守恒、dense tail 与未剪枝坐标零几何。
原型不改变历史 checkpoint schema，也不加载 Global 训练状态伪装为 Block-wise。

单测覆盖：局部 log-volume、全局 L2、1:3/2:4/3:5 分块顺序、zero-energy、
极小 FP32 扰动、未剪枝参数/tail、FP64 状态、checkpoint 往返与拒绝错配、
BN 每 step 仅一次 running-stat 更新、固定随机种子精确 resume、NaN/Inf 拒绝。
TinyNet 合成数据的真实两步 SGD/SAM smoke 包含在 CPU 单测中；没有 CIFAR 正式训练。

执行：在候选项目根设置 PYTHONPATH=src 与当前目录，运行
`python -m pytest -q tests prototypes/test_blockwise.py`。
完整现有测试加原型共 116 项通过；其中原型 9 项。CPU PyTorch 2.13.0，
历史训练为 PyTorch 2.8.0+cu128，不能把本地 CPU PASS 当作 CUDA 正式验收。

后续提案：先冻结参数集合与 gamma_B 候选的理论尺度、低能量块政策与评价终点；
预登记梯度固定和带噪声轨迹；再申请 CUDA 短程验收与主实验预算。
没有授权时不启动 200 epoch、不增加 seed、不搜索 Gamma_B。
