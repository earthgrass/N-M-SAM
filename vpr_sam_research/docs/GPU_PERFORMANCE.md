# RTX 5090 性能与精度验收

|方法/状态|step ms（含观测）|前向 ms|反向+loss ms|扰动 ms|反馈+commit ms|峰值 GiB|保存 s / MiB|
|---|---:|---:|---:|---:|---:|---:|---:|
|sam float64 PASS|63.33|6.36|23.97|7.74|0.02|0.860|0.092 / 85.4|
|vpr float32 FAIL|—|—|—|—|—|—|—|
|vpr float64 PASS|105.15|6.38|23.05|21.05|15.60|1.032|0.164 / 170.6|

|无逐步观测器的原生engine|step ms|峰值 GiB|schema2 checkpoint MiB|状态捕获+seal+写盘 s|
|---|---:|---:|---:|---:|
|sam|60.72|0.860|85.5|0.111|
|vpr|97.08|1.032|170.8|0.202|

真实 CIFAR-10 / ResNet18，batch128，FP32模型，固定gamma=1e-7（仅性能夹具），4个intra-op线程、2个workers；前2步预热不计入均值。每个精度/方法在独立进程中测量，避免前一个模型分配污染峰值。

SAM不维护log-radius状态，表中float64只是共同配置的占位标签；两方法模型均为FP32，SAM扰动仍使用原有FP64归约。

观测总时间包含同步、hooks及诊断归约；反向时间包含loss计算，不能作为纯CUDA kernel时间。DataLoader数字是主进程等待时间，不能解释为全部CPU增强成本。原生engine对照同初始化/顺序，VPR步时开销约60%；checkpoint组件计时包含state捕获、seal及序列化，不覆盖save闭包的全部JSON与字典构造时间。诊断checkpoint不是schema2训练checkpoint，保存时间/大小与原生产物分列。长程step_rows历史增长会增加checkpoint成本，12步结果不能代表末期文件大小。

精度优先推荐FP64 log-radius；任何FP32体积门禁失败均停止该配置，不放宽容差。FP64、FP32状态本体分别为d×8、d×4字节；峰值使用torch.cuda.max_memory_allocated，包含模型、梯度、SGD momentum、激活及临时FP64张量，但不包含全部驱动/context/NVML显存。辅助单向量microbenchmark采用原有独立夹具容差，不能替代真实训练门禁。

- sam/float64: volume max=0.0, ellipse max error=6.878413116861992e-11
- vpr/float32: log-volume drift 2.849877001054324e-10 exceeds tolerance 1.0330038299573153e-10; no recentering applied
- vpr/float64: volume max=1.2108369862318114e-15, ellipse max error=3.521911928761057e-11

未验证独占调度或长时间热稳定性，短测不外推为200 epoch资源保证。机器可读数据见 GPU_PERFORMANCE.json。
