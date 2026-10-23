# PR 1 最后一轮小范围整改

基线：84e2587。仅修复设备一致性和绘图checkpoint一致性，保留前轮实现与全部测试。
不修改 geometry.py、step.py、VPR-SAM公式、Gamma或正式训练方案。

## P1 正式设备与比较规则

- 正式配置必须显式指定用户批准的逻辑设备 `cuda:N`；`cpu` 或未解析的 `cuda` 不能启动正式训练。
  N仍由之后的资源批准决定，本轮没有选择任何实际服务器设备。
- 启动时核验目标设备为RTX 5090、compute capability 12.0，且具有可记录的GPU UUID。
  不允许CPU回退；设备门禁在数据IO之前执行。
- checkpoint/environment.json中的execution记录绑定实际模型参数设备，包含请求设备、解析后的类型/索引、
  目标GPU名称/UUID、CUDA_VISIBLE_DEVICES映射和实际intra-op/inter-op线程数。GPU可见不等于使用了GPU。
- comparison_protocol保留device和threads。比较要求相同逻辑设备、物理UUID/映射与环境，
  intra-op线程严格相等；inter-op线程也通过完整execution指纹要求一致。
  这是一条保守的一致性规则；若未来跨GPU分配实验，须另行批准比较策略，不能静默放宽。
- schema仍为2；执行记录参与现有provenance、恢复、后处理和比较校验。没有迁移或改写旧产物。
  源码变更仍会明确拒绝旧源码产物，新的smoke写入新目录。
- 正反例覆盖正式CPU/未指定索引拒绝、模拟合法CUDA目标、错误索引/型号/UUID/线程、
  CUDA不可用、GPU可见但实际CPU、混合设备比较与线程不同的比较。
  CUDA正反例只mock元数据，没有创建CUDA张量或执行kernel。

## P2 图表选定同一checkpoint

- 绘图统一读取汇总指定的final或best_validation；几何直方图、训练/验证历史、扰动轨迹和层预算迁移
  全部来自该快照内的历史与几何，历史截断到选定快照，不再读取final的外置历史文件。
- 硬剪枝与恢复图核对其选定父checkpoint哈希；恢复继续使用汇总指定的recovery_checkpoint。
  checkpoint相关标题/图例标注来源，输出figure_sources.json保存文件路径、SHA256、epoch和step。
- 回归测试使用实际不同的best/final快照，核对直方图数据、绘图读取路径、dense/hard/recovery来源，
  并验证哈希不匹配时拒绝绘图。

## 验证与边界

本轮完整测试与合成SAM/VPR核心对照的机器可读记录见validation/final_audit_results.json和
validation/final_audit_pytest.xml。保留前轮63项测试；仅将原正式配置门禁测试的合法设备输入从CPU
调整为cuda:0，原有逐字段协议检查不变。

未连接服务器、未执行CUDA正式验收、Gamma校准或200 epoch训练。收到的服务器凭据没有写入任何文件或提交。
PR 1保持待审、未合并，等待再次复核。
