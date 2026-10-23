# V1 审计整改报告

整改基线：`758d628`，保留原提交，不重写或合并覆盖 V1。
本轮仅修改独立工程；不修改冻结数学公式，不改历史代码、数据和checkpoint，不连接服务器。

## 问题与修复对应

| 审计项 | 修复 | 验证重点 |
| --- | --- | --- |
| P1 剪枝/恢复配对统计缺失 | 新增seed级统计与完整parent→hard→recovery证据链；输出JSON/CSV及配对图 | 双seed SAM/VPR端到端、已知差值参考、拒绝混合N:M和不完整结果 |
| P1 几何配置混入seed统计 | 同方法跨seed严格校验geometry，跨方法允许算法专有几何变化 | gamma、warm-up、dtype分别有拒绝测试；合法跨方法对照通过 |
| P1 后处理来源校验不足 | artifact schema2，模型/配置/Mask身份与源文件校验；评估JSON绑定父checkpoint | 修改源码/模型/配置/Mask/评估来源时拒绝；V1不静默迁移 |
| P2 非单位A完整一步缺失 | 人工设置全局零和非单位log-radius，独立计算g1/epsilon/g2/SGD/反馈 | 包含非零momentum与weight decay的参数、momentum和几何参考对照 |
| P2 scope与稀疏率歧义 | canonical名exclude_first_and_pointwise；保留旧别名与原选择；同时报告三种稀疏率和分母 | 包含Linear头、排除首层/1×1，尾块及全模型分母测试 |
| P2 正式协议门禁 | 按用户既定要求采用全流程冻结；正式train同时验证N:M和恢复预算 | 逐字段未定/非法值拒绝；没有选择任何具体正式参数 |

## 统计合同

研究主指标不再只存于单run列表：

- `postprocessing.hard.accuracy/loss/loss_increase/accuracy_drop`；
- `postprocessing.recovery.accuracy/loss`；
- 每项含各方法n/mean/sample_std与相同seed的差值、均值差、bootstrap区间；
- `postprocessing.csv` 保存seed级观测与hard/recovery checkpoint SHA256；
- 新图展示hard/recovery均值与SD，以及配对seed差值和bootstrap区间。

缺失N:M结果、不同scope/tail/mask/BN规则、不同恢复预算/顺序或不完整seed集合不能被默默汇总。
多个N:M协议在同一run中存在时，须用 --pruning-protocol 显式选一个。
统一checkpoint选择分别为dense `--checkpoint` 和恢复 `--recovery-checkpoint`。
test统计必须具有dense/hard/recovery各自的显式test评估证据；validation与test不混用。
小样本bootstrap仅为探索性统计，不代表正式论文结论。

## 来源合同与兼容性

artifact schema从1升为2，config schema保持1。新模型指纹包含shape/dtype，故与V1指纹不直接比较。
源文件不一致时明确拒绝剪枝、恢复、评估和汇总；不提供自动迁移或跳过检查的路径。
旧V1运行产物没有被改写。需要用冻结的新源码生成新的短程验证产物。
这是一项科研可比性检查，不是针对恶意伪造文件的数字签名；pickle仍仅用于可信产物。
模型hash会在checkpoint写入时读取模型状态，可能增加GPU同步/传输成本，服务器需实测。

## 本轮验证

本轮63项测试全部通过；独立wheel完成六方法×双seed共12条完整流程和CPU/tiny断点验收。
最终测试项、数量、结果、环境、源码指纹与smoke证据见 `docs/validation/audit_fix_results.json` 和
`docs/validation/audit_fix_pytest.xml`。V1的40项记录继续保留在原文件名中，不当作本轮结果。

本轮动态验证限于本地CPU与显式合成数据，包括双seed SAM/VPR的完整剪枝恢复统计、
ResNet18两步smoke，以及device验收脚本在CPU/tiny上的训练和恢复断点对照。
配置中的1:3及seed17/18仅是工程测试夹具，不是已批准的正式实验设置。

## 仍未验收

- RTX5090/CUDA运行、真实CIFAR-10加载与完整训练、多worker GPU逐位恢复；
- RTX5090几何微基准、ResNet18实际step time与峰值显存，以及FP32/FP64取舍；
- 正式N:M、数据split、gamma/精度/warm-up、seed数量、恢复预算、checkpoint选择和GPU分配。

已提供 `scripts/device_acceptance.py` 供目标设备获批后运行；本轮没有执行CUDA或服务器任务。
数学核心 `optim/geometry.py`、两步更新 `optim/step.py` 相对V1保持不变。
建议先审查本PR的统计与来源合同，再安排服务器验收，之后才批准正式随机初始化训练。
