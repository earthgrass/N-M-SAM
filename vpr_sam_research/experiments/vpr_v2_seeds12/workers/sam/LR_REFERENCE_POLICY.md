# LR参考值管理修正，2026-10-24

用户授权将跨平台末位差异与算法一致性检查分开。训练配置、scheduler.py、core.py、data.py、
model.py及官方调度调用顺序均不改变。Windows reference/LR_EXPECTED.json保持原字节。
原服务器LR_GATE_FAILURE_DIAGNOSIS.json、失败日志和45k/5k产物保持原路径与原SHA。

在新的独立服务器工作目录中，lr_reference.py从锁定SSAM源码校验SHA后AST加载官方类，
仅去掉注册/配置装饰器；使用冻结配置生成200轮官方序列。项目调度器另建SGD和scheduler，
在同一环境独立执行。epoch-end step(zero-based epoch)不变，400个字段全部严格相等才放行。
没有容差、特殊数值替换或跳过测试；回归测试证明一个ULP的变化也会拒绝。

artifacts/lr_reference/保存官方序列、项目序列、逐字段报告、Windows与服务器差异、
配置及其SHA、Git commit、Python/PyTorch/CUDA/系统环境、两个调度器源码SHA与所有序列SHA。
训练入口启动时再次核对环境、配置、参考文件SHA和项目严格执行结果；不匹配即停止。
正式日志learning_rates_expected.json采用服务器官方参考；learning_rates_windows.json保留
原Windows参考；learning_rates_actual.json保存训练实际使用和epoch后LR。

原跨平台比较失败仍明确记录为差异，不改写为完全一致；本次门禁是同一服务器环境内
官方与项目的严格数值一致性。原始失败工作区不重新执行，不覆盖日志。新版本完整重跑
CPU/CUDA、官方对照、数据、BN、RNG、断点及真实数据smoke，全部通过才运行唯一seed0轨迹。
