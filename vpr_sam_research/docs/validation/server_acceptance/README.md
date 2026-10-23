# 本轮公开验收证据

测量源码：`bf4f123e01a9b0819376a4def7d2e82e5c56975d`；审计父基线：`ee57bc68b9a2f5f71dffb067f67416a9461f8db1`。
文档提交可以晚于测量提交；renderer核对全部已声明源码/配置/脚本哈希后才能生成报告。

- `local_pytest.xml` / `server_pytest.xml`：本地、服务器完整84项测试，服务器保留fork警告。
- `kernels.json` / `math.json`：真实GPU内核与非单位椭球独立参考。
- `data.json`：官方完整性、样本数、划分与Manifest指纹；测试集未用于指标选择。
- `smoke.json`：合成ResNet18完整路径及短程真实数据。
- `real_resume.json`：真实45000张训练划分，SAM/VPR，workers=2，batch/epoch中断各到353步，逐位核对。
- `recovery_resume.json`：固定mask恢复的GPU精确续训。
- `real_pruning.json`：真实CIFAR两方法，3步稠密last→1:3硬剪枝→统一352步SGD恢复→验证，以及逐位恢复续训。
- `workflow_evidence/`：完整合成路径的配置、环境、指标、剪枝和恢复来源链；合成准确率没有科研意义。
- `g1_energy.csv` / `g1_layer_energy.json`：300步真实SAM g1全局统计与参数张量能量贡献。
- `candidate_trajectories.csv` / `vpr_candidate_*_layers.json`：三个分别从同seed重新随机初始化、初始权重相同的候选，记录100步轨迹及预算贡献；仅一个seed。
- `underflow_audit.json`：9个张量/步事件均低于FP32舍入为零阈值；完整重复三条轨迹，模型、优化器和几何状态逐位一致。重放不是新的独立seed。
- `real_resume_geometry.json`：保守候选353步的额外数值记录；cosine horizon=2是smoke夹具。
- `performance_vpr_float32_failure.json`：FP32 log-radius首步体积门禁失败；停止该配置，不放宽容差。
- `epoch_boundary_failure_9b6c02d.json`：首次GPU验收发现的真实checkpoint发布顺序缺陷。整改后在新目录重跑，不删除原失败。
- `package_sources.json`：实际wheel源、SHA256与torch/torchvision库11501/203个文件的RECORD复核。
- `final_state.json`：测量提交、干净工作树、独立import位置、正式批准变量未设置及GPU进程快照。

公开JSON是脱敏证据，不是可加载checkpoint。原始UUID及所有模型/数据保留在独立项目本地，公开文件不替换它们。
`evidence_manifest.json`绑定公开证据、补充启动器、报告和renderer的SHA256。
checkpoint中的`source_hashes()`覆盖src/configs/scripts等声明集合；`reproduction/`是另行绑定的操作启动器，不冒充已纳入该集合。

## 重建短程验收

这些命令只用于明确授权后的短程复验，必须使用全新目录/输出命名空间；现有证据不覆盖。
先按部署文档安装锁定依赖，核对分配GPU的实际UUID与逻辑索引。本轮验证的是cuda:0；不能在另一台机器盲目复用该索引。
所有命令从独立工程根目录执行；下面没有正式训练入口。

```bash
python scripts/preflight.py --require-5090 --device cuda:0
python -m pytest -q --basetemp=artifacts/acceptance-tests
python scripts/server_validation.py kernels --device cuda:0
python scripts/server_validation.py math --device cuda:0
python scripts/server_validation.py data --device cuda:0
python scripts/server_validation.py smoke --device cuda:0
```

`data`显式准备官方CIFAR，失败报错，不使用FakeData。网络慢时可手动准备同一官方整包，但必须核对MD5并由torchvision重新验证，不能取消完整性检查。

补充启动器的原始执行副本见`reproduction/`。在新项目内将它们复制到`artifacts/server_acceptance/`，核对哈希再运行：

```bash
# 测量三种精度/方法，各独立进程；随后原生engine步时与300步g1采样。
# 若FP32因体积容差失败，脚本只允许独立的已通过FP64路径继续。
bash artifacts/server_acceptance/performance_and_g1.sh
# 先阅读gamma_estimate.json；本轮只批准0.5/1/2倍gamma0的固定候选。
python scripts/server_validation.py candidates --device cuda:0
python artifacts/server_acceptance/underflow_audit.py
python artifacts/server_acceptance/recovery_resume.py
python artifacts/server_acceptance/real_resume.py
python artifacts/server_acceptance/real_pruning.py
```

`real_resume.py`每条轨迹止于353步；`real_pruning.py`使用3步稠密smoke的`last.pt`和统一1 epoch SGD恢复，不把它伪装成正式final checkpoint。
各方法及恢复从本项目自己的新产物开始；所有真实测试集仅做完整性读取，未运行测试指标。
SSH任务使用`nohup ... > 独立日志 2>&1 < /dev/null &`；本轮结束后所有计算任务均已结束。

## 报告再生成

原始证据包只含JSON/XML/日志/启动器，不含数据和模型；公开文件只保留脱敏后的选择集。
完整本地原始包SHA256：`c7554d8165673acab9806f66f41c6b1650faa3bb3eea24b35cb235122a0dec70`。
在本地从可信证据包解压后：

```bash
python docs/validation/render_server_reports.py --raw artifacts/server_session/raw/server_acceptance
```

renderer会核对源码集合、核心文件哈希、配对初始化和样本顺序，重新生成五份报告及图表。
原始包保留在本地`artifacts/server_session/`，不上传Git；原始GPU UUID不进入公开报告。
