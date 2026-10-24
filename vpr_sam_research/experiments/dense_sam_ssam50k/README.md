# Dense-SAM SSAM-aligned full50k seed0

唯一轨迹：CIFAR10官方50k/10k，CIFAR-ResNet18，seed0，200epoch/78000steps。
每epoch随机采样器打乱，390×128=49920训练样本，丢弃80个尾样本；无内部Validation。
标准SAM/SGD，LR.05、WD.001、rho.1；双次BN更新；SSAM原始CosineLRscheduler调用顺序。
完整授权及最新seed覆盖见reference/。配置差异见CONFIG_COMPARISON.md。

## 运行

独立目录、无历史Python模块依赖。只读使用已存在官方数据，不下载或上传数据。
优先使用已验收torch2.8/CUDA12.8运行时；无需安装旧CUDA。

```bash
mkdir -p artifacts
python -m pytest -q tests --basetemp=artifacts/local-tests
# 部署者先生成绑定Git commit及逐源码SHA的SOURCE_FREEZE.json：
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8 python -u launch.py
```

launch.py先执行CPU/CUDA单测和2批真实训练数据smoke，捕获历史文件基线SHA。
所有门禁通过，才启动train.py；从随机初始化重新开始，smoke权重丢弃。
外层进程应以start_new_session=True及日志重定向脱离SSH。失败即停止，无自动重启或后续队列。
原暂停45k任务与本轮不同，不得恢复其checkpoint或继续其step计数。

本轮每轮在完整10k Test上评价并选择Best Test（严格升高、最早并列）。
best_test.pt和last.pt明确区分；Test参与选择，因此Best不声称独立无偏泛化。
不另行运行VPR、Guide、剪枝、BN校准、恢复、其他seed或超参数搜索。

## 产物

artifacts/dense_sam_ssam50k_20261024/seed0/：best_test.pt、last.pt、epoch_metrics.csv/jsonl、
step_metrics.jsonl、selection_metadata.json、resolved_config.json、environment.json、
source_manifest.json、data_manifest.json、checkpoint_hashes.json、stdout.log、
latest_telemetry.json、status.json、失败时failure.json；完成后FINAL_AUDIT.json、FINAL_REPORT.md、SHA256SUMS。
artifacts/另外保存CPU/CUDA测试日志、真实smoke、历史源文件/产物哈希、PID/启动/控制器日志。

checkpoint包含完整SGD momentum、调度器、Python/NumPy/CPU/CUDA RNG和来源身份，
临时文件及已发布文件均重读验证。本轮不自动resume失败运行；恢复功能通过小模型测试验证，
实际如需手动恢复需另行确认失败原因，不能覆盖重跑不理想结果。

完成后以独立CPU审计程序重算Best Epoch、检查200轮样本/步数、核验best/last来源与所有BN计数，
复核历史哈希，无额外Test选模或GPU训练。最后自动停止；不依赖聊天持续监督。

LR参考值管理按LR_REFERENCE_POLICY.md：Windows序列保持原样，正式参考由目标服务器
上的官方调度器独立生成。项目序列在相同环境下逐项严格比较400个字段；不使用容差。
每个正式epoch再次严格比较实际LR与该服务器官方参考；原跨平台差异及失败记录全部保留。
