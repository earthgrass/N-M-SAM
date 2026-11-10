# SAM Seed 2：77.43% 的来源与本次回传

77.43% 是历史 SAM Seed 2 Epoch-200 Dense Final 经条件贪心 2:4 选择与联合 OBS 补偿后的固定 Final Test 结果：10,000 张中正确 7,743 张，CE 0.839221。该实验使用 1,024 张固定训练图片构建 Hessian；先按物理单坐标 OBS 代价选第一删除位置，再虚拟条件更新选择第二位置，最后对选定两坐标联合补偿。19 层 Mask 的 5,578,752 个删除位置、严格 2:4、零值保持、父模型哈希和 Test 前冻结均经独立审计。

本次新服务器的 SAM Seed 2 也完成 200 Epoch、78,200 steps，并回传完整 Final checkpoint 与身份、配置、逐轮日志。新 checkpoint SHA256 为 `070c8d04af80a0bf011647aa8bad65e65e3fd64bedde64780adc029b40b07ec4`；历史父 checkpoint SHA256 为 `558ffa8da0a886a1291746fb6b5d979a2e83938b1150136568edea5e116fb977`。两份 checkpoint 的配置和元数据不同，但 **122 个模型状态张量（参数与 BN buffers）逐项 Bitwise Equal**，模型状态 SHA256 同为 `7f95f8c13548d1d3edaeab45f9f6efe6fd4d3ae70e950b880361bf24e694274d`。因此历史 77.43% 的补偿实验以与本次 Final 完全相同的模型状态为起点；本次没有重复剪枝，也没有再次访问 Test。

用户要求在 SAM Seed 2 完成并回传后停止。串行队列控制器先被暂停，SAM Seed 2 子进程正常完成；验证远端与本地 SHA256、200 Epoch/78,200 steps 和训练期无 Test 访问后，控制器已终止。Seed 3 未启动。原队列状态文件保留为停止前快照，独立停止记录说明最终状态。

证据：[新 Final 回传证明](../../../artifacts/dense_sam_apbr_l2_seed1to4_20261103_local_copy/seed2/dense_sam/LOCAL_SHA256.json)、[逐张量身份核验](../../../artifacts/dense_sam_apbr_l2_seed1to4_20261103_local_copy/seed2/dense_sam/DENSE_IDENTITY_AND_RESULT_ORIGIN.json)、[历史条件 OBS 结果](../../../artifacts/sam_conditional_seed2_cpu_20261104_v1/results.json)、[历史独立审计](../../../artifacts/sam_conditional_seed2_cpu_20261104_v1/independent_audit.json)、[停止记录](../../../artifacts/dense_sam_apbr_l2_seed1to4_20261103_local_copy/USER_STOP_AFTER_SAM_SEED2_20261104.json)。
