# 结果来源检查

本轮只读既有结果，不进行任何重新评测或实验聚合替换。

## 已验证旧版本

实验根：
`C:/Users/earthgrass73/Desktop/NM稀疏/APBR-post-training-benchmark/vpr_sam_research/experiments/post_training_nm_benchmark`

| 数值 | 原始JSON字段 | 版本 | 新论文处置 |
|---|---|---|---|
| 77.83% | `artifacts/apbr_nearoptimal_conditional_seed0_full_cpu_20261103_v1/results.json::raw_hard_test.true.test_accuracy` | APBR-VPR Seed0，tau=.01，true | 审计历史 |
| 77.43% | `artifacts/sam_apbr_conditional_seed1_cpu_20261103_v1/results.json::test.apbr_geo_conditional.test_accuracy` | APBR-VPR Seed1，tau=.01，true | 审计历史 |
| 77.63±0.28% | `docs/research/sam_apbr_l2_conditional_seed01_20261103/comparison.json::rows.apbr_vpr_geometric` | 上述两个运行的既有汇总 | 不归给无tau方法 |
| SAM 77.43% | 原稿消融表未绑定JSON；真实SAM Seed2的 `test.physical.test_accuracy` 也有0.7743 | 论文该格来源映射UNKNOWN | UNVERIFIED，移出新正式表 |
| 无tau APBR-first完整方法 | 未发现已验证对应评测记录 | 本轮论文定义 | PENDING |

旧原稿表格仍保存在工程外原论文及本修订 `before/` 中。没有将UNKNOWN标记成实验失败，也没有将新方法PENDING填为零。

## 原参考行

SAM `55.54±1.36`、Wanda `56.48±3.83`、SparseGPT `77.26±0.42` 逐字保留自原稿 `paper/tables/own_results.tex`。本轮不重算它们或重新审计整个三Seed矩阵。新表注明它们是已完成参考行：前两项mask-only，SparseGPT含native OBS。新APBR结果为PENDING，因而不存在把77.63当作无tau完成结果的跨版本排名。

## 已重算的小文件SHA256

| 文件 | SHA256 |
|---|---|
| nearoptimal adapter | `24b7eb92eae8fbb1e0e9dbf88eafa06de9673898318ffd0770a80a0a1759847a` |
| Seed0 runner | `f5371f7a5fffacde6399017b41e26e60431e4c96a84352630845eb58c0aebab8` |
| Seed1 runner | `49085de0b997aa5c4e14b69e55798024ea228ccc95013d8c14c841ecffc6abb6` |
| Seed0 preflight | `7c078f3e07d92f8e71da106831aff075750ddaed526f4139bc8343b9b4cd441f` |
| Seed0 pre_test_freeze | `7e82fed6c8d04a0a1751ae1de9aa3e875752157c2b0e5829cbeb768796e82f9d` |
| Seed0 results | `a928dddbd9a431a2d6f1cb98a1ee3210484efe1b3115d67ce0237e6ca2a2dbe8` |
| Seed1 preflight | `cfe9bb5cccd359b8dcac095d692177424a54f90848b6733b6976370e0e9c7bac` |
| Seed1 pre_test_freeze | `ba00b0aa1a46851b3eb25e985cec5d8c9e701f18f02e8a8f42054a0b3ffe569e` |
| Seed1 results | `d545fd5c59b515f9c22187eca347050038d6719a1bb8009e354fc52962536e55` |

上述source哈希与preflight/freeze/results相符，支持对实际算法版本的判定。大checkpoint和mask哈希详见来源审计；本轮未重新读取其全部内容。
