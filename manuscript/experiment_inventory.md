# Experiment inventory

| Cohort | Seeds | Training | Pruning endpoint | Final Test | Hardware |
|---|---|---|---|---|---|
| Historical CIFAR-100 formal | 0,1,2 | Dense-SAM/APBR-SAM, 200 epochs | Native SparseGPT; other methods | Fixed final only | Seed 0/1 Dense-SAM on RTX 4080 SUPER; other stages differ |
| Joint rebased exploratory | 0,1 | SAM-Joint/APBR-Joint, 200 epochs | Native SparseGPT and separate groupwise variants | Fixed final only | Seed 1 on RTX 5090 |
| Temporal sparse Joint | 0,1 | APBR-Joint guided at epochs 10,20,...,200 | Planned native SparseGPT and variants | Pending | RTX 5090, running |

All numerical paper claims are tied to the copied Final Test JSON, historical analysis CSV/Markdown, or timing CSV. Training and pruning do not access Test; evaluation is after checkpoint freeze. No Recovery-20 in the Joint results.
