# Overnight paired SAM / VPR-L2 seeds0,1,2

This bundle adds only seeds1/2 to the already running frozen seed0 experiment.
Read PROTOCOL_FREEZE.md; configs/ contains all four new dense configurations.
workers/sam is copied from the exact completed96.77% baseline code;
workers/vpr from the seed0 cea8f8c experiment. Changes concern seed guards,
paired anchor verification, deployment identity and reporting. Geometry math,
model, augmentation, optimizer and LR scheduler remain unchanged.

Server root: /root/vpr_v2_seeds12_20261024, on the separate31GB-free filesystem.
Existing dataset and approved Python environment are read-only inputs.

```sh
cd /root/vpr_v2_seeds12_20261024/bundle_verified
nohup /root/autodl-tmp/vpr_l2_seed0_20261024/workspace/vpr_sam_research/.venv/bin/python -u overnight.py > artifacts/overnight.log 2>&1 < /dev/null &
```

All worker directories/configs are materialized and source-frozen before
waiting. No additional GPU task starts until seed0's controller is completed
and its full comparison audit is PASS. Run SAM1, VPR1+postprocess1, SAM2,
VPR2+postprocess2 sequentially. Each worker independently runs mandatory
CPU/CUDA, native exact LR, data, resource, source and disposable smoke gates.
Any failure stops further work without restart; no ongoing seed0 task is touched.

Progress/failure: bundle_verified/artifacts/queue_status.json and queue_failure.json.
Worker logs: runs/{sam,vpr}_seed{1,2}/project/artifacts/.
Final aggregate: bundle_verified/artifacts/final_seeds012/FINAL_REPORT.md,
per_seed_metrics.csv, statistics.json, evidence.json, curves and SHA256SUMS.
No notification automation is created. No additional seed after2 is authorized.

All Best endpoints use Test for checkpoint selection. Report Final alongside
Best and their losses, and do not claim an independent unbiased Test result.
The statistical unit is seed (n3), with mean/sampleSD and exploratory paired
t intervals; one shared VPR dense trajectory underlies its two Mask variants.

The original bundle/ directory retains the rejected first deployment and its
SOURCE_GATE_DIAGNOSIS.json. Its scanner excluded six official data/*.py source
files as if they were datasets; no source bytes differed. The verified bundle
corrects only source enumeration, includes a regression test, and retains
strict full-manifest equality before any new GPU task can start.
