# APBR-GPT Joint local calibration

- Status: `PENDING_REAL_ASSET`
- Calibration source: `SYNTHETIC_INTERFACE_ONLY`
- Final Test accessed: `false`
- CUDA used: `false`

## lambda_p

- Value: `0.22283573953983229`
- CE eligible-gradient norm: `14.232480934860922`
- Pruning eligible-gradient norm: `3.1934915297366024`
- Achieved ratio: `0.050000000000000003`
- Acceptance interval: `[1e-06, 1000000.0]`

## eta_s

- Value: `1.924796974207071e-08`
- Energy-update norm: `1.1509977316957351e-06`
- Independent pruning-update norm: `5.7549886584786755e-08`
- Achieved ratio: `0.050000000000000003`
- Energy/pruning cosine: `0.041208555877769809`
- Combined projected update norm: `1.1548017405167878e-06`

These numerical values validate the calibration interfaces only. They are not
authorized formal-training constants because the historical Seed 0 initial
tensor state cannot be reconstructed under the current local PyTorch runtime.
They must be recalibrated once on the verified server model and fixed training
batch before formal training.
