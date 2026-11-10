# Mask and compensation endpoint comparison

All endpoint metadata is in `mask_compensation_endpoint_comparison.csv`.

## Shared identity

- Dense parent SHA256: `5bf3011c4fc15c999c6dc62f6a2d6b7a2ffebcd6931145367f58e8fbee5fdbdc`
- Calibration-ID SHA256: `3b7da27d79762235c87f18571bc73224d6e6f2b3ba663b7d9a4a3de312f23157`
- Config SHA256: `56ff197b0860dfd0f4a0768c5609031214e6405c51557009a9977582b38400e7`
- Eligible mapping SHA256: `99e0e044a7e4aea0c9f547d3283bdfd5e48f8081a0a7a65ff74491bcd9272c6a`
- BN state SHA256: `263e76f336191cf41addbf0c66688704b60a3b9ba3a1c9c7c4a04179df6c0444`
- Epoch index 199; 200 completed Epochs; recovery 0
- Relative damping 0.01; block size 128; FP32 pruning compute

## What can and cannot be attributed

Mask-only and Fixed-mask OBS use the identical Mask SHA256 `5e5aeb5a51dec89e8387447a706948b1e9c7abedecee0b7923b6584061eaea8c`. Their difference is a valid same-support compensation comparison:

- accuracy: 48.06% → 74.72%, +26.66 pp;
- CE: 2.084161 → 0.960037, -1.124123.

Native SparseGPT uses Mask SHA256 `6d9c2e54985f4e77dd97a2b90f68ef2576e8102b6319cc615b9204b6507f18f0`. It reaches 76.59%, but its 1.87 pp advantage over fixed OBS combines support selection and the native dynamic execution path. It must not be called a compensation-only gain.

Static/native coordinate agreement is 79.23%, and identical-group rate is 60.66%. Static/magnitude agreement is 95.28% by coordinate and 90.62% by complete group. These figures show that the static Final-Q branch remained close to magnitude selection, whereas Native SparseGPT materially changed support through its dynamic process.

