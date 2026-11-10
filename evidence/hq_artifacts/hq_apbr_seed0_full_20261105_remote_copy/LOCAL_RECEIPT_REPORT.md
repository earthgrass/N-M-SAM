# HC-SAM Seed0 local receipt and audit

| Endpoint | Accuracy |
|---|---:|
| HC-SAM Dense Final | 78.51% |
| Final-Q Mask-only | 48.06% |
| Final-Q + Fixed-mask OBS | 74.72% |
| HC-SAM + Native SparseGPT | 76.59% |

200 epochs / 78,200 steps; 200 training-only calibration refreshes. Strict 2:4 checked for all three sparse endpoints. No Recovery or new Test evaluation.
Original performance gate FAIL retained; explicit research override used.
All remote copies preserved; local SHA256 and checkpoint/Mask audits PASS.
