# APBR-GPT Joint Optimization — Phase 2 Final Report

Date: 2026-11-01 (Asia/Shanghai)

## Final status

| Gate | Status | Reason |
|---|---|---|
| `LOCAL_CALIBRATION_PASS` | **NOT CLAIMED — `PENDING_REAL_ASSET`** | Calibration interfaces passed, but the exact historical Seed 0 initial tensor asset is unavailable under the local runtime. |
| `CPU_NUMERICAL_GATE_PASS` | **PASS** | 85 selected project tests passed; the isolated bundle rerun produced 84 passed and one expected pinned-source skip. |
| `SERVER_BUNDLE_READY` | **NOT CLAIMED** | The v2 candidate bundle is internally valid, but required initial assets, real-model calibration, GPU timing gates, and formal orchestration remain explicitly blocked. |

No SSH connection, CUDA execution, Final Test access, training, Recovery, or
historical artifact overwrite occurred in Phase 2.

## Calibration evidence

Calibration source: `SYNTHETIC_INTERFACE_ONLY`.

| Quantity | Value | Acceptance |
|---|---:|---|
| Synthetic lambda_p | 0.2228357395398323 | Within [1e-6, 1e6] |
| Achieved weight-gradient ratio | 0.0500000000000000 | Target 0.05 ± 5e-5 |
| Synthetic eta_s | 1.924796974207071e-8 | Finite positive analytic solution |
| Achieved projected scale-update ratio | 0.0500000000000000 | Target 0.05, relative tolerance 0.1% |
| Energy/pruning update cosine | 0.04120855587776981 | Diagnostic only |

These values validate interfaces only and are forbidden as formal Seed 0
training constants. Real values must be calibrated once on the verified model,
fixed training batch, and frozen candidate factors.

## CPU tests

- New Joint selector plus calibration tests: 18 passed.
- Combined original APBR, Original SparseGPT, existing Joint OBS, and new
  calibration suite: 85 passed.
- Test execution from the isolated bundle: 84 passed, one expected skip because
  the ignored upstream ROSE source clone is intentionally absent.
- XML: `calibration/calibration_cpu_tests.xml`.

## Bundle integrity

- Candidate bundle manifest entries: 83.
- Extracted files including manifests: 85.
- Extracted size: 269,932,514 bytes.
- Compressed v2 size: 252,448,462 bytes.
- Manifest verification failures: 0.
- Credential hits: 0.
- Absolute Windows path hits: 0.
- Test-data hits: 0.
- `__pycache__` directories: 0.
- v2 archive SHA256:
  `377decc429d94cef4e30dfda8ae8b0bbb13d7b0554ffc1d7485c9b26638c4218`.

The first archive is retained as failed audit evidence: its long-path
extraction/old-manifest check failed. It is superseded by v2 and must not be
uploaded.

## Outstanding requirements

1. Reconstruct and freeze the exact Seed 0 initial model in the formal
   PyTorch 2.8/CUDA environment; verify expected tensor SHA256
   `87aa9dd7492a89adf9e3ab78ca4a3227f30bc8de53b18a3bd73cd61ab0bd759d`.
2. Bind the all-zero initial APBR Geometry checkpoint to that verified model.
3. Perform real one-shot lambda_p and eta_s calibration on the fixed train-only
   batch without optimizer or BN updates.
4. Implement and audit the complete formal training/pruning orchestration.
5. Pass single-batch, one-epoch, Hessian-refresh, memory, and 3x runtime GPU
   gates before enabling the guarded shell commands.

See `server_bundle/MISSING_ASSETS.md` for the machine-transfer boundary.
