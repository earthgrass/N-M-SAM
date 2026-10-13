# Implementation-v1 actual experiment log

USER OVERRIDE RECEIVED after M3 code creation: stop all research/training smoke.
An already-completed M3 CLI produced effective_smoke_a5e241fc4ae544b0ab95cae72889c1b4
before the stop could cancel it. It is historical only. No additional dataset reads
or research execution follow this override. Subsequent entries are UNIT_TESTED.
M3 unit-only suite: 80 passed (actual runtime recorded by test output), plus compileall.
M4: 82 passed in unit-only suite; constant-Hessian six-candidate identity, exact
residual decomposition, zero-radius undefined direction, compatible L2 subtraction,
constant ranking and label rejection. Numerical equality is unit evidence only,
not experimental support for route A. Dataset/GPU mechanism validation pending.
M5 unit suite: 88 passed; four curvature classes over six supports, KKT/grid
reference, hard case, zero-gradient one-step counterexample, new support domain,
DeltaF=DeltaL+DeltaS and exact quadratic local-model identity. No network research
experiment was executed. These are UNIT_TESTED, not evidence of improved accuracy.

No historical results are reused. Each stage below is updated only after execution.

M2.0 P0: data source was missing from the stable Git tree due to unanchored data/.
Fix anchors ignores and explicitly tracks data source. G0 PASSED: independent
remote clone 5cb5556, 58 passed in 10.25s; Dense-SAM CLI 3 steps,
CE=1.3439754247665405, epsilon=.05000000074505806, restoration=0.
Clone installation used existing dependencies with --no-deps --no-build-isolation.
Local archive safety/split tests: 65 passed in 7.34s. Initial 22642688-byte
archive rejected with EOFError; replacement archive is being validated separately.
G1 PASSED: replacement size=170498071; archive SHA256
6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce;
split SHA256=62d58f141ce65ae7e8cdc3f9e2f8c3d0faffadb95de902715bb2afd4e32c48ba.
Actual counts=50000/10000; train-core=45000, validation=5000, classes=10.
CPU TinyCNN two-batch SAM CE=2.4539566040,2.3158249855; epsilon=.05,
restoration=0, 2 forwards each. This is a data/runtime smoke, not accuracy evidence.

M2.1 G2 PASSED: 73 passed in 6.42s. Pure/STE inactive gradients, allowed global
epsilon, mask lock, BN one update, rho=0 and fault rollback verified.
FakeData and real CIFAR four-core methods each ran two batches.
CIFAR final CE: dense SGD 2.3202581; dense SAM 2.3673310;
sparse SGD 2.3439302; sparse SAM 2.3931329. Sparse epsilon=.05, restoration=0.
This core smoke uses dense L2=0 and sparse default master L2=.001 and is NOT
a fair method comparison. M2.2 will enforce one explicit L2 protocol for all six.

M2.2 G3 PASSED: 75 passed in 7.15s; six methods each completed 2 batches of
FakeData and real CIFAR with common L2=.001, seed0, constant LR and shared batches.
Actual CIFAR artifact run_id=a92364ecb8024b64945b97d4bd27e59f, code source e91872f
plus recorded dirty tree. Final CE in method order: 2.5459023,2.5925188,2.5381720,
2.5787444,2.5750530,2.5391388. Mask updates occur only after complete step;
Taylor scores W_pre*g_pre. These perturbed-point training CE values do not imply
better test generalization. Full local JSON records timing, flips, legality and labels.
