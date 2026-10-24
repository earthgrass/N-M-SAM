# Explicitly authorized recovery extension

The user additionally authorized exactly one 20-epoch fixed-mask SGD recovery
for Full VPR-L2 seed0. This supersedes the earlier no-recovery scope solely for
this one trajectory. Existing 200-epoch training, geometry, gamma, parameter
set, BN training rules and pruning masks remain unchanged.

Execution order: existing Dense validation -> original Raw Hard -> independent
BN-only recalibration -> this recovery from ORIGINAL, UNCALIBRATED Raw Hard.
The existing dense controller finishes before the external extension starts.
No task runs concurrently on the GPU. The independent extension does not add
source/config/script files to the immutable dense workspace or change its HEAD.
It imports the unchanged f742733 audited runtime and invokes its original
engine.train(recover_from=raw_hard), with no optimizer or math reimplementation.

The V1 seed0 recovery configuration is inherited exactly: fresh SGD,
20 epochs, batch128, lr=.005, momentum=.9, weight_decay=.0005, nesterov=false,
cosine eta_min=0, deterministic named seed0 data/augmentation streams, original
fixed mask, original training BN rules, no AMP/TF32/compile, final checkpoint.
All configuration fields are compared to recovery_config(V1 frozen seed0),
allowing only the output identity difference. Require20epochs/7040steps,
V1 recovery sample-order SHA, identical mask and Raw Hard parent SHA.

Recovery evaluation logs all20epoch validation accuracy/loss values. Primary
AULC is trapezoidal accuracy area over epochs0..20, where epoch0 is Raw Hard,
divided by20; report accuracy fractions plus raw area and the 1..20 variant.
Use the same definition for V1 Dense-SAM and VPR-SAM, without V1 checkpoint
reloads. Report final accuracy/loss, Hard->Recovery gain and Dense->Recovery
gap, with full source and checkpoint identities. No Test metrics or selecting
best validation as the final result. Dense accuracy is only a base metric;
primary questions are structured pruning robustness and recoverability.

Fail closed on provenance, numeric, hardware or space errors, never retry or
restart automatically. No other seed, recovery, hyperparameter or mask change.
The existing dense/non-training reports are retained as historical scope;
recovery-inclusive final reports and audit explicitly supersede that scope.
After this one recovery and CPU report generation, stop all further GPU tasks.
