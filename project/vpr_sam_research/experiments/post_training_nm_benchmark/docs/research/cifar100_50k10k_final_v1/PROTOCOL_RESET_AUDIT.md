# CIFAR-100 50k/10k protocol reset audit

## Decision

The formal protocol is reset to all 50,000 official training examples, no
validation split, and fixed Final checkpoint selection. The official 10,000
test examples are absent from the training entry point and are available only
through the independent final evaluator.

## Old run disposition

The running 45k/5k Dense-SAM Seed 0 process was positively identified by PID,
working directory, full command line, output identity and GPU UUID. It was
normally terminated at epoch 14 with `SIGTERM`, then deleted with user
authorization after hashes were captured. It is classified
`ABORTED_PROTOCOL_MISMATCH_45K5K`; resume and formal reporting are forbidden.

## New protocol gates

- Official archive: 169,001,437 bytes; MD5
  `eb9058c3a382ffc7106e4002c42a8d85`; SHA256
  `85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7`.
- Train pickle: 50,000 uint8 rows, 500 per fine class.
- Test pickle: 10,000 uint8 rows, 100 per fine class.
- Train loader: shuffle enabled, `drop_last=false`, 391 batches and exactly
  50,000 unique sample positions per epoch.
- Dense/APBR budget: fixed 200 epochs; 78,200 optimizer steps.
- Selection: no Best checkpoint; `final_epoch_200.pt` is the only primary
  dense parent.
- Recovery: fixed 20 epochs; `final_epoch_020.pt` is the only primary recovery
  checkpoint.
- Old-protocol checkpoints fail the new checkpoint validator by protocol ID.

## Test isolation code audit

`run_cifar100_50k_formal_training.py` opens only the extracted `train` pickle.
It constructs no Test dataset or loader and contains no Test/validation metric,
early stopping, `best_test.pt`, or `best_val.pt` path. Epoch checkpoints are
published atomically to rolling `last.pt`; completion publishes
`final_epoch_200.pt` only after exact epoch/step budget checks.

`evaluate_cifar100_50k_final.py` is a separate read-only evaluator. It accepts
only an epoch-200 dense Final, Raw-Hard checkpoint, or epoch-20 Recovery Final;
it records the exact checkpoint SHA256 and has no optimizer, selection, pruning
or tuning path. It has not been run for model selection.

## Automated verification

The corrected suite completed with `24 passed`. The tests cover the frozen
50k/10k configuration, all-sample reachability, Test-loader absence, fixed
Final selection, rejection of old 45k/5k checkpoints, model/data identities,
and the previously accepted baseline adapter contracts.

Status before remote preflight and Git freeze: `LOCAL_GATES_PASS`.
