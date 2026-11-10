# Frozen Blockwise APBR Experiment Protocol

The experiment contains two independently implemented methods and six from-scratch runs: `global_block_feedback` and `equal_block_energy`, each at Seeds 0, 1, and 2 for 200 epochs. The common training/data/model protocol and the actual-formula correspondence are frozen in `configs/apbr_blockwise_perturbation_cifar100_20261028.yaml` and `PREFLIGHT_AUDIT.md`.

Execution order is Seed 0 Scheme 4, Seed 0 Scheme 1, then the same pair for Seeds 1 and 2. A failure stops the queue without automatic retry. Training never constructs Test. After all six fixed Epoch-200 checkpoints are frozen, each receives one Dense Final evaluation, the unchanged formal APBR 2:4 mask generator, one Raw Hard Final evaluation, and a full mask/zero audit. No Recovery or second-order compensation is allowed.
