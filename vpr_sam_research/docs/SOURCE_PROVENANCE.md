# Source custody and publication scope

Frozen producer commits:

- V1 training: d4c58d66b248b620e0e058b92e79100f173970a6
- VPR-L2 dense and magnitude runtime: f7427339de6d6589f53b0191068448cf642901b2
- Three-mask diagnostics: 492c5976032741c17600acfcbc823bbf7569789b
- Magnitude recovery extension: f0e3a4acac8650370afb9865ff82a4765d2af676
- Guided recovery: 1345d24d557e88974eb6cc5cadabf4cae754d742

results/research_delivery/EXPERIMENT_PROVENANCE_MANIFEST.json binds original
logs/configs and checkpoint hashes. Private checkpoints, geometry tensors,
complete Masks and CIFAR data remain on the approved server. Obtain an
authorized copy from the project owner and verify SHA256; no public binary
download endpoint or server credential is published. An independent offsite
backup of large checkpoints has not been verified by this delivery.

New results are small evidence only. Public logical run IDs replace private
host/account paths. Historical branches and metrics remain unchanged. The
archive is a new branch, not a migration of an old frozen producer checkout;
never use its changed source tree to bypass historical provenance validation.

The candidate prototype has a separate branch/PR. Neither Draft PR is merged.
