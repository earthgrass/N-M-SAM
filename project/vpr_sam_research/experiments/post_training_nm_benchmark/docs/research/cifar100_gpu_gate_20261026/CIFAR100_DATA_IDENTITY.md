# CIFAR-100 Data Identity

## Current status

`DATA_IDENTITY_PASS`

The user-provided archive is present at:

`C:\Users\earthgrass73\Desktop\NM稀疏\data\cifar100_official_20261026\cifar-100-python.tar.gz`

## Frozen source requirements

- Official landing page: <https://www.cs.toronto.edu/~kriz/cifar.html>
- Requested URL: <https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz>
- Server redirect observed: <https://cave.cs.toronto.edu/kriz/cifar-100-python.tar.gz>
- Expected bytes: `169001437`
- Published MD5: `eb9058c3a382ffc7106e4002c42a8d85`

Verified archive SHA256: `85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7`.

The audit script refused to unpickle content before byte count and published MD5 matched. It then calculated SHA256, performed path-safe extraction, hashed every extracted file, and validated train/test arrays and labels.

## Incomplete download evidence

These files are deliberately named so they cannot be mistaken for the archive:

| File | Bytes | SHA256 | Disposition |
|---|---:|---|---|
| `download_attempt1_incomplete.bin` | 11,264,000 | `387a9fd4f7a2bad8431277bf53c5deb53fee2f92d7c8224392dac35d5345fe76` | Single-stream attempt interrupted because the official host was too slow |
| `download_attempt2_range_failed.part` | 169,001,437 | `a9f3eebb011cbf800b4e64658f8e74884b7bc828172a27a551dd7422011423a8` | Preallocated range file; TLS EOF occurred, therefore incomplete despite its logical length |
| `download_attempt3_user_stopped.part` | 169,001,437 | `cc5cf9248bc73a7d6f49a078421743c93d6512ede3fbffaf5af6fa483d4e1dff` | Preallocated range file stopped at the user's request; incomplete despite its logical length |

No incomplete file has the required archive name.

## Completed identity checks

The frozen audit entry point is `scripts/audit_cifar100_identity.py`. It verified:

1. archive size, official MD5, computed SHA256, download/source metadata and safe extraction;
2. exact extracted file inventory and per-file SHA256;
3. train `50000 × 3072` and test `10000 × 3072` `uint8` arrays, interpreted as `NCHW = N × 3 × 32 × 32`;
4. fine labels in `[0, 99]`, 100 names, train count 500/class and test count 100/class;
5. coarse labels in `[0, 19]` and 20 names;
6. exact duplicate pixel rows within each split and cross-split overlap;
7. array/label hashes, fixed 32 training IDs, and train-only normalization statistics.

All semantic checks passed:

- train: `50000 × 3072`, test: `10000 × 3072`, both `uint8`;
- fine labels span 0–99 with exactly 500 training and 100 test examples per class;
- coarse labels span 0–19; metadata contains 100 fine and 20 coarse class names;
- extracted `train`, `test`, `meta`, and zero-byte `file.txt~` inventory is complete;
- train-only channel mean: `[0.507075159237132, 0.486548873314951, 0.440917843367034]`;
- train-only population std: `[0.267334285879241, 0.256438462917088, 0.276150471325684]`.

## Exact duplicate audit

Equality means all 3072 channel-major `uint8` pixel bytes match; labels are not included in the duplicate key.

- Training split: 14 repeated rows in 14 groups; 9 groups contain differing fine labels.
- Test split: 2 repeated rows in 2 groups; both groups contain differing fine labels.
- Cross split: 10 unique pixel hashes overlap; 6 have no shared fine label between splits.

These observations are retained exactly as found. They do not indicate transfer corruption because the complete archive matches the official published MD5. No sample was removed, relabeled or selected based on the test split.

The test split was used only for identity checks. It was not used to select the model, preprocessing, pruning strategy, damping, or any hyperparameter. Full distributions, class names, hashes, duplicate counts, preprocessing and fixed sample IDs are in `GPU_GATE_TESTS/dataset_identity.json`.

