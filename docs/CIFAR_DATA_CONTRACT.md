# Offline CIFAR contract

Local archive only, never downloaded or uploaded. SHA256 identifies the archive;
official torchvision CIFAR Python file MD5 values are verified BEFORE unpickling.
Reject absolute/parent/Windows paths, links, devices, duplicates, >100 members or
>512 MiB unpacked size. Extract member-by-member to temporary external cache,
verify, then rename atomically. Cached files are reverified on every reuse.

Train/test expected 50000/10000; seed-fixed training-only 45000/5000 split.
Train crop/flip/normalize; calibration/validation/test deterministic normalize.
No formal test samples select masks. Windows Unicode paths supported via pathlib.
