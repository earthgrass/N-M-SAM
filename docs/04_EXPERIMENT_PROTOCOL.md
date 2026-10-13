# Experiment and validation protocol

Toy runs record run_id, Git SHA, source config text/hash, seed, Python/Torch/
torchvision/package versions, device, model, dataset/split, optimizer, loss names,
augmentation, mask scope, step and actual metrics. Results are gitignored;
compact validation summaries may be committed as documentation. Toy training
accuracy is not test accuracy. Nothing is downloaded by FakeData.

Reproducibility scope: seeded single CPU FP32/FP64, torch deterministic algorithms,
one thread, materialized batches and RNG replay. Across library versions or other
hardware equality is not asserted. CUDA/AMP/DDP verification is future work.

Tests: analytic quadratic gradient locations; global epsilon; exact restoration;
rho=0 vs manual SGD; same batch and stochastic replay; BN first-update and second
freeze incl. cumulative momentum; post-g2 clip; L2 at both locations; multi-step
Nesterov recurrence; exceptions in forward/backward/optimizer; zero/nonfinite
direction; strict N:M layout, legality, enumeration and ties. No skipped planned
tests will be represented as passing tests.

Future checkpoints must include model+buffers, optimizer, scheduler, step,
Python/NumPy/torch CPU/CUDA RNG, config, mask/version/scope and sampler state.
model.state_dict alone is insufficient for exact resume. Not implemented in v0.1.

Research uses training-internal calibration for mask selection, independent
holdout for mechanism checks and final test set only for independent evaluation.
New checkpoint data must come from this project; no imported old results/weights.

