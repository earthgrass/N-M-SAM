# Baseline Pre-flight Audit

Status: **PASS**

## Dataset audit

- train_dataset_size = 50000
- train_eval_dataset_size = 50000
- test_dataset_size = 10000
- num_classes = 10
- no subset is used

## Transform audit

- train: RandomCrop(32,padding=4) -> RandomHorizontalFlip -> ToTensor -> Cutout -> Normalize
- Cutout: one 16x16 hole; implementation `data/strong_baseline_transforms.py`; applied in the [0,1] tensor domain before normalization
- test: ToTensor -> Normalize only
- clean train-eval: ToTensor -> Normalize only
- mean = (0.4914, 0.4822, 0.4465); std = (0.247, 0.2435, 0.2616)
- MixUp/CutMix/AutoAugment/RandAugment/EMA/label smoothing: disabled

## Architecture audit

- model: CIFAR-ResNet18
- stage shapes: `{"input": [2, 3, 32, 32], "stem": [2, 64, 32, 32], "layer1": [2, 64, 32, 32], "layer2": [2, 128, 16, 16], "layer3": [2, 256, 8, 8], "layer4": [2, 512, 4, 4], "gap": [2, 512], "logits": [2, 10]}`
- expected shapes matched: True
- parameters: `{"total_parameters": 11173962, "trainable_parameters": 11173962, "conv2d_parameters": 11159232, "linear_parameters": 5130, "batchnorm_learnable_parameters": 9600, "fp32_parameter_memory_mib": 42.625282287597656}`

## Dense-only audit

- number_of_sparse_wrappers = 0
- number_of_nm_masks = 0
- number_of_sparse_parameters = 0
- ordinary Conv2d/Linear dense forward only

## Optimizer audit

- optimizer class: SGD
- param groups: `[{"lr": 0.1, "momentum": 0.9, "dampening": 0, "weight_decay": 0.0005, "nesterov": false, "maximize": false, "foreach": null, "differentiable": false, "fused": null, "parameter_tensors": 62}]`
- every trainable tensor appears exactly once: True
- no SAM wrapper

## Scheduler audit

- CosineAnnealingLR, T_max=200, eta_min=0
- epoch-level; `scheduler.step()` runs after each completed epoch
- epoch 1 trains at lr=0.1; no pre-epoch scheduler step
- warmup=0

## Reproducibility and precision

- seed=0 for Python, NumPy, and PyTorch; CUDA all-device seed if applicable
- FP32 only; no AMP/BF16/FP16
- environment: `{"os": "Windows-11-10.0.26200-SP0", "python": "3.12.10 (tags/v3.12.10:0cc8128, Apr  8 2025, 12:21:36) [MSC v.1943 64 bit (AMD64)]", "pytorch": "2.13.0+cpu", "torchvision": "0.28.0+cpu", "device": "cpu", "cpu": "Intel64 Family 6 Model 197 Stepping 2, GenuineIntel", "logical_processors": 16, "cuda_available": false, "cuda_version": null, "cudnn_version": null, "deterministic_algorithms": false, "cudnn_deterministic": false, "cudnn_benchmark": false, "precision": "FP32"}`
- deterministic algorithms=False; cuDNN benchmark=False

## Failures

- none
