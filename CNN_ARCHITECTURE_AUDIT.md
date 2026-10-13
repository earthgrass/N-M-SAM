# CNN Architecture Audit

Model: ResNet18-CIFAR10. Input is `3 x 32 x 32`; the stem is a 3x3, stride-1, padding-1 convolution; there is no leading max-pool; the classifier has 10 outputs.

N:M layout: Conv2d `[out,in,kh,kw] -> [out,in*kh*kw]`, Linear `[out,in]`; full blocks use 2:4, tails remain dense. Sparse scope shown: `phase1`.

| layer_name | type | input/output | kernel | stride | weight shape | bias | BatchNorm | N:M | blocks | tail weights | decision |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---|
| conv1 | Conv2d | 3 -> 64 | (3, 3) | (1, 1) | (64, 3, 3, 3) | no | bn1 | no | 0 | 0 | Phase-1 default excludes first convolution |
| layer1.0.conv1 | Conv2d | 64 -> 64 | (3, 3) | (1, 1) | (64, 64, 3, 3) | no | layer1.0.bn1 | yes | 9216 | 0 | selected by scope=phase1 |
| layer1.0.conv2 | Conv2d | 64 -> 64 | (3, 3) | (1, 1) | (64, 64, 3, 3) | no | layer1.0.bn2 | yes | 9216 | 0 | selected by scope=phase1 |
| layer1.1.conv1 | Conv2d | 64 -> 64 | (3, 3) | (1, 1) | (64, 64, 3, 3) | no | layer1.1.bn1 | yes | 9216 | 0 | selected by scope=phase1 |
| layer1.1.conv2 | Conv2d | 64 -> 64 | (3, 3) | (1, 1) | (64, 64, 3, 3) | no | layer1.1.bn2 | yes | 9216 | 0 | selected by scope=phase1 |
| layer2.0.conv1 | Conv2d | 64 -> 128 | (3, 3) | (2, 2) | (128, 64, 3, 3) | no | layer2.0.bn1 | yes | 18432 | 0 | selected by scope=phase1 |
| layer2.0.conv2 | Conv2d | 128 -> 128 | (3, 3) | (1, 1) | (128, 128, 3, 3) | no | layer2.0.bn2 | yes | 36864 | 0 | selected by scope=phase1 |
| layer2.0.shortcut.0 | Conv2d | 64 -> 128 | (1, 1) | (2, 2) | (128, 64, 1, 1) | no | layer2.0.shortcut.1 | no | 0 | 0 | Phase-1 default excludes 1x1/downsample convolution |
| layer2.1.conv1 | Conv2d | 128 -> 128 | (3, 3) | (1, 1) | (128, 128, 3, 3) | no | layer2.1.bn1 | yes | 36864 | 0 | selected by scope=phase1 |
| layer2.1.conv2 | Conv2d | 128 -> 128 | (3, 3) | (1, 1) | (128, 128, 3, 3) | no | layer2.1.bn2 | yes | 36864 | 0 | selected by scope=phase1 |
| layer3.0.conv1 | Conv2d | 128 -> 256 | (3, 3) | (2, 2) | (256, 128, 3, 3) | no | layer3.0.bn1 | yes | 73728 | 0 | selected by scope=phase1 |
| layer3.0.conv2 | Conv2d | 256 -> 256 | (3, 3) | (1, 1) | (256, 256, 3, 3) | no | layer3.0.bn2 | yes | 147456 | 0 | selected by scope=phase1 |
| layer3.0.shortcut.0 | Conv2d | 128 -> 256 | (1, 1) | (2, 2) | (256, 128, 1, 1) | no | layer3.0.shortcut.1 | no | 0 | 0 | Phase-1 default excludes 1x1/downsample convolution |
| layer3.1.conv1 | Conv2d | 256 -> 256 | (3, 3) | (1, 1) | (256, 256, 3, 3) | no | layer3.1.bn1 | yes | 147456 | 0 | selected by scope=phase1 |
| layer3.1.conv2 | Conv2d | 256 -> 256 | (3, 3) | (1, 1) | (256, 256, 3, 3) | no | layer3.1.bn2 | yes | 147456 | 0 | selected by scope=phase1 |
| layer4.0.conv1 | Conv2d | 256 -> 512 | (3, 3) | (2, 2) | (512, 256, 3, 3) | no | layer4.0.bn1 | yes | 294912 | 0 | selected by scope=phase1 |
| layer4.0.conv2 | Conv2d | 512 -> 512 | (3, 3) | (1, 1) | (512, 512, 3, 3) | no | layer4.0.bn2 | yes | 589824 | 0 | selected by scope=phase1 |
| layer4.0.shortcut.0 | Conv2d | 256 -> 512 | (1, 1) | (2, 2) | (512, 256, 1, 1) | no | layer4.0.shortcut.1 | no | 0 | 0 | Phase-1 default excludes 1x1/downsample convolution |
| layer4.1.conv1 | Conv2d | 512 -> 512 | (3, 3) | (1, 1) | (512, 512, 3, 3) | no | layer4.1.bn1 | yes | 589824 | 0 | selected by scope=phase1 |
| layer4.1.conv2 | Conv2d | 512 -> 512 | (3, 3) | (1, 1) | (512, 512, 3, 3) | no | layer4.1.bn2 | yes | 589824 | 0 | selected by scope=phase1 |
| linear | Linear | 512 -> 10 | - | - | (10, 512) | yes | - | yes | 1280 | 0 | selected by scope=phase1 |
