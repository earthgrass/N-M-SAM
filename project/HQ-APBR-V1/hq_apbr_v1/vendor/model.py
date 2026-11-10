"""Independent 100-class wrapper around the unchanged audited model source."""
from collections import OrderedDict
from torch import nn

from .model_source import BasicBlock, ResNet

MODEL_ID = "ssam2022_cifar_resnet18_c100_v1"


def build_model() -> ResNet:
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=100)


def eligible_layers(model: nn.Module) -> OrderedDict[str, nn.Conv2d]:
    layers = [(name, module) for name, module in model.named_modules()
              if isinstance(module, (nn.Conv2d, nn.Linear))]
    return OrderedDict(layers[1:-1])
