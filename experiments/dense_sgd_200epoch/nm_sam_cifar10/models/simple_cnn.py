"""Tiny CNN used purely for smoke testing the SAM / N:M pipeline."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.max_pool2d(out, 2)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = F.max_pool2d(out, 2)
        out = out.flatten(1)
        out = F.relu(self.fc1(out), inplace=True)
        return self.fc2(out)
