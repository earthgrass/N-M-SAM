from torch import nn


class TinyMLP(nn.Module):
    """[B,1,4,4] -> [B,3], with BN to exercise state semantics."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(16, 8), nn.BatchNorm1d(8),
                                 nn.Tanh(), nn.Linear(8, 3))

    def forward(self, inputs):
        return self.net(inputs)

