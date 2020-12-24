import torch
import torch.nn as nn


class TestNet(nn.Module):
    def __init__(self, nlayer: int, dim: int):
        nn.Module.__init__(self)
        self.layers = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(nlayer)
        ])
        self.feedforward = nn.Linear(dim, 1)
        

    def forward(self, data: torch.Tensor):
        for layer in self.layers:
            data = layer(data)
        return self.feedforward(data)

