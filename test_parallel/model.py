import torch
import torch.nn as nn


class TestNet(nn.Module):
    def __init__(self, nlayer: int, dim: int):
        self.layers = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(nlayer)
        ])
        

    def forward(self, data: torch.Tensor):
        return self.layers(data)

