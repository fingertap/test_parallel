import torch
from torch.utils.data import Dataset


class TestDataSet(Dataset):
    def __init__(self, length: int, shape: torch.Size):
        self.length = length
        self.shape = shape

    def __len__(self):
        return self.length

    def __getitem__(self, item: int) -> torch.Tensor:
        return torch.rand(self.shape)
        