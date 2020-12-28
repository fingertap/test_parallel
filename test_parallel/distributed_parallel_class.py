import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import List


from model import TestNet


class DDPModel:
    def __init__(self, model: nn.Module, device_ids: List[int]):
        pass
