import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, DataLoader

from data import TestDataSet
from model import TestNet
from utils.logging import LoggingModule


class Logger(LoggingModule):
    pass


device = 'cuda'
batch_size = 80
nlayers = 5
ndata = 500000
dmodel = 1000
lr = 0.1
dataset = TestDataSet(ndata, dmodel)
sampler = RandomSampler(dataset)
dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
model = TestNet(nlayers, dmodel).to(device)
optimizer = optim.SGD(model.parameters(), lr)
logger = Logger('single gpu')

logger.info('Run on {} gpus with batchsize = {}.'.format(
    torch.cuda.device_count(), batch_size
))
start_timestamp = time.time()

# Forward and backward pass
for batch in dataloader:
    optimizer.zero_grad()
    batch = batch.to(device)
    result = model(batch)
    result.mean().backward()

logger.info('Time ellapsed: {}s'.format(time.time() - start_timestamp))
