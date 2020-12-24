import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data import TestDataSet
from model import TestNet
from utils.logging import LoggingModule


class Logger(LoggingModule):
    pass


batch_size = 80
nlayers = 5
ndata = 500000
dmodel = 1000
lr = 0.1

dist.init_process_group(backend='nccl')
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

dataset = TestDataSet(ndata, dmodel)
sampler = DistributedSampler(dataset)
dataloader = DataLoader(dataset, batch_size, sampler=sampler)
model = nn.parallel.DistributedDataParallel(
    TestNet(nlayers, dmodel).to(device),
    device_ids=[local_rank],
    output_device=local_rank
)
optimizer = optim.SGD(model.parameters(), lr)
logger = Logger('data parallel')

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

