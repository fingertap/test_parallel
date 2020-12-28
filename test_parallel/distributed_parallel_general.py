import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from model import TestNet
from data import TestDataSet
from utils.logging import LoggingModule


class Logger(LoggingModule):
    pass


batch_size = 80
nlayers = 5
ndata = 500000
dmodel = 1000
lr = 0.1

def worker(rank, world_size):
    dist.init_process_group(
        backend='nccl',
        rank=rank,
        world_size=world_size,
        init_method="file:///tmp/helloworld"
    )
    model = nn.parallel.DistributedDataParallel(
        TestNet(nlayers, dmodel).to(rank),
        device_ids=[rank],
        output_device=[rank]
    )

    dataset = TestDataSet(ndata, dmodel)
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size, sampler=sampler)
    optimizer = optim.SGD(model.parameters(), lr)
    logger = Logger('DDP_general')

    if rank == 0:
        logger.info('Run on {} gpus with batchsize = {}.'.format(
            world_size, batch_size
        ))
    start_timestamp = time.time()

    # Forward and backward pass
    for batch in dataloader:
        optimizer.zero_grad()
        batch = batch.to(rank)
        result = model(batch)
        result.mean().backward()

    if rank == 0:
        logger.info('Time ellapsed: {}s'.format(time.time() - start_timestamp))

if __name__ == '__main__':
    world_size = 8
    mp.spawn(
        worker, args=(world_size,), nprocs=world_size, join=True
    )
    