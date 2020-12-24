# Test PyTorch DataParallel and DistributedDataParallel

To run the test:

```bash
$ python test_parallel/single_gpu.py
$ python test_parallel/data_parallel.py
$ CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 test_parallel/distributed_parallel.py
```