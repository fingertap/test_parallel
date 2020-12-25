# Test PyTorch DataParallel and DistributedDataParallel

Using a single GPU:

```bash
$ python test_parallel/single_gpu.py
2020-12-25 15:06:09,641 [INFO] <single gpu> Run on 1 gpus with batchsize = 80.
2020-12-25 15:06:38,735 [INFO] <single gpu> Time ellapsed: 29.093183994293213ss
```

Using `torch.nn.DataParallel`:

```bash
$ python test_parallel/data_parallel.py
2020-12-25 14:59:04,152 [INFO] <data parallel> Run on 8 gpus with batchsize = 640.
2020-12-25 14:59:59,101 [INFO] <data parallel> Time ellapsed: 54.948784828186035s
```

Using `torch.nn.DistributedDataParallel`:

```bash
$ CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 test_parallel/distributed_parallel_cmd.py
*****************************************
Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
*****************************************
2020-12-25 15:03:34,742 [INFO] <data parallel> Run on 8 gpus with batchsize = 80.
2020-12-25 15:03:34,743 [INFO] <data parallel> Run on 8 gpus with batchsize = 80.
2020-12-25 15:03:34,748 [INFO] <data parallel> Run on 8 gpus with batchsize = 80.
2020-12-25 15:03:34,751 [INFO] <data parallel> Run on 8 gpus with batchsize = 80.
2020-12-25 15:03:34,752 [INFO] <data parallel> Run on 8 gpus with batchsize = 80.
2020-12-25 15:03:34,752 [INFO] <data parallel> Run on 8 gpus with batchsize = 80.
2020-12-25 15:03:34,752 [INFO] <data parallel> Run on 8 gpus with batchsize = 80.
2020-12-25 15:03:34,753 [INFO] <data parallel> Run on 8 gpus with batchsize = 80.
2020-12-25 15:03:42,484 [INFO] <data parallel> Time ellapsed: 7.740724802017212s
2020-12-25 15:03:42,484 [INFO] <data parallel> Time ellapsed: 7.7421228885650635s
2020-12-25 15:03:42,484 [INFO] <data parallel> Time ellapsed: 7.7332093715667725s
2020-12-25 15:03:42,484 [INFO] <data parallel> Time ellapsed: 7.7319347858428955s
2020-12-25 15:03:42,484 [INFO] <data parallel> Time ellapsed: 7.735737562179565s
2020-12-25 15:03:42,485 [INFO] <data parallel> Time ellapsed: 7.731153964996338s
2020-12-25 15:03:42,485 [INFO] <data parallel> Time ellapsed: 7.7325520515441895s
2020-12-25 15:03:42,487 [INFO] <data parallel> Time ellapsed: 7.733853816986084s
```
