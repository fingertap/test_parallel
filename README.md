# Test PyTorch DataParallel and DistributedDataParallel

Using a single GPU:

```bash
$ python test_parallel/single_gpu.py
2020-12-24 16:42:53,301 [INFO] <single gpu> Run on 10 gpus with batchsize = 10.
2020-12-24 16:45:09,088 [INFO] <single gpu> Time ellapsed: 135.78714036941528s
```

Using `torch.nn.DataParallel`:

```bash
$ python test_parallel/data_parallel.py
2020-12-24 16:46:08,315 [INFO] <data parallel> Run on 8 gpus with batchsize = 80.
2020-12-24 16:49:42,705 [INFO] <data parallel> Time ellapsed: 214.38877725601196s
```

Using `torch.nn.DistributedDataParallel`:

```bash
$ CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 test_parallel/distributed_parallel.py
*****************************************
Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
*****************************************
2020-12-24 16:36:38,113 [INFO] <data parallel> Run on 8 gpus with batchsize = 80.
2020-12-24 16:36:38,122 [INFO] <data parallel> Run on 8 gpus with batchsize = 80.
2020-12-24 16:36:38,124 [INFO] <data parallel> Run on 8 gpus with batchsize = 80.
2020-12-24 16:36:38,125 [INFO] <data parallel> Run on 8 gpus with batchsize = 80.
2020-12-24 16:36:38,127 [INFO] <data parallel> Run on 8 gpus with batchsize = 80.
2020-12-24 16:36:38,128 [INFO] <data parallel> Run on 8 gpus with batchsize = 80.
2020-12-24 16:36:38,129 [INFO] <data parallel> Run on 8 gpus with batchsize = 80.
2020-12-24 16:36:38,130 [INFO] <data parallel> Run on 8 gpus with batchsize = 80.
2020-12-24 16:36:45,794 [INFO] <data parallel> Time ellapsed: 7.6636962890625s
2020-12-24 16:36:45,794 [INFO] <data parallel> Time ellapsed: 7.669627666473389s
2020-12-24 16:36:45,794 [INFO] <data parallel> Time ellapsed: 7.668665885925293s
2020-12-24 16:36:45,794 [INFO] <data parallel> Time ellapsed: 7.680647611618042s
2020-12-24 16:36:45,794 [INFO] <data parallel> Time ellapsed: 7.671835899353027s
2020-12-24 16:36:45,795 [INFO] <data parallel> Time ellapsed: 7.666041374206543s
2020-12-24 16:36:45,795 [INFO] <data parallel> Time ellapsed: 7.6677258014678955s
2020-12-24 16:36:45,795 [INFO] <data parallel> Time ellapsed: 7.665920734405518s
```

$$\sin^2\theta + \cos^2\theta=1$$