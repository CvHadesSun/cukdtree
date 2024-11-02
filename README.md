# cukdtree
warp the cudaKdtree into python api, original repo: https://github.com/ingowald/cudaKDTree


## install 

need to install torch first in your python env.

```shell
pip install https://github.com/CvHadesSun/cukdtree.git 

```

or 
```shell
git clone https://github.com/CvHadesSun/cukdtree.git
cd cukdtree && python setup.py install 
```

## usage

example
```python
import torch # must import first
import cuda_kdtree
import time

device = torch.device("cuda")

points_ref = torch.randn(size=(100_000_000, 3), dtype=torch.float32, device=device, requires_grad=True) * 1e3
points_query = torch.randn(size=(100_1000_000, 3), dtype=torch.float32, device=device, requires_grad=True) * 1e3

t0=time.time()
cu_kdtree = cuda_kdtree.build_kdtree(points_ref)
t1 = time.time()
print('Time to build the KD-Tree on the GPU:', t1-t0)

t2 = time.time()
dists = cuda_kdtree.query(points_query, points_ref,cu_kdtree)
t3 = time.time()
print('Time to search the 5 nearest neighbors on the GPU:', t3-t2)

```

## change-log
- [x] warp kdtree, query knn=1 into python api 
- [] knn api
- [] other input.
- [] class warp.


## reference
- https://github.com/ingowald/cudaKDTree