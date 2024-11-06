import torch
import cuda_kdtree
from scipy.spatial import KDTree 
import numpy as np
import time

def test_gpu(input_pts,query_pts):
    num_src = input_pts.size(0)
    # num_query = query_pts.size(0)
    device = input_pts.device
    points_ind = torch.arange(num_src, dtype=torch.int32, device=device)

    box,data_ptr = cuda_kdtree.build_kdtree_with_indices(input_pts,points_ind)
    dists,inds,ori_inds = cuda_kdtree.query_from_kdtree(query_pts, data_ptr,num_src,box)
    final_inds = ori_inds[inds]
    torch.cuda.empty_cache()

def test_cpu(input_pts,query_pts):
    kdtree = KDTree(input_pts)
    dists_ref, inds_ref = kdtree.query(query_pts, k=1)




ni=10_000_000
nqs = [10_000,100_000,100_000_000]


device = torch.device("cuda")
for nq in nqs:

    points_ref = torch.randn(size=(ni, 3), dtype=torch.float32, device=device, requires_grad=False) * 1e3
    points_query = torch.randn(size=(nq, 3), dtype=torch.float32, device=device, requires_grad=False) * 1e3

    points_ref_cpu= points_ref.cpu().numpy()
    points_query_cpu= points_query.cpu().numpy()
    t0 = time.time()

    for i in range(10):
        # test_gpu(points_ref,points_query)
        test_cpu(points_ref_cpu,points_query_cpu)

    t1 = time.time()

    print(f"{nq} time: {(t1-t0)/10}")


