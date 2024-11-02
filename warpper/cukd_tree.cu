#include <torch/extension.h>
#include "warpper/warp.cuh"
#include "warpper/cukd_tree.h"


using namespace cukd;
using namespace cukd::common;

__global__
void d_fcp(float   *d_results,
           float3  *d_queries,
           int      numQueries,
           /*! the world bounding box computed by the builder */
           const cukd::box_t<float3> *d_bounds,
           float3  *d_nodes,
           int      numNodes,
           float    cutOffRadius)
{
  int tid = threadIdx.x+blockIdx.x*blockDim.x;
  if (tid >= numQueries) return;

  using point_t = float3;
  point_t queryPos = d_queries[tid];
  FcpSearchParams params;
  params.cutOffRadius = cutOffRadius;
  int closestID
    = cukd::cct::fcp
    (queryPos,*d_bounds,d_nodes,numNodes,params);
  
  d_results[tid]
    = (closestID < 0)
    ? INFINITY
    : distance(queryPos,d_nodes[closestID]);
}


// registeredExitHooks=false;

cukd::box_t<float3>* build_kdtree(const at::Tensor src_points){ // tris:[N,3]
    assert(src_points.sizes()[1] == 3);

    int numPoints = src_points.size(0);
    float3* d_points = (float3*)src_points.data_ptr<float>();
    cukd::box_t<float3> *d_bounds;
    cudaMallocManaged((void**)&d_bounds,sizeof(cukd::box_t<float3>));
    cukd::buildTree(d_points,numPoints,d_bounds);
    CUKD_CUDA_SYNC_CHECK();
    // if (!registeredExitHooks)
    // {
    //     registeredExitHooks = true;
    //     py::module_::import("atexit").attr("register")(py::module_::import("cuda_kdtree"));
    // }
    return d_bounds;

}



at::Tensor query(const at::Tensor query_points,const at::Tensor d_points,cukd::box_t<float3> *d_bounds){ // only knn=1
    float3* d_queries_f3 = (float3*)query_points.data_ptr<float>();
    float3* d_points_f3 = (float3*)d_points.data_ptr<float>();
    int numQueries = query_points.size(0);
    int numPoints = d_points.size(0);
    float cutOffRadius = std::numeric_limits<float>::infinity();
    float  *d_results;
    CUKD_CUDA_CALL(MallocManaged((void**)&d_results,numQueries*sizeof(*d_results)));

    int bs = 128;
    int nb = divRoundUp(numQueries,bs);
    
    d_fcp<<<nb,bs>>>(d_results,d_queries_f3,numQueries,d_bounds,d_points_f3,numPoints,cutOffRadius);
    cudaDeviceSynchronize();

    int device;
    // CHECK_CUDA(cudaGetDevice(&device));
    cudaGetDevice(&device);
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, device);
    at::Tensor result = torch::from_blob(d_results, { numQueries, }, options).clone();

    // if (!registeredExitHooks)
    // {
    //     registeredExitHooks = true;
    //     py::module_::import("atexit").attr("register")(py::module_::import("cuda_kdtree"));
    // }
    return result;
}

// at::Tensor build_kdtree(const at::Tensor src_points, const int num)
// at::Tensor query(const at::Tensor query_points, const int num, const int knn);