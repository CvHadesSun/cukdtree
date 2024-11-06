#include <torch/extension.h>
#include "warpper/warp.cuh"
#include "warpper/cukd_tree.h"
#include <tuple>

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

__global__
void d_fcp(float   *d_results,
          int      *d_indices,
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

    d_indices[tid] = closestID;
  
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



at::Tensor query_dist(const at::Tensor query_points,const at::Tensor d_points,cukd::box_t<float3> *d_bounds){ // only knn=1
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

std::tuple<at::Tensor, at::Tensor> query(const at::Tensor query_points,const at::Tensor d_points,cukd::box_t<float3> *d_bounds){ // only knn=1
  float3* d_queries_f3 = (float3*)query_points.data_ptr<float>();
  float3* d_points_f3 = (float3*)d_points.data_ptr<float>();
  int numQueries = query_points.size(0);
  int numPoints = d_points.size(0);
  float cutOffRadius = std::numeric_limits<float>::infinity();
  float  *d_results;
  int *d_indices;
  CUKD_CUDA_CALL(MallocManaged((void**)&d_results,numQueries*sizeof(*d_results)));
  CUKD_CUDA_CALL(MallocManaged((void**)&d_indices,numQueries*sizeof(*d_indices)));
  int bs = 128;
  int nb = divRoundUp(numQueries,bs);
  
  d_fcp<<<nb,bs>>>(d_results,d_indices,d_queries_f3,numQueries,d_bounds,d_points_f3,numPoints,cutOffRadius);
  cudaDeviceSynchronize();

  int device;
  // CHECK_CUDA(cudaGetDevice(&device));
  cudaGetDevice(&device);
  auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, device);
  auto options_int = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, device);
  at::Tensor result = torch::from_blob(d_results, { numQueries, }, options).clone();
  at::Tensor indices = torch::from_blob(d_indices, { numQueries, }, options_int).clone();

  // if (!registeredExitHooks)
  // {
  //     registeredExitHooks = true;
  //     py::module_::import("atexit").attr("register")(py::module_::import("cuda_kdtree"));
  // }
  cudaFree(d_results);
  cudaFree(d_indices);
  return std::make_tuple(result, indices);
}


cukd::box_t<float4>* build_kdtree_with_index(const at::Tensor src_points){

  float4* d_points = (float4*)src_points.data_ptr<float>();
  int numPoints = src_points.size(0);
  cukd::box_t<float4> *d_bounds;
  cudaMallocManaged((void**)&d_bounds,sizeof(cukd::box_t<float3>));
  using data_traits=cukd::default_data_traits<float4>;
  cukd::buildTree<float4,data_traits>(d_points,numPoints,d_bounds);
  CUKD_CUDA_SYNC_CHECK();
  return d_bounds;

}

// 


__global__ void copyPoints(PointPlusPayload *d_points,
  float3 *d_inputs,
  int* d_indices,
  int numPoints)
{
int tid = threadIdx.x+blockIdx.x*blockDim.x;
if (tid >= numPoints) return;
d_points[tid].position = d_inputs[tid];
d_points[tid].payload = d_indices[tid];
}

__global__ void copyPointsInverse(PointPlusPayload *d_points,
  float3 *d_inputs,
  int* d_indices,
  int numPoints)
{
int tid = threadIdx.x+blockIdx.x*blockDim.x;
if (tid >= numPoints) return;
d_inputs[tid] = d_points[tid].position;
d_indices[tid] = d_points[tid].payload;
}


std::tuple<cukd::box_t<float3>*,PointPlusPayload*> build_kdtree_with_index(const at::Tensor src_points,
  const at::Tensor src_index){
    int numPointPlusPayloads = src_points.size(0);
    assert(src_points.sizes()[0] == src_index.sizes()[0]);
    float3* d_inputs = (float3*)src_points.data_ptr<float>();
    int* d_indices = (int*)src_index.data_ptr<int>();

    cukd::box_t<float3> *d_bounds;
    CUKD_CUDA_CALL(MallocManaged((void **)&d_bounds,sizeof(*d_bounds)));

    PointPlusPayload *data = 0;
    CUKD_CUDA_CALL(MallocManaged((void **)&data,numPointPlusPayloads*sizeof(PointPlusPayload)));
    copyPoints<<<divRoundUp(numPointPlusPayloads,128),128>>>(data,d_inputs,d_indices,numPointPlusPayloads);
    cukd::buildTree<PointPlusPayload,PointPlusPayload_traits>(data,numPointPlusPayloads,d_bounds);

    return std::make_tuple(d_bounds,data);
  }

  // __global__
  // void callFCP(PointPlusPayload *data, int numData,
  //              cukd::box_t<float3> *d_worldBounds)
  // {
  //   int tid = threadIdx.x+blockIdx.x*blockIdx.x;
  //   if (tid >= numData) return;

  //   int result = cukd::stackBased::fcp<PointPlusPayload,PointPlusPayload_traits>
  //     (data[tid].position,*d_worldBounds,data,numData);
  // }

  std::tuple<at::Tensor, at::Tensor> get_data(PointPlusPayload *data,int num_data){

    float3* d_points;
    int* d_indices;
    CUKD_CUDA_CALL(MallocManaged((void **)&d_points,num_data*sizeof(*d_points)));
    CUKD_CUDA_CALL(MallocManaged((void **)&d_indices,num_data*sizeof(*d_indices)));
    copyPointsInverse<<<divRoundUp(num_data,128),128>>>(data,d_points,d_indices,num_data);
    CUKD_CUDA_SYNC_CHECK();

    int device;
    // CHECK_CUDA(cudaGetDevice(&device));
    cudaGetDevice(&device);
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, device);
    auto options_int = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, device);
    at::Tensor result = torch::from_blob(d_points, {3*num_data, }, options).clone();
    at::Tensor indices = torch::from_blob(d_indices, { num_data, }, options_int).clone();

    cudaFree(d_points);
    cudaFree(d_indices);

    return std::make_tuple(result, indices);
  }

  std::tuple<at::Tensor, at::Tensor,at::Tensor> query_from_kdtree(
    const at::Tensor query_points, PointPlusPayload *src_data,int num_data,cukd::box_t<float3> *d_bounds){
    float3* d_queries_f3 = (float3*)query_points.data_ptr<float>();
    int numQueries = query_points.size(0);
    int numPoints = num_data;

    //
    float3* d_points;
    int* d_indices;
    CUKD_CUDA_CALL(MallocManaged((void **)&d_points,num_data*sizeof(*d_points)));
    CUKD_CUDA_CALL(MallocManaged((void **)&d_indices,num_data*sizeof(*d_indices)));
    copyPointsInverse<<<divRoundUp(num_data,128),128>>>(src_data,d_points,d_indices,numPoints);

    //
    float cutOffRadius = std::numeric_limits<float>::infinity();
    float  *d_results;
    int* d_indices_knn;
    CUKD_CUDA_CALL(MallocManaged((void**)&d_results,numQueries*sizeof(*d_results)));
    CUKD_CUDA_CALL(MallocManaged((void **)&d_indices_knn,numQueries*sizeof(*d_indices_knn)));
    int bs = 128;
    int nb = divRoundUp(numQueries,bs);
    
    d_fcp<<<nb,bs>>>(d_results,d_indices_knn,d_queries_f3,numQueries,d_bounds,d_points,numPoints,cutOffRadius);
    cudaDeviceSynchronize();


    int device;
    cudaGetDevice(&device);
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, device);
    auto options_int = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, device);
    at::Tensor result = torch::from_blob(d_results, { numQueries, }, options).clone();
    at::Tensor indices = torch::from_blob(d_indices, { numPoints, }, options_int).clone();
    at::Tensor indices_knn = torch::from_blob(d_indices_knn, { numQueries, }, options_int).clone();

    cudaFree(d_results);
    cudaFree(d_indices);
    cudaFree(d_points);
    cudaFree(d_indices_knn);
    return std::make_tuple(result, indices_knn,indices);

    }