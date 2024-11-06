#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>

#include "warpper/warp.cuh"

using namespace cukd;
using namespace cukd::common;

struct PointPlusPayload {
  float3 position;
  int payload;
};

struct PointPlusPayload_traits : public cukd::default_data_traits<float3> {
  using point_t = float3;

  static inline __device__ __host__ float3
  get_point(const PointPlusPayload &data) {
    return data.position;
  }

  static inline __device__ __host__ float get_coord(
      const PointPlusPayload &data, int dim) {
    return cukd::get_coord(get_point(data), dim);
  }

  enum { has_explicit_dim = false };

  /*! !{ just defining this for completeness, get/set_dim should never
    get called for this type because we have set has_explicit_dim
    set to false. note traversal should ONLY ever call this
    function for data_t's that define has_explicit_dim to true */
  static inline __device__ int get_dim(const PointPlusPayload &) { return -1; }
};

cukd::box_t<float3> *build_kdtree(const at::Tensor src_points);
// cukd::box_t<float4>* build_kdtree_with_index(const at::Tensor src_points);
std::tuple<cukd::box_t<float3> *, PointPlusPayload *> build_kdtree_with_index(
    const at::Tensor src_points, const at::Tensor src_index);
std::tuple<at::Tensor, at::Tensor> query(const at::Tensor query_points,
                                         const at::Tensor d_points,
                                         cukd::box_t<float3> *d_bounds);

std::tuple<at::Tensor, at::Tensor, at::Tensor> query_from_kdtree(
    const at::Tensor query_points, PointPlusPayload *src_data, int num_data,
    cukd::box_t<float3> *d_bounds);

std::tuple<at::Tensor, at::Tensor> get_data(PointPlusPayload *data,
                                            int num_data);

// std::tuple<at::Tensor, at::Tensor> query
void free_cached_memory();