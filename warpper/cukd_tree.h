#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>

#include "warpper/warp.cuh"

using namespace cukd;
using namespace cukd::common;

cukd::box_t<float3>* build_kdtree(const at::Tensor src_points);
at::Tensor query(const at::Tensor query_points, const at::Tensor d_points,
                 cukd::box_t<float3>* d_bounds);
void free_cached_memory();