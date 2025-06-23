#ifndef FAST_GICP_CUDA_FIND_VOXEL_CORRESPONDENCES_CUH
#define FAST_GICP_CUDA_FIND_VOXEL_CORRESPONDENCES_CUH

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <thrust/pair.h>
#include "cuda_types.h"
#include <fast_gicp/cuda/gaussian_voxelmap.cuh>

namespace fast_gicp {
namespace cuda {

void find_voxel_correspondences(
  const fast_gicp::cuda::device_vector<Eigen::Vector3f>& src_points,
  const GaussianVoxelMap& voxelmap,
  const thrust::device_ptr<const Eigen::Isometry3f>& x_ptr,
  const fast_gicp::cuda::device_vector<Eigen::Vector3i>& offsets,
  fast_gicp::cuda::device_vector<thrust::pair<int, int>>& correspondences);

}  // namespace cuda
}  // namespace fast_gicp

#endif
