/**
 * @file brute_force_knn.cu
 * @brief Pure Thrust/CUB implementation of k-nearest neighbor search
 *
 * Phase 4 of CUDA 12.x modernization - Algorithm Modernization
 * This implementation replaces the nvbio dependency with a pure Thrust/CUB solution
 * that requires no external dependencies beyond CUDA Core Compute Libraries.
 */

#include <fast_gicp/cuda/brute_force_knn.cuh>
#include <fast_gicp/cuda/cuda_context.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>

#include <cub/device/device_radix_sort.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <cub/device/device_select.cuh>

#include <cuda_runtime.h>
#include <limits>

namespace fast_gicp {
namespace cuda {

namespace {

/**
 * @brief Compute squared distances between a source point and all target points
 */
struct compute_distances_functor {
  const Eigen::Vector3f* target_points;
  const Eigen::Vector3f source_point;
  int num_targets;

  compute_distances_functor(const Eigen::Vector3f* targets, const Eigen::Vector3f& source, int n) : target_points(targets), source_point(source), num_targets(n) {}

  __device__ __forceinline__ thrust::pair<float, int> operator()(int idx) const {
    if (idx >= num_targets) {
      return thrust::make_pair(std::numeric_limits<float>::max(), -1);
    }

    Eigen::Vector3f diff = target_points[idx] - source_point;
    float sq_dist = diff.squaredNorm();
    return thrust::make_pair(sq_dist, idx);
  }
};

/**
 * @brief Kernel for brute-force k-nearest neighbor search using block-level operations
 *
 * Each block processes one source point, computing distances to all target points
 * and maintaining the k-nearest neighbors using CUB's block-level sort.
 */
template <int BLOCK_SIZE, int K>
__global__ void brute_force_knn_kernel(
  const Eigen::Vector3f* __restrict__ source_points,
  const Eigen::Vector3f* __restrict__ target_points,
  int num_source,
  int num_target,
  thrust::pair<float, int>* __restrict__ k_neighbors) {
  // Shared memory for block-level operations
  using BlockRadixSort = cub::BlockRadixSort<float, BLOCK_SIZE, K, int>;
  __shared__ typename BlockRadixSort::TempStorage temp_storage;

  // Each block handles one source point
  int source_idx = blockIdx.x;
  if (source_idx >= num_source) return;

  const Eigen::Vector3f& source_point = source_points[source_idx];

  // Thread-local arrays for k-nearest neighbors
  float thread_distances[K];
  int thread_indices[K];

// Initialize with maximum distances
#pragma unroll
  for (int i = 0; i < K; ++i) {
    thread_distances[i] = std::numeric_limits<float>::max();
    thread_indices[i] = -1;
  }

  // Each thread processes a subset of target points
  for (int target_idx = threadIdx.x; target_idx < num_target; target_idx += BLOCK_SIZE) {
    // Compute distance
    Eigen::Vector3f diff = target_points[target_idx] - source_point;
    float sq_dist = diff.squaredNorm();

    // Check if this distance should be in top-k
    if (sq_dist < thread_distances[K - 1]) {
      // Insert into sorted position
      thread_distances[K - 1] = sq_dist;
      thread_indices[K - 1] = target_idx;

      // Sort to maintain order
      for (int i = K - 1; i > 0 && thread_distances[i] < thread_distances[i - 1]; --i) {
        // Swap
        float tmp_dist = thread_distances[i];
        int tmp_idx = thread_indices[i];
        thread_distances[i] = thread_distances[i - 1];
        thread_indices[i] = thread_indices[i - 1];
        thread_distances[i - 1] = tmp_dist;
        thread_indices[i - 1] = tmp_idx;
      }
    }
  }

  // Block-level reduction to find global k-nearest
  __syncthreads();

  // Perform block-wide sort of all thread-local k-nearest candidates
  BlockRadixSort(temp_storage).Sort(thread_distances, thread_indices);
  __syncthreads();

  // First K threads write the final results
  if (threadIdx.x < K) {
    k_neighbors[source_idx * K + threadIdx.x] = thrust::make_pair(thread_distances[threadIdx.x], thread_indices[threadIdx.x]);
  }
}

/**
 * @brief Alternative implementation using Thrust for smaller datasets
 *
 * This approach computes all pairwise distances and uses thrust::sort
 * to find k-nearest neighbors. More suitable for smaller point clouds.
 */
void brute_force_knn_thrust_impl(
  const thrust::device_vector<Eigen::Vector3f>& source,
  const thrust::device_vector<Eigen::Vector3f>& target,
  int k,
  thrust::device_vector<thrust::pair<float, int>>& k_neighbors,
  CudaExecutionContext& ctx,
  bool do_sort) {
  int num_source = source.size();
  int num_target = target.size();

  // Ensure output buffer is properly sized
  k_neighbors.resize(num_source * k);

  // Process each source point
  for (int src_idx = 0; src_idx < num_source; ++src_idx) {
    // Temporary buffers for this source point
    thrust::device_vector<thrust::pair<float, int>> distances(num_target);

    // Get raw pointer to source point
    const Eigen::Vector3f* src_ptr = thrust::raw_pointer_cast(source.data() + src_idx);
    const Eigen::Vector3f* tgt_ptr = thrust::raw_pointer_cast(target.data());

    // Compute distances to all target points
    thrust::transform(ctx.policy(), thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(num_target), distances.begin(), [src_ptr, tgt_ptr] __device__(int idx) {
      Eigen::Vector3f diff = tgt_ptr[idx] - *src_ptr;
      return thrust::make_pair(diff.squaredNorm(), idx);
    });

    // Sort by distance
    thrust::sort(ctx.policy(), distances.begin(), distances.end(), [] __device__(const thrust::pair<float, int>& a, const thrust::pair<float, int>& b) {
      return a.first < b.first;
    });

    // Copy k-nearest to output
    thrust::copy_n(ctx.policy(), distances.begin(), k, k_neighbors.begin() + src_idx * k);
  }
}

/**
 * @brief Fast implementation using CUB's device-wide operations
 *
 * This is the most efficient implementation for large datasets,
 * using CUB's optimized device-wide algorithms.
 */
void brute_force_knn_cub_impl(
  const thrust::device_vector<Eigen::Vector3f>& source,
  const thrust::device_vector<Eigen::Vector3f>& target,
  int k,
  thrust::device_vector<thrust::pair<float, int>>& k_neighbors,
  CudaExecutionContext& ctx,
  bool do_sort) {
  int num_source = source.size();
  int num_target = target.size();

  // Ensure output buffer is properly sized
  k_neighbors.resize(num_source * k);

  // Configure kernel launch parameters
  const int BLOCK_SIZE = 256;
  const int MAX_K = 32;  // Maximum k value we support efficiently

  if (k > MAX_K) {
    // Fall back to thrust implementation for large k
    brute_force_knn_thrust_impl(source, target, k, k_neighbors, ctx, do_sort);
    return;
  }

  // Launch kernel based on k value
  dim3 block(BLOCK_SIZE);
  dim3 grid(num_source);

  const Eigen::Vector3f* src_ptr = thrust::raw_pointer_cast(source.data());
  const Eigen::Vector3f* tgt_ptr = thrust::raw_pointer_cast(target.data());
  thrust::pair<float, int>* out_ptr = thrust::raw_pointer_cast(k_neighbors.data());

  // Launch appropriate kernel based on k
  switch (k) {
    case 1:
      brute_force_knn_kernel<BLOCK_SIZE, 1><<<grid, block, 0, ctx.stream()>>>(src_ptr, tgt_ptr, num_source, num_target, out_ptr);
      break;
    case 5:
      brute_force_knn_kernel<BLOCK_SIZE, 5><<<grid, block, 0, ctx.stream()>>>(src_ptr, tgt_ptr, num_source, num_target, out_ptr);
      break;
    case 10:
      brute_force_knn_kernel<BLOCK_SIZE, 10><<<grid, block, 0, ctx.stream()>>>(src_ptr, tgt_ptr, num_source, num_target, out_ptr);
      break;
    case 20:
      brute_force_knn_kernel<BLOCK_SIZE, 20><<<grid, block, 0, ctx.stream()>>>(src_ptr, tgt_ptr, num_source, num_target, out_ptr);
      break;
    default:
      // For other k values, use thrust implementation
      brute_force_knn_thrust_impl(source, target, k, k_neighbors, ctx, do_sort);
      return;
  }

  // Check for kernel errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error("KNN kernel launch failed: " + std::string(cudaGetErrorString(err)));
  }
}

}  // anonymous namespace

/**
 * @brief Public interface for brute-force k-nearest neighbor search
 *
 * This implementation uses pure Thrust/CUB algorithms, requiring no external
 * dependencies beyond the CUDA Core Compute Libraries.
 */
void brute_force_knn_search(
  const thrust::device_vector<Eigen::Vector3f>& source,
  const thrust::device_vector<Eigen::Vector3f>& target,
  int k,
  thrust::device_vector<thrust::pair<float, int>>& k_neighbors,
  bool do_sort) {
  // Validate inputs
  if (source.empty() || target.empty() || k <= 0) {
    k_neighbors.clear();
    return;
  }

  if (k > static_cast<int>(target.size())) {
    k = target.size();
  }

  // Create execution context for this operation
  CudaExecutionContext ctx("knn_search");

  // Choose implementation based on problem size
  int num_source = source.size();
  int num_target = target.size();

  // For small problems, use thrust implementation
  if (num_source * num_target < 100000 || k > 32) {
    brute_force_knn_thrust_impl(source, target, k, k_neighbors, ctx, do_sort);
  } else {
    // For larger problems, use optimized CUB implementation
    brute_force_knn_cub_impl(source, target, k, k_neighbors, ctx, do_sort);
  }

  // Ensure all operations complete
  ctx.synchronize();
}

}  // namespace cuda
}  // namespace fast_gicp
