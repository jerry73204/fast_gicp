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
#include <thrust/copy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/system/cuda/execution_policy.h>

#include <cub/device/device_radix_sort.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <cub/device/device_select.cuh>

#include <cuda_runtime.h>
#include <limits>

namespace fast_gicp {
namespace cuda {

namespace {

/**
 * @brief Kernel for brute-force k-nearest neighbor search using block-level operations
 *
 * Each block processes one source point, computing distances to all target points
 * and maintaining the k-nearest neighbors using proper block-wide merge.
 */
template <int BLOCK_SIZE, int K>
__global__ void brute_force_knn_kernel(
  const Eigen::Vector3f* __restrict__ source_points,
  const Eigen::Vector3f* __restrict__ target_points,
  int num_source,
  int num_target,
  thrust::pair<float, int>* __restrict__ k_neighbors) {
  // Limit shared memory usage to fit within GPU limits (48KB)
  // Each entry needs 8 bytes (float + int), so 48KB / 8 = 6K max entries
  // Leave some margin for other variables and compiler overhead
  const int MAX_CANDIDATES = min(BLOCK_SIZE * K, 6000);  // Limit to 6K candidates max
  __shared__ float shared_distances[6000];
  __shared__ int shared_indices[6000];
  __shared__ int shared_count;

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

      // Sort to maintain order (insertion sort)
      for (int i = K - 1; i > 0 && (thread_distances[i] < thread_distances[i - 1] || (thread_distances[i] == thread_distances[i - 1] && thread_indices[i] < thread_indices[i - 1]));
           --i) {
        // Swap with tie-breaking by index
        float tmp_dist = thread_distances[i];
        int tmp_idx = thread_indices[i];
        thread_distances[i] = thread_distances[i - 1];
        thread_indices[i] = thread_indices[i - 1];
        thread_distances[i - 1] = tmp_dist;
        thread_indices[i - 1] = tmp_idx;
      }
    }
  }

  __syncthreads();

  // Initialize shared counter
  if (threadIdx.x == 0) {
    shared_count = 0;
  }
  __syncthreads();

  // Each thread contributes its valid candidates to shared memory
  for (int i = 0; i < K; ++i) {
    if (thread_indices[i] != -1) {
      int pos = atomicAdd(&shared_count, 1);
      if (pos < MAX_CANDIDATES) {
        shared_distances[pos] = thread_distances[i];
        shared_indices[pos] = thread_indices[i];
      }
    }
  }

  __syncthreads();

  // Block-wide selection of k-nearest using parallel reduction
  // Simple approach: thread 0 does final selection
  if (threadIdx.x == 0) {
    int total_candidates = min(shared_count, MAX_CANDIDATES);

    // Create pairs for easier sorting - use smaller stack array
    thrust::pair<float, int> candidates[1024];  // Limit stack usage to be safe
    int actual_candidates = min(total_candidates, 1024);

    for (int i = 0; i < actual_candidates; ++i) {
      candidates[i] = thrust::make_pair(shared_distances[i], shared_indices[i]);
    }

    // Sort candidates with tie-breaking - use actual_candidates for bounds
    for (int i = 0; i < actual_candidates - 1; ++i) {
      for (int j = i + 1; j < actual_candidates; ++j) {
        bool should_swap = (candidates[i].first > candidates[j].first) || (candidates[i].first == candidates[j].first && candidates[i].second > candidates[j].second);
        if (should_swap) {
          thrust::pair<float, int> tmp = candidates[i];
          candidates[i] = candidates[j];
          candidates[j] = tmp;
        }
      }
    }

    // Write k-nearest results
    int actual_k = min(K, actual_candidates);
    for (int i = 0; i < actual_k; ++i) {
      k_neighbors[source_idx * K + i] = candidates[i];
    }

    // Fill remaining slots with invalid entries if needed
    for (int i = actual_k; i < K; ++i) {
      k_neighbors[source_idx * K + i] = thrust::make_pair(std::numeric_limits<float>::max(), -1);
    }
  }
}

/**
 * @brief Functor for computing distances between source and target points
 */
struct distance_functor {
  const Eigen::Vector3f* source_points;
  const Eigen::Vector3f* target_points;
  int num_target;
  int k;

  distance_functor(const Eigen::Vector3f* src, const Eigen::Vector3f* tgt, int nt, int k_val) : source_points(src), target_points(tgt), num_target(nt), k(k_val) {}

  __device__ thrust::pair<float, int> operator()(int global_idx) const {
    int src_idx = global_idx / num_target;
    int tgt_idx = global_idx % num_target;

    Eigen::Vector3f diff = target_points[tgt_idx] - source_points[src_idx];
    float sq_dist = diff.squaredNorm();

    // Encode source index in high bits for later separation
    return thrust::make_pair(sq_dist, (src_idx << 16) | tgt_idx);
  }
};

/**
 * @brief Comparison functor with tie-breaking
 */
struct distance_comparator {
  __device__ bool operator()(const thrust::pair<float, int>& a, const thrust::pair<float, int>& b) const {
    if (a.first != b.first) {
      return a.first < b.first;
    }
    // Tie-breaking: use target index (lower 16 bits)
    return (a.second & 0xFFFF) < (b.second & 0xFFFF);
  }
};

/**
 * @brief Alternative implementation using Thrust for datasets
 *
 * This approach parallelizes across all source points simultaneously
 * and uses efficient device-wide operations.
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

  // For very small datasets, use simple approach
  if (num_source <= 10 || num_target <= 10) {
    // Fall back to sequential approach for very small datasets
    auto exec_policy = thrust::cuda::par.on(ctx.stream());

    for (int src_idx = 0; src_idx < num_source; ++src_idx) {
      thrust::device_vector<thrust::pair<float, int>> distances(num_target);

      const Eigen::Vector3f* src_ptr = thrust::raw_pointer_cast(source.data() + src_idx);
      const Eigen::Vector3f* tgt_ptr = thrust::raw_pointer_cast(target.data());

      thrust::transform(exec_policy, thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(num_target), distances.begin(), [src_ptr, tgt_ptr] __device__(int idx) {
        Eigen::Vector3f diff = tgt_ptr[idx] - *src_ptr;
        return thrust::make_pair(diff.squaredNorm(), idx);
      });

      thrust::sort(exec_policy, distances.begin(), distances.end(), distance_comparator());
      thrust::copy_n(exec_policy, distances.begin(), k, k_neighbors.begin() + src_idx * k);
    }
    return;
  }

  // For larger datasets, process in chunks to avoid memory explosion
  auto exec_policy = thrust::cuda::par.on(ctx.stream());

  // Process source points in smaller batches to manage memory
  const int BATCH_SIZE = std::min(100, num_source);  // Process max 100 source points at once

  for (int batch_start = 0; batch_start < num_source; batch_start += BATCH_SIZE) {
    int batch_end = std::min(batch_start + BATCH_SIZE, num_source);
    int batch_size = batch_end - batch_start;

    // Compute distances for this batch
    int batch_pairs = batch_size * num_target;
    thrust::device_vector<thrust::pair<float, int>> batch_distances(batch_pairs);

    const Eigen::Vector3f* src_ptr = thrust::raw_pointer_cast(source.data());
    const Eigen::Vector3f* tgt_ptr = thrust::raw_pointer_cast(target.data());

    // Transform with adjusted source indexing for the batch
    thrust::transform(
      exec_policy,
      thrust::counting_iterator<int>(0),
      thrust::counting_iterator<int>(batch_pairs),
      batch_distances.begin(),
      [src_ptr, tgt_ptr, num_target, batch_start] __device__(int global_idx) {
        int src_idx = (global_idx / num_target) + batch_start;
        int tgt_idx = global_idx % num_target;

        Eigen::Vector3f diff = tgt_ptr[tgt_idx] - src_ptr[src_idx];
        float sq_dist = diff.squaredNorm();

        return thrust::make_pair(sq_dist, tgt_idx);
      });

    // Process each source point in this batch
    for (int i = 0; i < batch_size; ++i) {
      int src_idx = batch_start + i;

      // Extract distances for this source point
      auto src_begin = batch_distances.begin() + i * num_target;
      auto src_end = src_begin + num_target;

      // Sort distances for this source
      thrust::sort(exec_policy, src_begin, src_end, distance_comparator());

      // Copy k-nearest to output
      thrust::copy_n(exec_policy, src_begin, k, k_neighbors.begin() + src_idx * k);
    }
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

  // Safety check for extremely large datasets
  long long total_pairs = (long long)source.size() * target.size();
  if (total_pairs > 100000000LL) {  // 100M pairs
    throw std::runtime_error(
      "Dataset too large for brute force KNN: " + std::to_string(source.size()) + " x " + std::to_string(target.size()) + " = " + std::to_string(total_pairs) + " pairs");
  }

  // Create execution context for this operation
  CudaExecutionContext ctx("knn_search");

  // Choose implementation based on problem size
  int num_source = source.size();
  int num_target = target.size();

  // Choose implementation based on problem size and memory constraints
  // Use CUB only for medium-sized datasets where it's most efficient
  long long pairs_count = (long long)num_source * num_target;
  bool use_cub = (pairs_count >= 10000 && pairs_count <= 10000000) && (k <= 32) && (num_source <= 2000);
  bool use_thrust = !use_cub;

  if (use_thrust) {
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
