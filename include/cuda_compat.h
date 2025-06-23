#pragma once

/**
 * @file cuda_compat.h
 * @brief CUDA 12.x compatibility layer for fast_gicp
 *
 * This header provides compatibility shims and modern type aliases
 * for upgrading fast_gicp to work with CUDA 12.x and modern Thrust.
 */

#include <cuda_runtime.h>
#include <thrust/version.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/system/cuda/execution_policy.h>

// Eigen includes
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace fast_gicp {
namespace cuda {

// CUDA version detection
#define FAST_GICP_CUDA_VERSION CUDA_VERSION
#define FAST_GICP_THRUST_VERSION THRUST_VERSION

// Require CUDA 12.x for this modernization
#if CUDA_VERSION < 12000
#error "This modernized version requires CUDA 12.0 or later"
#endif

#define FAST_GICP_CUDA_12_PLUS

/**
 * @brief Modern type aliases without explicit allocators
 *
 * These replace the old thrust::device_vector<T, thrust::device_allocator<T>>
 * patterns with modern defaults.
 */
template <typename T>
using device_vector = thrust::device_vector<T>;

template <typename T>
using host_vector = thrust::host_vector<T>;

// Specific types used throughout fast_gicp
using Points3f = device_vector<Eigen::Vector3f>;
using Points4f = device_vector<Eigen::Vector4f>;
using Matrices3f = device_vector<Eigen::Matrix3f>;
using Matrices4f = device_vector<Eigen::Matrix4f>;
using Indices = device_vector<int>;
using FloatVector = device_vector<float>;
using Correspondences = device_vector<thrust::pair<int, int>>;
using VoxelCoordinates = device_vector<Eigen::Vector3i>;

// Host vector aliases for consistency
using HostPoints3f = host_vector<Eigen::Vector3f>;
using HostMatrices3f = host_vector<Eigen::Matrix3f>;
using HostIndices = host_vector<int>;

/**
 * @brief Helper functions for raw pointer access
 */
template <typename T>
inline T* raw_ptr(device_vector<T>& vec) {
  return thrust::raw_pointer_cast(vec.data());
}

template <typename T>
inline const T* raw_ptr(const device_vector<T>& vec) {
  return thrust::raw_pointer_cast(vec.data());
}

/**
 * @brief CUDA error checking macros
 */
#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                                            \
  do {                                                                                              \
    cudaError_t error = call;                                                                       \
    if (error != cudaSuccess) {                                                                     \
      fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
      exit(1);                                                                                      \
    }                                                                                               \
  } while (0)
#endif

#ifndef CUDA_CHECK_LAST_ERROR
#define CUDA_CHECK_LAST_ERROR() CUDA_CHECK(cudaGetLastError())
#endif

/**
 * @brief Forward compatibility for execution policies
 *
 * Provides a consistent interface for both CUDA 11.x and 12.x
 */
class ExecutionPolicy {
public:
  static auto device() { return thrust::cuda::par; }

  static auto device_on(cudaStream_t stream) { return thrust::cuda::par.on(stream); }

  static auto host() { return thrust::host; }
};

/**
 * @brief Stream wrapper for RAII management
 *
 * This will be expanded in Phase 3, but provides basic compatibility
 */
class StreamWrapper {
private:
  cudaStream_t stream_;
  bool owns_stream_;

public:
  explicit StreamWrapper(cudaStream_t stream = nullptr) : stream_(stream), owns_stream_(stream == nullptr) {
    if (owns_stream_) {
      CUDA_CHECK(cudaStreamCreate(&stream_));
    }
  }

  ~StreamWrapper() {
    if (owns_stream_ && stream_) {
      cudaStreamDestroy(stream_);
    }
  }

  // Non-copyable but movable
  StreamWrapper(const StreamWrapper&) = delete;
  StreamWrapper& operator=(const StreamWrapper&) = delete;

  StreamWrapper(StreamWrapper&& other) noexcept : stream_(other.stream_), owns_stream_(other.owns_stream_) {
    other.stream_ = nullptr;
    other.owns_stream_ = false;
  }

  cudaStream_t get() const { return stream_; }

  auto policy() const { return ExecutionPolicy::device_on(stream_); }

  void synchronize() { CUDA_CHECK(cudaStreamSynchronize(stream_)); }
};

/**
 * @brief Event wrapper for RAII management
 */
class EventWrapper {
private:
  cudaEvent_t event_;

public:
  EventWrapper() { CUDA_CHECK(cudaEventCreate(&event_)); }

  ~EventWrapper() {
    if (event_) {
      cudaEventDestroy(event_);
    }
  }

  // Non-copyable but movable
  EventWrapper(const EventWrapper&) = delete;
  EventWrapper& operator=(const EventWrapper&) = delete;

  EventWrapper(EventWrapper&& other) noexcept : event_(other.event_) { other.event_ = nullptr; }

  cudaEvent_t get() const { return event_; }

  void record(cudaStream_t stream = nullptr) { CUDA_CHECK(cudaEventRecord(event_, stream)); }

  void synchronize() { CUDA_CHECK(cudaEventSynchronize(event_)); }

  float elapsed_time(const EventWrapper& start) const {
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start.event_, event_));
    return ms;
  }
};

/**
 * @brief Compatibility shims for deprecated APIs
 *
 * These will be removed in later phases but provide transition support
 */
namespace compat {
// Placeholder for thrust::system::cuda::detail compatibility
// Will be implemented in Phase 3 if needed
}

}  // namespace cuda
}  // namespace fast_gicp

// Global compatibility macros
#define FAST_GICP_DEVICE_VECTOR(T) ::fast_gicp::cuda::device_vector<T>
#define FAST_GICP_HOST_VECTOR(T) ::fast_gicp::cuda::host_vector<T>
#define FAST_GICP_RAW_PTR(vec) ::fast_gicp::cuda::raw_ptr(vec)
