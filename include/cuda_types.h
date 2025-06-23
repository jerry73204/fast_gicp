#pragma once

/**
 * @file cuda_types.h
 * @brief Modern CUDA type system for fast_gicp Phase 2 modernization
 *
 * This header provides modernized type aliases that balance performance
 * with CUDA 12.x compatibility. Critical design decisions:
 *
 * - Device vectors: Use modern defaults (no explicit allocators)
 * - Host vectors: Preserve Eigen alignment for SIMD performance
 * - Type safety: Static assertions for critical properties
 */

#include <cuda_runtime.h>
#include <thrust/version.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/system/cuda/execution_policy.h>

// Eigen includes with alignment support
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace fast_gicp {
namespace cuda {

// Version detection for compatibility
#define FAST_GICP_CUDA_TYPES_VERSION 2
#define FAST_GICP_THRUST_VERSION THRUST_VERSION

/**
 * @brief Modern device vector aliases without explicit allocators
 *
 * CUDA 12.x thrust::device_vector handles allocation optimally without
 * explicit allocator specifications.
 */
template <typename T>
using device_vector = thrust::device_vector<T>;

/**
 * @brief Host vector aliases with selective alignment preservation
 *
 * For performance-critical Eigen types, we preserve aligned allocation
 * to ensure SIMD vectorization. For other types, use defaults.
 */
template <typename T>
using host_vector = thrust::host_vector<T>;

template <typename T>
using aligned_host_vector = thrust::host_vector<T, Eigen::aligned_allocator<T>>;

/**
 * @brief Specialized type aliases for fast_gicp algorithms
 *
 * These types are used throughout the CUDA implementation and are
 * optimized for the specific requirements of GICP algorithms.
 */

// === Device-side types (GPU memory) ===
using Points3f = device_vector<Eigen::Vector3f>;
using Points4f = device_vector<Eigen::Vector4f>;
using Matrices3f = device_vector<Eigen::Matrix3f>;
using Matrices4f = device_vector<Eigen::Matrix4f>;
using Indices = device_vector<int>;
using FloatVector = device_vector<float>;
using Correspondences = device_vector<thrust::pair<int, int>>;
using VoxelCoordinates = device_vector<Eigen::Vector3i>;

// === Host-side types (CPU memory with alignment) ===
// These preserve alignment for optimal SIMD performance
using HostPoints3f = aligned_host_vector<Eigen::Vector3f>;
using HostPoints4f = aligned_host_vector<Eigen::Vector4f>;
using HostMatrices3f = aligned_host_vector<Eigen::Matrix3f>;
using HostMatrices4f = aligned_host_vector<Eigen::Matrix4f>;
using HostIndices = host_vector<int>;  // integers don't need special alignment
using HostFloatVector = host_vector<float>;
using HostVoxelCoordinates = host_vector<Eigen::Vector3i>;

/**
 * @brief Utility functions for safe pointer access
 */
template <typename T>
inline T* raw_ptr(device_vector<T>& vec) {
  return thrust::raw_pointer_cast(vec.data());
}

template <typename T>
inline const T* raw_ptr(const device_vector<T>& vec) {
  return thrust::raw_pointer_cast(vec.data());
}

template <typename T>
inline T* raw_ptr(host_vector<T>& vec) {
  return thrust::raw_pointer_cast(vec.data());
}

template <typename T>
inline const T* raw_ptr(const host_vector<T>& vec) {
  return thrust::raw_pointer_cast(vec.data());
}

template <typename T>
inline T* raw_ptr(aligned_host_vector<T>& vec) {
  return thrust::raw_pointer_cast(vec.data());
}

template <typename T>
inline const T* raw_ptr(const aligned_host_vector<T>& vec) {
  return thrust::raw_pointer_cast(vec.data());
}

/**
 * @brief Type traits for alignment checking
 */
template <typename T>
struct is_eigen_fixed_size : std::false_type {};

template <>
struct is_eigen_fixed_size<Eigen::Vector3f> : std::true_type {};

template <>
struct is_eigen_fixed_size<Eigen::Vector4f> : std::true_type {};

template <>
struct is_eigen_fixed_size<Eigen::Matrix3f> : std::true_type {};

template <>
struct is_eigen_fixed_size<Eigen::Matrix4f> : std::true_type {};

/**
 * @brief Execution policy helpers for modern CUDA
 */
class ExecutionPolicy {
public:
  static auto device() { return thrust::cuda::par; }

  static auto device_on(cudaStream_t stream) { return thrust::cuda::par.on(stream); }

  static auto host() { return thrust::host; }
};

/**
 * @brief Memory transfer utilities with alignment preservation
 */
template <typename T>
void copy_to_device(const aligned_host_vector<T>& host_vec, device_vector<T>& device_vec) {
  device_vec.resize(host_vec.size());
  thrust::copy(host_vec.begin(), host_vec.end(), device_vec.begin());
}

template <typename T>
void copy_to_host(const device_vector<T>& device_vec, aligned_host_vector<T>& host_vec) {
  host_vec.resize(device_vec.size());
  thrust::copy(device_vec.begin(), device_vec.end(), host_vec.begin());
}

/**
 * @brief Static assertions to verify type properties
 */
namespace detail {
// Verify that our device types work correctly
static_assert(std::is_same_v<Points3f::value_type, Eigen::Vector3f>, "Points3f must contain Eigen::Vector3f");
static_assert(std::is_same_v<Matrices3f::value_type, Eigen::Matrix3f>, "Matrices3f must contain Eigen::Matrix3f");

// Verify alignment preservation for host types
static_assert(std::is_same_v<HostPoints3f::allocator_type, Eigen::aligned_allocator<Eigen::Vector3f>>, "HostPoints3f must use aligned allocator");
static_assert(std::is_same_v<HostMatrices3f::allocator_type, Eigen::aligned_allocator<Eigen::Matrix3f>>, "HostMatrices3f must use aligned allocator");

// Verify that device vectors use default allocators
static_assert(std::is_same_v<Points3f::allocator_type, thrust::device_allocator<Eigen::Vector3f>>, "Points3f should use default device allocator");
}  // namespace detail

}  // namespace cuda
}  // namespace fast_gicp

// Global type aliases for convenience (in fast_gicp namespace)
namespace fast_gicp {
// Commonly used device types
using CudaPoints3f = cuda::Points3f;
using CudaMatrices3f = cuda::Matrices3f;
using CudaIndices = cuda::Indices;
using CudaCorrespondences = cuda::Correspondences;

// Commonly used host types
using HostPoints3f = cuda::HostPoints3f;
using HostMatrices3f = cuda::HostMatrices3f;
}  // namespace fast_gicp

// Convenience macros for migration
#define FAST_GICP_DEVICE_VECTOR(T) ::fast_gicp::cuda::device_vector<T>
#define FAST_GICP_HOST_VECTOR(T) ::fast_gicp::cuda::host_vector<T>
#define FAST_GICP_ALIGNED_HOST_VECTOR(T) ::fast_gicp::cuda::aligned_host_vector<T>
#define FAST_GICP_RAW_PTR(vec) ::fast_gicp::cuda::raw_ptr(vec)
