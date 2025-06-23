#pragma once

/**
 * @file cuda_context.h
 * @brief Modern CUDA stream and execution management for fast_gicp
 *
 * Phase 3 of CUDA 12.x modernization - Stream Management
 * This header provides RAII-based stream management and modern execution patterns
 * to replace deprecated stream/event APIs.
 */

#include <cuda_runtime.h>
#include <thrust/execution_policy.h>
#include <thrust/system/cuda/execution_policy.h>

#include <string>
#include <memory>
#include <stdexcept>

// Include our type system
#include "cuda_types.h"

namespace fast_gicp {
namespace cuda {

/**
 * @brief RAII wrapper for CUDA streams with modern execution policy support
 *
 * This class provides thread-safe stream management with automatic cleanup,
 * replacing the deprecated unique_stream patterns with modern CUDA 12.x APIs.
 */
class CudaExecutionContext {
private:
  cudaStream_t stream_;
  std::string name_;
  bool owns_stream_;

public:
  /**
   * @brief Create a new execution context with its own stream
   * @param name Optional name for debugging/profiling
   */
  explicit CudaExecutionContext(const std::string& name = "default") : name_(name), owns_stream_(true) {
    cudaError_t err = cudaStreamCreate(&stream_);
    if (err != cudaSuccess) {
      throw std::runtime_error("Failed to create CUDA stream: " + std::string(cudaGetErrorString(err)));
    }
  }

  /**
   * @brief Create an execution context from an existing stream (non-owning)
   * @param stream Existing CUDA stream
   * @param name Optional name for debugging
   */
  CudaExecutionContext(cudaStream_t stream, const std::string& name = "external") : stream_(stream), name_(name), owns_stream_(false) {}

  // Disable copy construction/assignment
  CudaExecutionContext(const CudaExecutionContext&) = delete;
  CudaExecutionContext& operator=(const CudaExecutionContext&) = delete;

  // Enable move construction/assignment
  CudaExecutionContext(CudaExecutionContext&& other) noexcept : stream_(other.stream_), name_(std::move(other.name_)), owns_stream_(other.owns_stream_) {
    other.stream_ = nullptr;
    other.owns_stream_ = false;
  }

  CudaExecutionContext& operator=(CudaExecutionContext&& other) noexcept {
    if (this != &other) {
      cleanup();
      stream_ = other.stream_;
      name_ = std::move(other.name_);
      owns_stream_ = other.owns_stream_;
      other.stream_ = nullptr;
      other.owns_stream_ = false;
    }
    return *this;
  }

  ~CudaExecutionContext() { cleanup(); }

  /**
   * @brief Get Thrust execution policy for this stream
   * @return Thrust parallel execution policy on this stream
   */
  auto policy() const { return thrust::cuda::par.on(stream_); }

  /**
   * @brief Get the raw CUDA stream handle
   */
  cudaStream_t stream() const { return stream_; }

  /**
   * @brief Get the context name
   */
  const std::string& name() const { return name_; }

  /**
   * @brief Synchronize this stream
   */
  void synchronize() {
    cudaError_t err = cudaStreamSynchronize(stream_);
    if (err != cudaSuccess) {
      throw std::runtime_error("Stream synchronization failed: " + std::string(cudaGetErrorString(err)));
    }
  }

  /**
   * @brief Wait for an event on this stream
   * @param event CUDA event to wait for
   */
  void wait_event(cudaEvent_t event) {
    cudaError_t err = cudaStreamWaitEvent(stream_, event, 0);
    if (err != cudaSuccess) {
      throw std::runtime_error("Stream wait event failed: " + std::string(cudaGetErrorString(err)));
    }
  }

  /**
   * @brief Query if all operations on this stream have completed
   * @return true if stream is idle, false if work is pending
   */
  bool is_complete() const {
    cudaError_t err = cudaStreamQuery(stream_);
    if (err == cudaSuccess) {
      return true;
    } else if (err == cudaErrorNotReady) {
      return false;
    } else {
      throw std::runtime_error("Stream query failed: " + std::string(cudaGetErrorString(err)));
    }
  }

private:
  void cleanup() {
    if (owns_stream_ && stream_ != nullptr) {
      cudaStreamSynchronize(stream_);
      cudaStreamDestroy(stream_);
    }
  }
};

/**
 * @brief RAII wrapper for CUDA events with modern timing support
 *
 * Replaces the deprecated unique_eager_event with a clean, modern interface
 * for event management and timing.
 */
class CudaEvent {
private:
  cudaEvent_t event_;
  bool owns_event_;

public:
  /**
   * @brief Create a new CUDA event
   * @param flags Event creation flags (default: cudaEventDefault)
   */
  explicit CudaEvent(unsigned int flags = cudaEventDefault) : owns_event_(true) {
    cudaError_t err = cudaEventCreateWithFlags(&event_, flags);
    if (err != cudaSuccess) {
      throw std::runtime_error("Failed to create CUDA event: " + std::string(cudaGetErrorString(err)));
    }
  }

  /**
   * @brief Create wrapper for existing event (non-owning)
   */
  CudaEvent(cudaEvent_t event, bool take_ownership = false) : event_(event), owns_event_(take_ownership) {}

  // Disable copy
  CudaEvent(const CudaEvent&) = delete;
  CudaEvent& operator=(const CudaEvent&) = delete;

  // Enable move
  CudaEvent(CudaEvent&& other) noexcept : event_(other.event_), owns_event_(other.owns_event_) {
    other.event_ = nullptr;
    other.owns_event_ = false;
  }

  CudaEvent& operator=(CudaEvent&& other) noexcept {
    if (this != &other) {
      cleanup();
      event_ = other.event_;
      owns_event_ = other.owns_event_;
      other.event_ = nullptr;
      other.owns_event_ = false;
    }
    return *this;
  }

  ~CudaEvent() { cleanup(); }

  /**
   * @brief Record this event on a stream
   */
  void record(cudaStream_t stream = 0) {
    cudaError_t err = cudaEventRecord(event_, stream);
    if (err != cudaSuccess) {
      throw std::runtime_error("Failed to record CUDA event: " + std::string(cudaGetErrorString(err)));
    }
  }

  /**
   * @brief Record this event on an execution context
   */
  void record(const CudaExecutionContext& ctx) { record(ctx.stream()); }

  /**
   * @brief Wait for this event to complete
   */
  void synchronize() {
    cudaError_t err = cudaEventSynchronize(event_);
    if (err != cudaSuccess) {
      throw std::runtime_error("Event synchronization failed: " + std::string(cudaGetErrorString(err)));
    }
  }

  /**
   * @brief Query if this event has been recorded
   * @return true if recorded, false if pending
   */
  bool is_complete() const {
    cudaError_t err = cudaEventQuery(event_);
    if (err == cudaSuccess) {
      return true;
    } else if (err == cudaErrorNotReady) {
      return false;
    } else {
      throw std::runtime_error("Event query failed: " + std::string(cudaGetErrorString(err)));
    }
  }

  /**
   * @brief Calculate elapsed time between two events
   * @param start Start event (must have been recorded before this event)
   * @return Elapsed time in milliseconds
   */
  float elapsed_time(const CudaEvent& start) const {
    float ms = 0.0f;
    cudaError_t err = cudaEventElapsedTime(&ms, start.event_, event_);
    if (err != cudaSuccess) {
      throw std::runtime_error("Failed to calculate elapsed time: " + std::string(cudaGetErrorString(err)));
    }
    return ms;
  }

  /**
   * @brief Get raw event handle
   */
  cudaEvent_t get() const { return event_; }

private:
  void cleanup() {
    if (owns_event_ && event_ != nullptr) {
      cudaEventDestroy(event_);
    }
  }
};

/**
 * @brief Timer utility using CUDA events for accurate GPU timing
 */
class CudaTimer {
private:
  CudaEvent start_;
  CudaEvent stop_;
  bool running_;

public:
  CudaTimer() : start_(static_cast<unsigned int>(cudaEventDefault)), stop_(static_cast<unsigned int>(cudaEventDefault)), running_(false) {}

  /**
   * @brief Start timing on a stream
   */
  void start(cudaStream_t stream = 0) {
    start_.record(stream);
    running_ = true;
  }

  /**
   * @brief Start timing on an execution context
   */
  void start(const CudaExecutionContext& ctx) { start(ctx.stream()); }

  /**
   * @brief Stop timing on a stream
   */
  void stop(cudaStream_t stream = 0) {
    if (!running_) {
      throw std::runtime_error("Timer not started");
    }
    stop_.record(stream);
    running_ = false;
  }

  /**
   * @brief Stop timing on an execution context
   */
  void stop(const CudaExecutionContext& ctx) { stop(ctx.stream()); }

  /**
   * @brief Get elapsed time in milliseconds
   * @return Elapsed time, or -1 if timer not stopped
   */
  float elapsed_ms() {
    if (running_) {
      return -1.0f;
    }
    stop_.synchronize();
    return stop_.elapsed_time(start_);
  }
};

/**
 * @brief Global execution context manager for default operations
 */
class GlobalExecutionContext {
private:
  static std::unique_ptr<CudaExecutionContext> default_context_;

public:
  /**
   * @brief Get or create the default execution context
   */
  static CudaExecutionContext& get_default() {
    if (!default_context_) {
      default_context_ = std::make_unique<CudaExecutionContext>("global_default");
    }
    return *default_context_;
  }

  /**
   * @brief Reset the default context (useful for cleanup)
   */
  static void reset() { default_context_.reset(); }
};

// Initialize static member
inline std::unique_ptr<CudaExecutionContext> GlobalExecutionContext::default_context_ = nullptr;

/**
 * @brief Helper function to get default execution policy
 */
inline auto default_execution_policy() {
  return GlobalExecutionContext::get_default().policy();
}

/**
 * @brief Scoped stream synchronization guard
 */
class StreamSyncGuard {
private:
  cudaStream_t stream_;

public:
  explicit StreamSyncGuard(cudaStream_t stream) : stream_(stream) {}
  explicit StreamSyncGuard(const CudaExecutionContext& ctx) : stream_(ctx.stream()) {}

  ~StreamSyncGuard() { cudaStreamSynchronize(stream_); }

  // Non-copyable, non-movable
  StreamSyncGuard(const StreamSyncGuard&) = delete;
  StreamSyncGuard& operator=(const StreamSyncGuard&) = delete;
  StreamSyncGuard(StreamSyncGuard&&) = delete;
  StreamSyncGuard& operator=(StreamSyncGuard&&) = delete;
};

}  // namespace cuda
}  // namespace fast_gicp
