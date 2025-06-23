/**
 * @file knn_benchmark.cpp
 * @brief Performance benchmarks for the Pure Thrust/CUB k-nearest neighbor implementation
 *
 * This file provides performance testing and stress testing for the new KNN implementation.
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>

#include <Eigen/Core>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#ifdef USE_VGICP_CUDA
#include <fast_gicp/cuda/brute_force_knn.cuh>
#include <cuda_runtime.h>
#endif

class KNNBenchmark {
private:
  std::mt19937 rng_;

  // Generate random point cloud
  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> generateRandomPoints(int num_points, float range = 10.0f) {
    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> points;
    points.reserve(num_points);

    std::uniform_real_distribution<float> dist(-range, range);

    for (int i = 0; i < num_points; ++i) {
      points.emplace_back(dist(rng_), dist(rng_), dist(rng_));
    }

    return points;
  }

  // Time a function and return microseconds
  template <typename Func>
  double timeFunction(Func&& func, int warmup_runs = 2, int test_runs = 5) {
    // Warmup
    for (int i = 0; i < warmup_runs; ++i) {
      func();
    }

#ifdef USE_VGICP_CUDA
    cudaDeviceSynchronize();
#endif

    // Actual timing
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < test_runs; ++i) {
      func();
    }

#ifdef USE_VGICP_CUDA
    cudaDeviceSynchronize();
#endif

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    return static_cast<double>(duration) / test_runs;
  }

public:
  KNNBenchmark() : rng_(42) {}  // Fixed seed for reproducibility

#ifdef USE_VGICP_CUDA

  void benchmarkKNNPerformance() {
    std::cout << "\n=== KNN Performance Benchmark ===" << std::endl;
    std::cout << std::setw(12) << "Source Size" << std::setw(12) << "Target Size" << std::setw(8) << "k" << std::setw(15) << "Time (μs)" << std::setw(15) << "Throughput"
              << std::endl;

    std::vector<std::pair<int, int>> sizes = {{100, 500}, {500, 1000}, {1000, 2000}, {2000, 5000}, {5000, 10000}, {10000, 20000}};

    std::vector<int> k_values = {1, 5, 10, 20};

    for (const auto& size_pair : sizes) {
      int source_size = size_pair.first;
      int target_size = size_pair.second;

      auto source = generateRandomPoints(source_size);
      auto target = generateRandomPoints(target_size);

      thrust::host_vector<Eigen::Vector3f> h_source(source.begin(), source.end());
      thrust::host_vector<Eigen::Vector3f> h_target(target.begin(), target.end());
      thrust::device_vector<Eigen::Vector3f> d_source = h_source;
      thrust::device_vector<Eigen::Vector3f> d_target = h_target;

      for (int k : k_values) {
        thrust::device_vector<thrust::pair<float, int>> d_results;

        double avg_time = timeFunction([&]() { fast_gicp::cuda::brute_force_knn_search(d_source, d_target, k, d_results, false); });

        // Calculate throughput (operations per second)
        long long total_operations = static_cast<long long>(source_size) * target_size;
        double throughput = total_operations / (avg_time * 1e-6);  // ops per second

        std::cout << std::setw(12) << source_size << std::setw(12) << target_size << std::setw(8) << k << std::setw(15) << std::fixed << std::setprecision(1) << avg_time
                  << std::setw(15) << std::scientific << std::setprecision(2) << throughput << std::endl;
      }
    }
  }

  void benchmarkDifferentK() {
    std::cout << "\n=== K Value Performance Comparison ===" << std::endl;
    std::cout << std::setw(8) << "k" << std::setw(15) << "Time (μs)" << std::setw(15) << "Implementation" << std::endl;

    auto source = generateRandomPoints(2000);
    auto target = generateRandomPoints(5000);

    thrust::host_vector<Eigen::Vector3f> h_source(source.begin(), source.end());
    thrust::host_vector<Eigen::Vector3f> h_target(target.begin(), target.end());
    thrust::device_vector<Eigen::Vector3f> d_source = h_source;
    thrust::device_vector<Eigen::Vector3f> d_target = h_target;

    // Test various k values to see when implementation switches
    std::vector<int> k_values = {1, 5, 10, 15, 20, 25, 30, 32, 35, 40, 50, 64, 100};

    for (int k : k_values) {
      thrust::device_vector<thrust::pair<float, int>> d_results;

      double avg_time = timeFunction([&]() { fast_gicp::cuda::brute_force_knn_search(d_source, d_target, k, d_results, false); });

      // Determine which implementation path was likely used
      std::string impl = (k <= 32) ? "CUB Kernel" : "Thrust Sort";

      std::cout << std::setw(8) << k << std::setw(15) << std::fixed << std::setprecision(1) << avg_time << std::setw(15) << impl << std::endl;
    }
  }

  void stressTest() {
    std::cout << "\n=== Stress Testing ===" << std::endl;

    // Test with very large point clouds
    std::vector<std::pair<int, int>> stress_sizes = {{50000, 100000}, {100000, 200000}};

    for (const auto& size_pair : stress_sizes) {
      try {
        std::cout << "Testing " << size_pair.first << " x " << size_pair.second << " points..." << std::flush;

        auto source = generateRandomPoints(size_pair.first);
        auto target = generateRandomPoints(size_pair.second);

        thrust::host_vector<Eigen::Vector3f> h_source(source.begin(), source.end());
        thrust::host_vector<Eigen::Vector3f> h_target(target.begin(), target.end());
        thrust::device_vector<Eigen::Vector3f> d_source = h_source;
        thrust::device_vector<Eigen::Vector3f> d_target = h_target;

        thrust::device_vector<thrust::pair<float, int>> d_results;

        double avg_time = timeFunction([&]() { fast_gicp::cuda::brute_force_knn_search(d_source, d_target, 10, d_results, false); }, 1, 3);  // Fewer runs for stress test

        std::cout << " PASSED (" << std::fixed << std::setprecision(1) << avg_time / 1000.0 << " ms)" << std::endl;

      } catch (const std::exception& e) {
        std::cout << " FAILED: " << e.what() << std::endl;
      }
    }
  }

  void memoryUsageTest() {
    std::cout << "\n=== Memory Usage Test ===" << std::endl;

    // Check GPU memory before and after large allocations
    size_t free_before, total_before;
    cudaMemGetInfo(&free_before, &total_before);

    std::cout << "GPU Memory before test: " << (total_before - free_before) / (1024 * 1024) << " MB used, " << free_before / (1024 * 1024) << " MB free" << std::endl;

    // Allocate large point clouds
    auto source = generateRandomPoints(10000);
    auto target = generateRandomPoints(20000);

    thrust::host_vector<Eigen::Vector3f> h_source(source.begin(), source.end());
    thrust::host_vector<Eigen::Vector3f> h_target(target.begin(), target.end());
    thrust::device_vector<Eigen::Vector3f> d_source = h_source;
    thrust::device_vector<Eigen::Vector3f> d_target = h_target;
    thrust::device_vector<thrust::pair<float, int>> d_results;

    size_t free_during, total_during;
    cudaMemGetInfo(&free_during, &total_during);

    std::cout << "GPU Memory during test: " << (total_during - free_during) / (1024 * 1024) << " MB used, " << free_during / (1024 * 1024) << " MB free" << std::endl;

    // Run KNN search multiple times
    for (int i = 0; i < 10; ++i) {
      fast_gicp::cuda::brute_force_knn_search(d_source, d_target, 10, d_results, false);
    }

    // Check for memory leaks
    size_t free_after, total_after;
    cudaMemGetInfo(&free_after, &total_after);

    std::cout << "GPU Memory after test: " << (total_after - free_after) / (1024 * 1024) << " MB used, " << free_after / (1024 * 1024) << " MB free" << std::endl;

    if (std::abs(static_cast<long long>(free_after) - static_cast<long long>(free_during)) < 1024 * 1024) {
      std::cout << "✓ No significant memory leak detected" << std::endl;
    } else {
      std::cout << "⚠ Potential memory leak: " << (free_during - free_after) / (1024 * 1024) << " MB difference" << std::endl;
    }
  }

  void repeatabilityTest() {
    std::cout << "\n=== Repeatability Test ===" << std::endl;

    auto source = generateRandomPoints(1000);
    auto target = generateRandomPoints(2000);

    thrust::host_vector<Eigen::Vector3f> h_source(source.begin(), source.end());
    thrust::host_vector<Eigen::Vector3f> h_target(target.begin(), target.end());
    thrust::device_vector<Eigen::Vector3f> d_source = h_source;
    thrust::device_vector<Eigen::Vector3f> d_target = h_target;

    // Run the same KNN search multiple times
    std::vector<thrust::host_vector<thrust::pair<float, int>>> results;

    for (int run = 0; run < 5; ++run) {
      thrust::device_vector<thrust::pair<float, int>> d_results;
      fast_gicp::cuda::brute_force_knn_search(d_source, d_target, 10, d_results, false);

      thrust::host_vector<thrust::pair<float, int>> h_results = d_results;
      results.push_back(h_results);
    }

    // Compare results for consistency
    bool all_identical = true;
    for (size_t i = 1; i < results.size(); ++i) {
      if (results[i].size() != results[0].size()) {
        all_identical = false;
        break;
      }

      for (size_t j = 0; j < results[0].size(); ++j) {
        if (std::abs(results[i][j].first - results[0][j].first) > 1e-5f || results[i][j].second != results[0][j].second) {
          all_identical = false;
          break;
        }
      }

      if (!all_identical) break;
    }

    if (all_identical) {
      std::cout << "✓ All runs produced identical results" << std::endl;
    } else {
      std::cout << "⚠ Results vary between runs" << std::endl;
    }
  }

#else

  void benchmarkKNNPerformance() {
    std::cout << "CUDA not available - skipping performance benchmark" << std::endl;
  }

  void benchmarkDifferentK() {
    std::cout << "CUDA not available - skipping k value benchmark" << std::endl;
  }

  void stressTest() {
    std::cout << "CUDA not available - skipping stress test" << std::endl;
  }

  void memoryUsageTest() {
    std::cout << "CUDA not available - skipping memory usage test" << std::endl;
  }

  void repeatabilityTest() {
    std::cout << "CUDA not available - skipping repeatability test" << std::endl;
  }

#endif

  void runAllBenchmarks() {
    std::cout << "KNN Implementation Benchmark Suite" << std::endl;
    std::cout << "===================================" << std::endl;

    benchmarkKNNPerformance();
    benchmarkDifferentK();
    stressTest();
    memoryUsageTest();
    repeatabilityTest();

    std::cout << "\nBenchmark suite completed." << std::endl;
  }
};

int main() {
  KNNBenchmark benchmark;
  benchmark.runAllBenchmarks();
  return 0;
}
