/**
 * @file knn_test.cpp
 * @brief Unit tests for the Pure Thrust/CUB k-nearest neighbor implementation
 *
 * This file tests the brute_force_knn_search function to ensure correctness
 * of the Phase 4 CUDA 12.x modernization implementation.
 */

#include <gtest/gtest.h>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>

#include <Eigen/Core>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#ifdef USE_VGICP_CUDA
#include <fast_gicp/cuda/brute_force_knn.cuh>
#endif

class KNNTest : public ::testing::Test {
protected:
  using Vector3fVector = std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>;

  void SetUp() override {
    // Initialize random number generator
    rng_.seed(42);  // Fixed seed for reproducibility
  }

  // Generate random point cloud
  Vector3fVector generateRandomPoints(int num_points, float range = 10.0f) {
    Vector3fVector points;
    points.reserve(num_points);

    std::uniform_real_distribution<float> dist(-range, range);

    for (int i = 0; i < num_points; ++i) {
      points.emplace_back(dist(rng_), dist(rng_), dist(rng_));
    }

    return points;
  }

  // Generate grid points
  Vector3fVector generateGridPoints(int grid_size) {
    Vector3fVector points;
    points.reserve(grid_size * grid_size * grid_size);

    for (int x = 0; x < grid_size; ++x) {
      for (int y = 0; y < grid_size; ++y) {
        for (int z = 0; z < grid_size; ++z) {
          points.emplace_back(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z));
        }
      }
    }

    return points;
  }

  // Compute ground truth KNN using CPU brute force
  std::vector<std::pair<float, int>> computeGroundTruthKNN(const Vector3fVector& source, const Vector3fVector& target, int source_idx, int k) {
    std::vector<std::pair<float, int>> distances;
    distances.reserve(target.size());

    // Compute all distances
    for (int i = 0; i < target.size(); ++i) {
      float dist = (target[i] - source[source_idx]).squaredNorm();
      distances.emplace_back(dist, i);
    }

    // Sort by distance
    std::partial_sort(distances.begin(), distances.begin() + std::min(k, static_cast<int>(distances.size())), distances.end(), [](const auto& a, const auto& b) {
      return a.first < b.first;
    });

    distances.resize(std::min(k, static_cast<int>(distances.size())));
    return distances;
  }

  // Compare KNN results with adaptive tolerance
  void compareKNNResults(const std::vector<thrust::pair<float, int>>& gpu_results, const std::vector<std::pair<float, int>>& cpu_results, float base_tolerance = 1e-4f) {
    ASSERT_EQ(gpu_results.size(), cpu_results.size());

    for (size_t i = 0; i < gpu_results.size(); ++i) {
      // Use adaptive tolerance based on distance magnitude
      float distance_magnitude = std::max(gpu_results[i].first, cpu_results[i].first);
      float adaptive_tolerance = std::max(base_tolerance, distance_magnitude * 1e-5f);

      // Check distance with adaptive tolerance
      EXPECT_NEAR(gpu_results[i].first, cpu_results[i].first, adaptive_tolerance)
        << "Distance mismatch at position " << i << " (GPU: " << gpu_results[i].first << ", CPU: " << cpu_results[i].first << ", tolerance: " << adaptive_tolerance << ")";

      // For distances within tolerance, check if indices are consistent
      if (std::abs(gpu_results[i].first - cpu_results[i].first) <= adaptive_tolerance) {
        float gpu_dist = gpu_results[i].first;

        // Find all CPU results with similar distance
        std::vector<int> valid_indices;
        for (const auto& cpu_result : cpu_results) {
          if (std::abs(cpu_result.first - gpu_dist) <= adaptive_tolerance) {
            valid_indices.push_back(cpu_result.second);
          }
        }

        // Check if GPU index is among valid indices
        bool found = std::find(valid_indices.begin(), valid_indices.end(), gpu_results[i].second) != valid_indices.end();
        EXPECT_TRUE(found) << "Index " << gpu_results[i].second << " not found among valid indices at position " << i << " (distance: " << gpu_dist
                           << ", tolerance: " << adaptive_tolerance << ")";
      }
    }
  }

  std::mt19937 rng_;
};

#ifdef USE_VGICP_CUDA

TEST_F(KNNTest, SmallPointCloud_VariousK) {
  // Test with small point cloud and various k values
  auto points = generateRandomPoints(50);

  thrust::host_vector<Eigen::Vector3f> h_points(points.begin(), points.end());
  thrust::device_vector<Eigen::Vector3f> d_source = h_points;
  thrust::device_vector<Eigen::Vector3f> d_target = h_points;

  std::vector<int> k_values = {1, 5, 10, 20, 30};

  for (int k : k_values) {
    thrust::device_vector<thrust::pair<float, int>> d_results;

    // Run GPU KNN
    fast_gicp::cuda::brute_force_knn_search(d_source, d_target, k, d_results, false);

    // Copy results to host
    thrust::host_vector<thrust::pair<float, int>> h_results = d_results;

    // Verify results for a few source points
    for (int src_idx = 0; src_idx < std::min(5, static_cast<int>(points.size())); ++src_idx) {
      auto gpu_knn = std::vector<thrust::pair<float, int>>(h_results.begin() + src_idx * k, h_results.begin() + (src_idx + 1) * k);

      auto cpu_knn = computeGroundTruthKNN(points, points, src_idx, k);

      compareKNNResults(gpu_knn, cpu_knn);
    }
  }
}

TEST_F(KNNTest, MediumPointCloud) {
  // Test with medium-sized point cloud
  auto source = generateRandomPoints(1000);
  auto target = generateRandomPoints(2000);

  thrust::host_vector<Eigen::Vector3f> h_source(source.begin(), source.end());
  thrust::host_vector<Eigen::Vector3f> h_target(target.begin(), target.end());
  thrust::device_vector<Eigen::Vector3f> d_source = h_source;
  thrust::device_vector<Eigen::Vector3f> d_target = h_target;

  int k = 10;
  thrust::device_vector<thrust::pair<float, int>> d_results;

  // Run GPU KNN
  fast_gicp::cuda::brute_force_knn_search(d_source, d_target, k, d_results, false);

  // Copy results to host
  thrust::host_vector<thrust::pair<float, int>> h_results = d_results;

  ASSERT_EQ(h_results.size(), source.size() * k);

  // Spot check a few points
  std::vector<int> test_indices = {0, 100, 500, 999};
  for (int src_idx : test_indices) {
    auto gpu_knn = std::vector<thrust::pair<float, int>>(h_results.begin() + src_idx * k, h_results.begin() + (src_idx + 1) * k);

    auto cpu_knn = computeGroundTruthKNN(source, target, src_idx, k);

    compareKNNResults(gpu_knn, cpu_knn);
  }
}

TEST_F(KNNTest, LargePointCloud_CUBKernel) {
  // Test large point cloud to trigger CUB kernel implementation
  auto source = generateRandomPoints(5000);
  auto target = generateRandomPoints(10000);

  thrust::host_vector<Eigen::Vector3f> h_source(source.begin(), source.end());
  thrust::host_vector<Eigen::Vector3f> h_target(target.begin(), target.end());
  thrust::device_vector<Eigen::Vector3f> d_source = h_source;
  thrust::device_vector<Eigen::Vector3f> d_target = h_target;

  // Test k values that trigger different kernel paths
  std::vector<int> k_values = {1, 5, 10, 20};

  for (int k : k_values) {
    thrust::device_vector<thrust::pair<float, int>> d_results;

    // Run GPU KNN
    fast_gicp::cuda::brute_force_knn_search(d_source, d_target, k, d_results, false);

    // Copy results to host
    thrust::host_vector<thrust::pair<float, int>> h_results = d_results;

    ASSERT_EQ(h_results.size(), source.size() * k);

    // Verify first and last points
    for (int src_idx : {0, static_cast<int>(source.size() - 1)}) {
      auto gpu_knn = std::vector<thrust::pair<float, int>>(h_results.begin() + src_idx * k, h_results.begin() + (src_idx + 1) * k);

      // Check that results are valid
      for (const auto& result : gpu_knn) {
        EXPECT_GE(result.first, 0.0f) << "Invalid distance";
        EXPECT_GE(result.second, 0) << "Invalid index";
        EXPECT_LT(result.second, target.size()) << "Index out of bounds";
      }
    }
  }
}

TEST_F(KNNTest, EdgeCase_KLargerThanTarget) {
  // Test when k > number of target points
  auto source = generateRandomPoints(10);
  auto target = generateRandomPoints(5);

  thrust::host_vector<Eigen::Vector3f> h_source(source.begin(), source.end());
  thrust::host_vector<Eigen::Vector3f> h_target(target.begin(), target.end());
  thrust::device_vector<Eigen::Vector3f> d_source = h_source;
  thrust::device_vector<Eigen::Vector3f> d_target = h_target;

  int k = 10;  // Larger than target size
  thrust::device_vector<thrust::pair<float, int>> d_results;

  // Run GPU KNN - should clamp k to target size
  fast_gicp::cuda::brute_force_knn_search(d_source, d_target, k, d_results, false);

  // Copy results to host
  thrust::host_vector<thrust::pair<float, int>> h_results = d_results;

  // Should return min(k, target.size()) neighbors for each source point
  ASSERT_EQ(h_results.size(), source.size() * std::min(k, static_cast<int>(target.size())));
}

TEST_F(KNNTest, EdgeCase_EmptyPointClouds) {
  // Test with empty point clouds
  thrust::device_vector<Eigen::Vector3f> d_empty;
  thrust::device_vector<Eigen::Vector3f> d_points(100);

  int k = 5;
  thrust::device_vector<thrust::pair<float, int>> d_results;

  // Empty source
  fast_gicp::cuda::brute_force_knn_search(d_empty, d_points, k, d_results, false);
  EXPECT_EQ(d_results.size(), 0);

  // Empty target
  fast_gicp::cuda::brute_force_knn_search(d_points, d_empty, k, d_results, false);
  EXPECT_EQ(d_results.size(), 0);

  // Both empty
  fast_gicp::cuda::brute_force_knn_search(d_empty, d_empty, k, d_results, false);
  EXPECT_EQ(d_results.size(), 0);
}

TEST_F(KNNTest, TestLargeK_ThrustPath) {
  // Test k > 32 to trigger thrust implementation path
  auto source = generateRandomPoints(100);
  auto target = generateRandomPoints(200);

  thrust::host_vector<Eigen::Vector3f> h_source(source.begin(), source.end());
  thrust::host_vector<Eigen::Vector3f> h_target(target.begin(), target.end());
  thrust::device_vector<Eigen::Vector3f> d_source = h_source;
  thrust::device_vector<Eigen::Vector3f> d_target = h_target;

  int k = 50;  // Larger than MAX_K (32) in CUB implementation
  thrust::device_vector<thrust::pair<float, int>> d_results;

  // Run GPU KNN - should use thrust implementation
  fast_gicp::cuda::brute_force_knn_search(d_source, d_target, k, d_results, false);

  // Copy results to host
  thrust::host_vector<thrust::pair<float, int>> h_results = d_results;

  ASSERT_EQ(h_results.size(), source.size() * k);

  // Verify first point
  auto gpu_knn = std::vector<thrust::pair<float, int>>(h_results.begin(), h_results.begin() + k);

  auto cpu_knn = computeGroundTruthKNN(source, target, 0, k);

  compareKNNResults(gpu_knn, cpu_knn);
}

TEST_F(KNNTest, GridPoints_ExactMatches) {
  // Test with grid points where we have exact matches
  auto points = generateGridPoints(5);  // 5x5x5 = 125 points

  thrust::host_vector<Eigen::Vector3f> h_points(points.begin(), points.end());
  thrust::device_vector<Eigen::Vector3f> d_points = h_points;

  int k = 7;  // Include self and 6 neighbors
  thrust::device_vector<thrust::pair<float, int>> d_results;

  // Run GPU KNN
  fast_gicp::cuda::brute_force_knn_search(d_points, d_points, k, d_results, false);

  // Copy results to host
  thrust::host_vector<thrust::pair<float, int>> h_results = d_results;

  // For grid points, we know the expected pattern
  // Point at (1,1,1) should have distance 0 to itself and distance 1 to 6 face neighbors
  int test_idx = 1 * 25 + 1 * 5 + 1;  // Point at (1,1,1)

  auto gpu_knn = std::vector<thrust::pair<float, int>>(h_results.begin() + test_idx * k, h_results.begin() + (test_idx + 1) * k);

  // First neighbor should be self with distance 0
  EXPECT_FLOAT_EQ(gpu_knn[0].first, 0.0f);
  EXPECT_EQ(gpu_knn[0].second, test_idx);

  // Next 6 should have distance 1 (face neighbors)
  int count_dist_1 = 0;
  for (int i = 1; i < k && i < gpu_knn.size(); ++i) {
    if (std::abs(gpu_knn[i].first - 1.0f) < 1e-5f) {
      count_dist_1++;
    }
  }
  EXPECT_EQ(count_dist_1, 6);
}

TEST_F(KNNTest, PerformanceBenchmark) {
  // Simple performance test - not for correctness but to ensure no major regression
  std::vector<int> sizes = {1000, 5000, 10000};
  int k = 10;

  for (int size : sizes) {
    auto source = generateRandomPoints(size);
    auto target = generateRandomPoints(size);

    thrust::host_vector<Eigen::Vector3f> h_source(source.begin(), source.end());
    thrust::host_vector<Eigen::Vector3f> h_target(target.begin(), target.end());
    thrust::device_vector<Eigen::Vector3f> d_source = h_source;
    thrust::device_vector<Eigen::Vector3f> d_target = h_target;

    thrust::device_vector<thrust::pair<float, int>> d_results;

    // Warm up
    fast_gicp::cuda::brute_force_knn_search(d_source, d_target, k, d_results, false);
    cudaDeviceSynchronize();

    // Time the operation
    auto start = std::chrono::high_resolution_clock::now();
    fast_gicp::cuda::brute_force_knn_search(d_source, d_target, k, d_results, false);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    // Just log the time - no hard assertions on performance
    std::cout << "KNN for " << size << " points: " << duration << " microseconds" << std::endl;
  }
}

#else

TEST_F(KNNTest, CUDANotAvailable) {
  GTEST_SKIP() << "CUDA not available - skipping KNN GPU tests";
}

#endif

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
