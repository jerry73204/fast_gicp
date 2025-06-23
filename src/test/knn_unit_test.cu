/**
 * @file knn_unit_test.cu
 * @brief Unit tests for KNN algorithm components
 *
 * Tests individual components of the KNN implementation in isolation
 * to ensure each part works correctly before integration testing.
 */

#include <gtest/gtest.h>
#include <vector>
#include <algorithm>
#include <random>
#include <cmath>

#include <Eigen/Core>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#ifdef USE_VGICP_CUDA
#include <fast_gicp/cuda/brute_force_knn.cuh>
#endif

class KNNUnitTest : public ::testing::Test {
protected:
  void SetUp() override {
    rng_.seed(42);  // Fixed seed for reproducible tests
  }

  // Generate deterministic test points
  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> generateGridPoints(int size) {
    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> points;
    points.reserve(size);

    int side = std::ceil(std::cbrt(size));
    for (int i = 0; i < size; ++i) {
      int x = i % side;
      int y = (i / side) % side;
      int z = i / (side * side);
      points.emplace_back(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z));
    }

    return points;
  }

  // Generate points with known distances
  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> generateKnownDistancePoints() {
    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> points;

    // Origin
    points.emplace_back(0.0f, 0.0f, 0.0f);
    // Unit distance points
    points.emplace_back(1.0f, 0.0f, 0.0f);  // distance = 1
    points.emplace_back(0.0f, 1.0f, 0.0f);  // distance = 1
    points.emplace_back(0.0f, 0.0f, 1.0f);  // distance = 1
    // sqrt(2) distance points
    points.emplace_back(1.0f, 1.0f, 0.0f);  // distance = sqrt(2)
    points.emplace_back(1.0f, 0.0f, 1.0f);  // distance = sqrt(2)
    points.emplace_back(0.0f, 1.0f, 1.0f);  // distance = sqrt(2)
    // sqrt(3) distance point
    points.emplace_back(1.0f, 1.0f, 1.0f);  // distance = sqrt(3)

    return points;
  }

  // CPU ground truth implementation
  std::vector<std::pair<float, int>> computeCPUKNN(
    const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& source,
    const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& target,
    int src_idx,
    int k) {
    std::vector<std::pair<float, int>> distances;
    distances.reserve(target.size());

    const auto& src_point = source[src_idx];
    for (size_t i = 0; i < target.size(); ++i) {
      Eigen::Vector3f diff = target[i] - src_point;
      float sq_dist = diff.squaredNorm();
      distances.emplace_back(sq_dist, static_cast<int>(i));
    }

    // Sort with tie-breaking by index
    std::sort(distances.begin(), distances.end(), [](const auto& a, const auto& b) {
      if (a.first != b.first) return a.first < b.first;
      return a.second < b.second;
    });

    distances.resize(std::min(k, static_cast<int>(distances.size())));
    return distances;
  }

  std::mt19937 rng_;
};

#ifdef USE_VGICP_CUDA

TEST_F(KNNUnitTest, SinglePointKnownDistances) {
  // Test with known point distances
  auto points = generateKnownDistancePoints();

  thrust::host_vector<Eigen::Vector3f> h_source(1);
  h_source[0] = points[0];  // Origin as source

  thrust::host_vector<Eigen::Vector3f> h_target(points.begin() + 1, points.end());  // All other points as targets

  thrust::device_vector<Eigen::Vector3f> d_source = h_source;
  thrust::device_vector<Eigen::Vector3f> d_target = h_target;
  thrust::device_vector<thrust::pair<float, int>> d_results;

  // Test k=3 (should get the 3 unit distance points)
  fast_gicp::cuda::brute_force_knn_search(d_source, d_target, 3, d_results);

  thrust::host_vector<thrust::pair<float, int>> h_results = d_results;

  ASSERT_EQ(h_results.size(), 3);

  // Should get indices 0, 1, 2 (the three unit distance points) with distance 1.0
  for (int i = 0; i < 3; ++i) {
    EXPECT_FLOAT_EQ(h_results[i].first, 1.0f) << "Distance " << i << " should be 1.0";
    EXPECT_GE(h_results[i].second, 0) << "Index should be valid";
    EXPECT_LT(h_results[i].second, 7) << "Index should be in range";
  }

  // Results should be sorted by distance, then by index
  for (int i = 0; i < 2; ++i) {
    EXPECT_LE(h_results[i].first, h_results[i + 1].first) << "Results should be sorted by distance";
    if (h_results[i].first == h_results[i + 1].first) {
      EXPECT_LT(h_results[i].second, h_results[i + 1].second) << "Ties should be broken by index";
    }
  }
}

TEST_F(KNNUnitTest, GridPointsExactDistances) {
  // Test with grid points for exact distance validation
  auto points = generateGridPoints(27);  // 3x3x3 grid

  thrust::host_vector<Eigen::Vector3f> h_source(1);
  h_source[0] = points[0];  // Corner point (0,0,0) as source

  thrust::host_vector<Eigen::Vector3f> h_target(points.begin() + 1, points.end());

  thrust::device_vector<Eigen::Vector3f> d_source = h_source;
  thrust::device_vector<Eigen::Vector3f> d_target = h_target;
  thrust::device_vector<thrust::pair<float, int>> d_results;

  // Test k=6 (should get 6 nearest neighbors)
  fast_gicp::cuda::brute_force_knn_search(d_source, d_target, 6, d_results);

  thrust::host_vector<thrust::pair<float, int>> h_results = d_results;

  ASSERT_EQ(h_results.size(), 6);

  // Verify against CPU implementation
  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> source_vec(h_source.begin(), h_source.end());
  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> target_vec(h_target.begin(), h_target.end());

  auto cpu_results = computeCPUKNN(source_vec, target_vec, 0, 6);

  for (size_t i = 0; i < h_results.size(); ++i) {
    EXPECT_FLOAT_EQ(h_results[i].first, cpu_results[i].first) << "Distance mismatch at position " << i;
    EXPECT_EQ(h_results[i].second, cpu_results[i].second) << "Index mismatch at position " << i;
  }
}

TEST_F(KNNUnitTest, EdgeCaseEmptyInputs) {
  thrust::device_vector<Eigen::Vector3f> d_source;
  thrust::device_vector<Eigen::Vector3f> d_target;
  thrust::device_vector<thrust::pair<float, int>> d_results;

  // Test empty source
  fast_gicp::cuda::brute_force_knn_search(d_source, d_target, 5, d_results);
  EXPECT_TRUE(d_results.empty()) << "Empty source should produce empty results";

  // Test empty target
  d_source.resize(1);
  d_source[0] = Eigen::Vector3f(0, 0, 0);
  fast_gicp::cuda::brute_force_knn_search(d_source, d_target, 5, d_results);
  EXPECT_TRUE(d_results.empty()) << "Empty target should produce empty results";
}

TEST_F(KNNUnitTest, EdgeCaseKLargerThanTarget) {
  auto points = generateGridPoints(5);

  thrust::host_vector<Eigen::Vector3f> h_source(1);
  h_source[0] = points[0];

  thrust::host_vector<Eigen::Vector3f> h_target(points.begin() + 1, points.end());  // 4 target points

  thrust::device_vector<Eigen::Vector3f> d_source = h_source;
  thrust::device_vector<Eigen::Vector3f> d_target = h_target;
  thrust::device_vector<thrust::pair<float, int>> d_results;

  // Request k=10 but only have 4 target points
  fast_gicp::cuda::brute_force_knn_search(d_source, d_target, 10, d_results);

  thrust::host_vector<thrust::pair<float, int>> h_results = d_results;

  // Should return all 4 target points
  EXPECT_EQ(h_results.size(), 4) << "Should return all available target points";

  // All results should have valid indices
  for (const auto& result : h_results) {
    EXPECT_GE(result.second, 0) << "All indices should be valid";
    EXPECT_LT(result.second, 4) << "All indices should be in range";
  }
}

TEST_F(KNNUnitTest, EdgeCaseSinglePoint) {
  thrust::device_vector<Eigen::Vector3f> d_source(1);
  thrust::device_vector<Eigen::Vector3f> d_target(1);
  thrust::device_vector<thrust::pair<float, int>> d_results;

  d_source[0] = Eigen::Vector3f(0, 0, 0);
  d_target[0] = Eigen::Vector3f(1, 1, 1);

  fast_gicp::cuda::brute_force_knn_search(d_source, d_target, 1, d_results);

  thrust::host_vector<thrust::pair<float, int>> h_results = d_results;

  ASSERT_EQ(h_results.size(), 1);
  EXPECT_FLOAT_EQ(h_results[0].first, 3.0f) << "Distance should be sqrt(3)^2 = 3";
  EXPECT_EQ(h_results[0].second, 0) << "Index should be 0";
}

TEST_F(KNNUnitTest, TieBreakingConsistency) {
  // Create points with identical distances to test tie-breaking
  thrust::device_vector<Eigen::Vector3f> d_source(1);
  thrust::device_vector<Eigen::Vector3f> d_target(4);
  thrust::device_vector<thrust::pair<float, int>> d_results;

  d_source[0] = Eigen::Vector3f(0, 0, 0);  // Origin

  // Four points at unit distance (all at distance 1.0)
  d_target[0] = Eigen::Vector3f(1, 0, 0);
  d_target[1] = Eigen::Vector3f(0, 1, 0);
  d_target[2] = Eigen::Vector3f(0, 0, 1);
  d_target[3] = Eigen::Vector3f(-1, 0, 0);

  fast_gicp::cuda::brute_force_knn_search(d_source, d_target, 4, d_results);

  thrust::host_vector<thrust::pair<float, int>> h_results = d_results;

  ASSERT_EQ(h_results.size(), 4);

  // All distances should be 1.0
  for (const auto& result : h_results) {
    EXPECT_FLOAT_EQ(result.first, 1.0f) << "All distances should be 1.0";
  }

  // Indices should be sorted (tie-breaking by index)
  for (size_t i = 0; i < h_results.size() - 1; ++i) {
    EXPECT_LT(h_results[i].second, h_results[i + 1].second) << "Indices should be sorted for tie-breaking";
  }
}

TEST_F(KNNUnitTest, MultipleSourcePoints) {
  auto points = generateGridPoints(10);

  thrust::host_vector<Eigen::Vector3f> h_source(points.begin(), points.begin() + 3);  // First 3 as source
  thrust::host_vector<Eigen::Vector3f> h_target(points.begin() + 3, points.end());    // Rest as target

  thrust::device_vector<Eigen::Vector3f> d_source = h_source;
  thrust::device_vector<Eigen::Vector3f> d_target = h_target;
  thrust::device_vector<thrust::pair<float, int>> d_results;

  int k = 3;
  fast_gicp::cuda::brute_force_knn_search(d_source, d_target, k, d_results);

  thrust::host_vector<thrust::pair<float, int>> h_results = d_results;

  ASSERT_EQ(h_results.size(), 3 * k);  // 3 source points * k neighbors each

  // Verify each source point's results independently
  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> source_vec(h_source.begin(), h_source.end());
  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> target_vec(h_target.begin(), h_target.end());

  for (int src_idx = 0; src_idx < 3; ++src_idx) {
    auto cpu_results = computeCPUKNN(source_vec, target_vec, src_idx, k);

    for (int i = 0; i < k; ++i) {
      int result_idx = src_idx * k + i;
      EXPECT_FLOAT_EQ(h_results[result_idx].first, cpu_results[i].first) << "Distance mismatch for source " << src_idx << " neighbor " << i;
      EXPECT_EQ(h_results[result_idx].second, cpu_results[i].second) << "Index mismatch for source " << src_idx << " neighbor " << i;
    }
  }
}

TEST_F(KNNUnitTest, LargeKValue) {
  // Test with k=32 (boundary case for CUB implementation)
  auto points = generateGridPoints(50);

  thrust::host_vector<Eigen::Vector3f> h_source(1);
  h_source[0] = points[0];

  thrust::host_vector<Eigen::Vector3f> h_target(points.begin() + 1, points.end());

  thrust::device_vector<Eigen::Vector3f> d_source = h_source;
  thrust::device_vector<Eigen::Vector3f> d_target = h_target;
  thrust::device_vector<thrust::pair<float, int>> d_results;

  int k = 32;
  fast_gicp::cuda::brute_force_knn_search(d_source, d_target, k, d_results);

  thrust::host_vector<thrust::pair<float, int>> h_results = d_results;

  ASSERT_EQ(h_results.size(), k);

  // Verify results are sorted
  for (size_t i = 0; i < h_results.size() - 1; ++i) {
    EXPECT_LE(h_results[i].first, h_results[i + 1].first) << "Results should be sorted by distance";
    if (h_results[i].first == h_results[i + 1].first) {
      EXPECT_LT(h_results[i].second, h_results[i + 1].second) << "Ties should be broken by index";
    }
  }

  // Verify against CPU implementation
  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> source_vec(h_source.begin(), h_source.end());
  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> target_vec(h_target.begin(), h_target.end());

  auto cpu_results = computeCPUKNN(source_vec, target_vec, 0, k);

  for (size_t i = 0; i < h_results.size(); ++i) {
    EXPECT_FLOAT_EQ(h_results[i].first, cpu_results[i].first) << "Distance mismatch at position " << i;
    EXPECT_EQ(h_results[i].second, cpu_results[i].second) << "Index mismatch at position " << i;
  }
}

#else

TEST_F(KNNUnitTest, CUDANotAvailable) {
  GTEST_SKIP() << "CUDA not available - skipping KNN unit tests";
}

#endif

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}