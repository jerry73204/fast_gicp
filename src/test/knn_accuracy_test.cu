/**
 * @file knn_accuracy_test.cu
 * @brief Numerical accuracy tests for KNN implementation
 *
 * Tests numerical accuracy and precision of the KNN implementation
 * with hand-calculated reference results and edge cases.
 */

#include <gtest/gtest.h>
#include <vector>
#include <algorithm>
#include <random>
#include <cmath>
#include <limits>

#include <Eigen/Core>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#ifdef USE_VGICP_CUDA
#include <fast_gicp/cuda/brute_force_knn.cuh>
#endif

class KNNAccuracyTest : public ::testing::Test {
protected:
  void SetUp() override {
    rng_.seed(12345);  // Fixed seed for reproducible tests
  }

  // Hand-calculated reference test case
  struct ReferenceTestCase {
    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> source;
    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> target;
    std::vector<std::vector<std::pair<float, int>>> expected_results;  // Per source point
    std::string description;
  };

  // Create reference test cases with known exact answers
  std::vector<ReferenceTestCase> createReferenceTestCases() {
    std::vector<ReferenceTestCase> cases;

    // Case 1: Origin to axis points
    {
      ReferenceTestCase case1;
      case1.description = "Origin to axis points";
      case1.source.emplace_back(0.0f, 0.0f, 0.0f);

      case1.target.emplace_back(1.0f, 0.0f, 0.0f);  // distance = 1
      case1.target.emplace_back(0.0f, 2.0f, 0.0f);  // distance = 4
      case1.target.emplace_back(0.0f, 0.0f, 3.0f);  // distance = 9
      case1.target.emplace_back(2.0f, 0.0f, 0.0f);  // distance = 4

      // Expected k=3 results: indices 0, 1, 3 (or 0, 3, 1) with distances 1, 4, 4
      case1.expected_results.resize(1);
      case1.expected_results[0] = {
        {1.0f, 0},
        {4.0f, 1},
        {4.0f, 3}  // Sorted by distance, tie-broken by index
      };

      cases.push_back(case1);
    }

    // Case 2: Multiple source points
    {
      ReferenceTestCase case2;
      case2.description = "Multiple source points";
      case2.source.emplace_back(0.0f, 0.0f, 0.0f);
      case2.source.emplace_back(1.0f, 0.0f, 0.0f);

      case2.target.emplace_back(0.5f, 0.0f, 0.0f);  // From src0: 0.25, from src1: 0.25
      case2.target.emplace_back(2.0f, 0.0f, 0.0f);  // From src0: 4, from src1: 1
      case2.target.emplace_back(0.0f, 1.0f, 0.0f);  // From src0: 1, from src1: sqrt(2)^2 = 2

      case2.expected_results.resize(2);
      // Source 0 results (k=2): target 0 (0.25), target 2 (1.0)
      case2.expected_results[0] = {{0.25f, 0}, {1.0f, 2}};
      // Source 1 results (k=2): target 0 (0.25), target 1 (1.0)
      case2.expected_results[1] = {{0.25f, 0}, {1.0f, 1}};

      cases.push_back(case2);
    }

    // Case 3: Tie-breaking test
    {
      ReferenceTestCase case3;
      case3.description = "Tie-breaking by index";
      case3.source.emplace_back(0.0f, 0.0f, 0.0f);

      // Four points at exactly the same distance
      case3.target.emplace_back(1.0f, 0.0f, 0.0f);   // distance = 1, index 0
      case3.target.emplace_back(0.0f, 1.0f, 0.0f);   // distance = 1, index 1
      case3.target.emplace_back(-1.0f, 0.0f, 0.0f);  // distance = 1, index 2
      case3.target.emplace_back(0.0f, -1.0f, 0.0f);  // distance = 1, index 3

      case3.expected_results.resize(1);
      case3.expected_results[0] = {
        {1.0f, 0},
        {1.0f, 1},
        {1.0f, 2}  // k=3, sorted by index for ties
      };

      cases.push_back(case3);
    }

    return cases;
  }

  // Test with precise floating point values
  void testPrecisionCase(const std::vector<float>& distances, float tolerance = 1e-6f) {
    thrust::device_vector<Eigen::Vector3f> d_source(1);
    thrust::device_vector<Eigen::Vector3f> d_target;
    thrust::device_vector<thrust::pair<float, int>> d_results;

    d_source[0] = Eigen::Vector3f(0.0f, 0.0f, 0.0f);

    // Create target points at specified distances along x-axis
    d_target.resize(distances.size());
    thrust::host_vector<Eigen::Vector3f> h_target(distances.size());
    for (size_t i = 0; i < distances.size(); ++i) {
      h_target[i] = Eigen::Vector3f(std::sqrt(distances[i]), 0.0f, 0.0f);
    }
    d_target = h_target;

    int k = std::min(3, static_cast<int>(distances.size()));
    fast_gicp::cuda::brute_force_knn_search(d_source, d_target, k, d_results);

    thrust::host_vector<thrust::pair<float, int>> h_results = d_results;

    // Verify distances match expected values
    std::vector<float> sorted_distances = distances;
    std::sort(sorted_distances.begin(), sorted_distances.end());

    for (int i = 0; i < k; ++i) {
      EXPECT_NEAR(h_results[i].first, sorted_distances[i], tolerance) << "Distance precision test failed at position " << i;
    }
  }

  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> generateRandomPoints(int count) {
    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> points;
    points.reserve(count);

    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    for (int i = 0; i < count; ++i) {
      points.emplace_back(dist(rng_), dist(rng_), dist(rng_));
    }

    return points;
  }

  std::mt19937 rng_;
};

#ifdef USE_VGICP_CUDA

TEST_F(KNNAccuracyTest, ReferenceTestCases) {
  auto test_cases = createReferenceTestCases();

  for (const auto& test_case : test_cases) {
    SCOPED_TRACE(test_case.description);

    thrust::host_vector<Eigen::Vector3f> h_source(test_case.source.begin(), test_case.source.end());
    thrust::host_vector<Eigen::Vector3f> h_target(test_case.target.begin(), test_case.target.end());

    thrust::device_vector<Eigen::Vector3f> d_source = h_source;
    thrust::device_vector<Eigen::Vector3f> d_target = h_target;
    thrust::device_vector<thrust::pair<float, int>> d_results;

    int k = test_case.expected_results[0].size();
    fast_gicp::cuda::brute_force_knn_search(d_source, d_target, k, d_results);

    thrust::host_vector<thrust::pair<float, int>> h_results = d_results;

    // Verify results for each source point
    for (size_t src_idx = 0; src_idx < test_case.source.size(); ++src_idx) {
      const auto& expected = test_case.expected_results[src_idx];

      for (size_t i = 0; i < expected.size(); ++i) {
        size_t result_idx = src_idx * k + i;
        EXPECT_FLOAT_EQ(h_results[result_idx].first, expected[i].first) << "Distance mismatch for source " << src_idx << " neighbor " << i;
        EXPECT_EQ(h_results[result_idx].second, expected[i].second) << "Index mismatch for source " << src_idx << " neighbor " << i;
      }
    }
  }
}

TEST_F(KNNAccuracyTest, FloatingPointPrecision) {
  // Test with various floating-point precision scenarios

  // Very small distances
  testPrecisionCase({1e-6f, 2e-6f, 3e-6f, 4e-6f}, 1e-9f);

  // Very large distances
  testPrecisionCase({1e6f, 2e6f, 3e6f, 4e6f}, 1e3f);

  // Mixed scale distances
  testPrecisionCase({1e-3f, 1.0f, 1e3f, 1e6f}, 1e-6f);

  // Nearly equal distances
  testPrecisionCase({1.0f, 1.0000001f, 1.0000002f, 1.0000003f}, 1e-6f);
}

TEST_F(KNNAccuracyTest, ExactlyEqualDistances) {
  // Test case where multiple points have exactly the same distance
  thrust::device_vector<Eigen::Vector3f> d_source(1);
  thrust::device_vector<Eigen::Vector3f> d_target(8);
  thrust::device_vector<thrust::pair<float, int>> d_results;

  d_source[0] = Eigen::Vector3f(0.0f, 0.0f, 0.0f);

  // 8 points at the corners of a unit cube (all at distance sqrt(3))
  thrust::host_vector<Eigen::Vector3f> h_target(8);
  int idx = 0;
  for (int x = 0; x < 2; ++x) {
    for (int y = 0; y < 2; ++y) {
      for (int z = 0; z < 2; ++z) {
        if (x || y || z) {  // Skip origin
          h_target[idx++] = Eigen::Vector3f(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z));
        }
      }
    }
  }
  h_target.resize(7);  // Remove the last empty slot
  d_target = h_target;

  fast_gicp::cuda::brute_force_knn_search(d_source, d_target, 7, d_results);

  thrust::host_vector<thrust::pair<float, int>> h_results = d_results;

  ASSERT_EQ(h_results.size(), 7);

  // All distances should be exactly sqrt(3)^2 = 3
  for (const auto& result : h_results) {
    EXPECT_FLOAT_EQ(result.first, 3.0f) << "All distances should be exactly 3.0";
  }

  // Indices should be sorted (tie-breaking)
  for (size_t i = 0; i < h_results.size() - 1; ++i) {
    EXPECT_LT(h_results[i].second, h_results[i + 1].second) << "Indices should be sorted for tie-breaking";
  }
}

TEST_F(KNNAccuracyTest, ExtremePrecisionValues) {
  // Test with values near floating-point limits
  thrust::device_vector<Eigen::Vector3f> d_source(1);
  thrust::device_vector<Eigen::Vector3f> d_target(3);
  thrust::device_vector<thrust::pair<float, int>> d_results;

  d_source[0] = Eigen::Vector3f(0.0f, 0.0f, 0.0f);

  thrust::host_vector<Eigen::Vector3f> h_target(3);
  h_target[0] = Eigen::Vector3f(std::numeric_limits<float>::epsilon(), 0.0f, 0.0f);
  h_target[1] = Eigen::Vector3f(2.0f * std::numeric_limits<float>::epsilon(), 0.0f, 0.0f);
  h_target[2] = Eigen::Vector3f(1e-10f, 0.0f, 0.0f);
  d_target = h_target;

  fast_gicp::cuda::brute_force_knn_search(d_source, d_target, 3, d_results);

  thrust::host_vector<thrust::pair<float, int>> h_results = d_results;

  ASSERT_EQ(h_results.size(), 3);

  // Results should be sorted by distance
  for (size_t i = 0; i < h_results.size() - 1; ++i) {
    EXPECT_LE(h_results[i].first, h_results[i + 1].first) << "Results should be sorted even for extreme precision values";
  }
}

TEST_F(KNNAccuracyTest, GraduallyIncreasingK) {
  // Test with gradually increasing k values to stress different code paths
  auto points = generateRandomPoints(100);

  thrust::host_vector<Eigen::Vector3f> h_source(1);
  h_source[0] = points[0];

  thrust::host_vector<Eigen::Vector3f> h_target(points.begin() + 1, points.end());

  thrust::device_vector<Eigen::Vector3f> d_source = h_source;
  thrust::device_vector<Eigen::Vector3f> d_target = h_target;

  // Test k values that trigger different implementation paths
  std::vector<int> k_values = {1, 5, 10, 20, 32, 40, 50, 99};

  for (int k : k_values) {
    SCOPED_TRACE("k = " + std::to_string(k));

    thrust::device_vector<thrust::pair<float, int>> d_results;
    fast_gicp::cuda::brute_force_knn_search(d_source, d_target, k, d_results);

    thrust::host_vector<thrust::pair<float, int>> h_results = d_results;

    ASSERT_EQ(h_results.size(), k);

    // Verify results are sorted
    for (size_t i = 0; i < h_results.size() - 1; ++i) {
      EXPECT_LE(h_results[i].first, h_results[i + 1].first) << "Results should be sorted for k=" << k;

      if (h_results[i].first == h_results[i + 1].first) {
        EXPECT_LT(h_results[i].second, h_results[i + 1].second) << "Ties should be broken by index for k=" << k;
      }
    }

    // All indices should be valid
    for (const auto& result : h_results) {
      EXPECT_GE(result.second, 0) << "Invalid index for k=" << k;
      EXPECT_LT(result.second, 99) << "Index out of range for k=" << k;
    }
  }
}

TEST_F(KNNAccuracyTest, DeterministicResults) {
  // Test that results are deterministic across multiple runs
  auto points = generateRandomPoints(50);

  thrust::host_vector<Eigen::Vector3f> h_source(points.begin(), points.begin() + 5);
  thrust::host_vector<Eigen::Vector3f> h_target(points.begin() + 5, points.end());

  thrust::device_vector<Eigen::Vector3f> d_source = h_source;
  thrust::device_vector<Eigen::Vector3f> d_target = h_target;

  // Run the same test multiple times
  thrust::host_vector<thrust::pair<float, int>> reference_results;

  for (int run = 0; run < 5; ++run) {
    thrust::device_vector<thrust::pair<float, int>> d_results;
    fast_gicp::cuda::brute_force_knn_search(d_source, d_target, 10, d_results);

    thrust::host_vector<thrust::pair<float, int>> h_results = d_results;

    if (run == 0) {
      reference_results = h_results;
    } else {
      ASSERT_EQ(h_results.size(), reference_results.size()) << "Result size changed between runs";

      for (size_t i = 0; i < h_results.size(); ++i) {
        EXPECT_FLOAT_EQ(h_results[i].first, reference_results[i].first) << "Distance changed between runs at position " << i;
        EXPECT_EQ(h_results[i].second, reference_results[i].second) << "Index changed between runs at position " << i;
      }
    }
  }
}

#else

TEST_F(KNNAccuracyTest, CUDANotAvailable) {
  GTEST_SKIP() << "CUDA not available - skipping KNN accuracy tests";
}

#endif

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}