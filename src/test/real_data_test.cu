/**
 * @file real_data_test.cpp
 * @brief Integration tests using real point cloud data for GPU KNN validation
 *
 * This file tests the GPU brute force KNN implementation with the same real
 * point cloud data used in gicp_test.cpp to ensure compatibility and accuracy.
 */

#include <gtest/gtest.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>

#include <fast_gicp/gicp/fast_vgicp.hpp>
#ifdef USE_VGICP_CUDA
#include <fast_gicp/gicp/fast_vgicp_cuda.hpp>
#include <fast_gicp/cuda/brute_force_knn.cuh>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#endif

class RealDataTest : public ::testing::Test {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using PointCloudConstPtr = pcl::PointCloud<pcl::PointXYZ>::ConstPtr;

  RealDataTest() {}

  virtual void SetUp() {
    // Try to load real data if available
    if (!data_directory.empty() && load(data_directory)) {
      has_real_data_ = true;
    } else {
      has_real_data_ = false;
      // Generate synthetic data as fallback
      generateSyntheticData();
    }
  }

  bool load(const std::string& data_directory) {
    relative_pose.setIdentity();

    std::ifstream ifs(data_directory + "/relative.txt");
    if (!ifs) {
      return false;
    }

    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
        ifs >> relative_pose(i, j);
      }
    }

    auto target_full = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    auto source_full = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();

    if (pcl::io::loadPCDFile(data_directory + "/251370668.pcd", *target_full) < 0 || pcl::io::loadPCDFile(data_directory + "/251371071.pcd", *source_full) < 0) {
      return false;
    }

    if (target_full->empty() || source_full->empty()) {
      return false;
    }

    // Apply voxel grid filtering
    pcl::VoxelGrid<pcl::PointXYZ> voxelgrid;
    voxelgrid.setLeafSize(0.2, 0.2, 0.2);

    auto filtered = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    voxelgrid.setInputCloud(target_full);
    voxelgrid.filter(*filtered);
    target = filtered;

    filtered = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    voxelgrid.setInputCloud(source_full);
    voxelgrid.filter(*filtered);
    source = filtered;

    return true;
  }

  void generateSyntheticData() {
    // Generate synthetic point clouds for testing when real data is not available
    auto target_cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    auto source_cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();

    // Create a synthetic scene
    for (int i = 0; i < 1000; ++i) {
      pcl::PointXYZ pt;
      pt.x = (i % 10) * 0.1f;
      pt.y = ((i / 10) % 10) * 0.1f;
      pt.z = (i / 100) * 0.1f;
      target_cloud->push_back(pt);
    }

    // Create source by applying a small transformation
    relative_pose = Eigen::Matrix4f::Identity();
    relative_pose(0, 3) = 0.1f;   // 10cm translation in x
    relative_pose(1, 3) = 0.05f;  // 5cm translation in y
    relative_pose(2, 3) = 0.02f;  // 2cm translation in z

    for (const auto& pt : *target_cloud) {
      pcl::PointXYZ new_pt;
      Eigen::Vector4f point(pt.x, pt.y, pt.z, 1.0f);
      Eigen::Vector4f transformed = relative_pose * point;
      new_pt.x = transformed[0];
      new_pt.y = transformed[1];
      new_pt.z = transformed[2];
      source_cloud->push_back(new_pt);
    }

    target_cloud->width = target_cloud->size();
    target_cloud->height = 1;
    target_cloud->is_dense = true;
    source_cloud->width = source_cloud->size();
    source_cloud->height = 1;
    source_cloud->is_dense = true;

    // Assign to const members
    target = target_cloud;
    source = source_cloud;
  }

  Eigen::Vector2f pose_error(const Eigen::Matrix4f estimated) const {
    Eigen::Matrix4f delta = relative_pose.inverse() * estimated;
    double t_error = delta.block<3, 1>(0, 3).norm();
    double r_error = Eigen::AngleAxisf(delta.block<3, 3>(0, 0)).angle();
    return Eigen::Vector2f(t_error, r_error);
  }

  static std::string data_directory;

  PointCloudConstPtr target;
  PointCloudConstPtr source;
  Eigen::Matrix4f relative_pose;
  bool has_real_data_;
};

std::string RealDataTest::data_directory;

TEST_F(RealDataTest, DataAvailabilityCheck) {
  EXPECT_NE(target, nullptr);
  EXPECT_NE(source, nullptr);
  EXPECT_FALSE(target->empty());
  EXPECT_FALSE(source->empty());

  if (has_real_data_) {
    std::cout << "Using real point cloud data with " << target->size() << " target points and " << source->size() << " source points" << std::endl;
  } else {
    std::cout << "Using synthetic point cloud data with " << target->size() << " target points and " << source->size() << " source points" << std::endl;
  }
}

#ifdef USE_VGICP_CUDA

TEST_F(RealDataTest, GPUBruteForceWithRealData) {
  // Test GPU brute force with real point cloud data
  auto vgicp = pcl::make_shared<fast_gicp::FastVGICPCuda<pcl::PointXYZ, pcl::PointXYZ>>();
  vgicp->setNearestNeighborSearchMethod(fast_gicp::NearestNeighborMethod::GPU_BRUTEFORCE);
  vgicp->setInputTarget(target);
  vgicp->setInputSource(source);

  auto aligned = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  vgicp->align(*aligned);

  EXPECT_TRUE(vgicp->hasConverged()) << "GPU BruteForce failed to converge with real data";

  // Check pose accuracy with more realistic tolerances
  Eigen::Vector2f errors = pose_error(vgicp->getFinalTransformation());
  const double t_tol = has_real_data_ ? 0.1 : 0.05;  // Real data is noisier, synthetic needs more tolerance too
  const double r_tol = has_real_data_ ? (2.0 * M_PI / 180.0) : (1.0 * M_PI / 180.0);

  EXPECT_LT(errors[0], t_tol) << "Translation error too large: " << errors[0];
  EXPECT_LT(errors[1], r_tol) << "Rotation error too large: " << errors[1];
}

TEST_F(RealDataTest, CompareAllMethodsWithRealData) {
  // Compare all neighbor search methods with real data
  const double t_tol = has_real_data_ ? 0.15 : 0.1;  // More tolerance for real data and synthetic
  const double r_tol = has_real_data_ ? (3.0 * M_PI / 180.0) : (2.0 * M_PI / 180.0);

  struct TestResult {
    fast_gicp::NearestNeighborMethod method;
    std::string method_name;
    Eigen::Matrix4f transform;
    bool converged;
    Eigen::Vector2f error;
  };

  std::vector<TestResult> results;

  // Test CPU KDTree
  {
    auto vgicp = pcl::make_shared<fast_gicp::FastVGICPCuda<pcl::PointXYZ, pcl::PointXYZ>>();
    vgicp->setNearestNeighborSearchMethod(fast_gicp::NearestNeighborMethod::CPU_PARALLEL_KDTREE);
    vgicp->setInputTarget(target);
    vgicp->setInputSource(source);

    auto aligned = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    vgicp->align(*aligned);

    TestResult result;
    result.method = fast_gicp::NearestNeighborMethod::CPU_PARALLEL_KDTREE;
    result.method_name = fast_gicp::neighborMethodToString(result.method);
    result.transform = vgicp->getFinalTransformation();
    result.converged = vgicp->hasConverged();
    result.error = pose_error(result.transform);
    results.push_back(result);
  }

  // Test GPU Brute Force
  {
    auto vgicp = pcl::make_shared<fast_gicp::FastVGICPCuda<pcl::PointXYZ, pcl::PointXYZ>>();
    vgicp->setNearestNeighborSearchMethod(fast_gicp::NearestNeighborMethod::GPU_BRUTEFORCE);
    vgicp->setInputTarget(target);
    vgicp->setInputSource(source);

    auto aligned = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    vgicp->align(*aligned);

    TestResult result;
    result.method = fast_gicp::NearestNeighborMethod::GPU_BRUTEFORCE;
    result.method_name = fast_gicp::neighborMethodToString(result.method);
    result.transform = vgicp->getFinalTransformation();
    result.converged = vgicp->hasConverged();
    result.error = pose_error(result.transform);
    results.push_back(result);
  }

  // Test GPU RBF Kernel
  {
    auto vgicp = pcl::make_shared<fast_gicp::FastVGICPCuda<pcl::PointXYZ, pcl::PointXYZ>>();
    vgicp->setNearestNeighborSearchMethod(fast_gicp::NearestNeighborMethod::GPU_RBF_KERNEL);
    vgicp->setInputTarget(target);
    vgicp->setInputSource(source);

    auto aligned = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    vgicp->align(*aligned);

    TestResult result;
    result.method = fast_gicp::NearestNeighborMethod::GPU_RBF_KERNEL;
    result.method_name = fast_gicp::neighborMethodToString(result.method);
    result.transform = vgicp->getFinalTransformation();
    result.converged = vgicp->hasConverged();
    result.error = pose_error(result.transform);
    results.push_back(result);
  }

  // Verify all methods
  for (const auto& result : results) {
    EXPECT_TRUE(result.converged) << result.method_name << " failed to converge";
    EXPECT_LT(result.error[0], t_tol) << result.method_name << " translation error: " << result.error[0];
    EXPECT_LT(result.error[1], r_tol) << result.method_name << " rotation error: " << result.error[1];

    std::cout << result.method_name << " - Translation error: " << result.error[0] << ", Rotation error: " << result.error[1] << " rad" << std::endl;
  }

  // Compare GPU BruteForce vs CPU KDTree consistency
  if (results.size() >= 2) {
    Eigen::Matrix4f diff = results[0].transform.inverse() * results[1].transform;
    float translation_diff = diff.block<3, 1>(0, 3).norm();
    float rotation_diff = Eigen::AngleAxisf(diff.block<3, 3>(0, 0)).angle();

    EXPECT_LT(translation_diff, 0.05f) << "CPU vs GPU translation difference too large: " << translation_diff;
    EXPECT_LT(rotation_diff, 0.2f) << "CPU vs GPU rotation difference too large: " << rotation_diff;
  }
}

TEST_F(RealDataTest, KNNDirectTestWithRealData) {
  // Test the KNN function directly with real point cloud data
  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> target_points;
  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> source_points;

  // Convert PCL points to Eigen vectors
  for (const auto& pt : *target) {
    target_points.emplace_back(pt.x, pt.y, pt.z);
  }
  for (const auto& pt : *source) {
    source_points.emplace_back(pt.x, pt.y, pt.z);
  }

  // Test with subset for performance
  size_t max_points = 1000;
  if (target_points.size() > max_points) {
    target_points.resize(max_points);
  }
  if (source_points.size() > max_points) {
    source_points.resize(max_points);
  }

  thrust::host_vector<Eigen::Vector3f> h_target(target_points.begin(), target_points.end());
  thrust::host_vector<Eigen::Vector3f> h_source(source_points.begin(), source_points.end());
  thrust::device_vector<Eigen::Vector3f> d_target = h_target;
  thrust::device_vector<Eigen::Vector3f> d_source = h_source;

  // Test different k values
  std::vector<int> k_values = {1, 5, 10, 20};

  for (int k : k_values) {
    thrust::device_vector<thrust::pair<float, int>> d_results;

    EXPECT_NO_THROW({ fast_gicp::cuda::brute_force_knn_search(d_source, d_target, k, d_results, false); }) << "KNN search failed for k=" << k << " with real data";

    // Verify result size
    EXPECT_EQ(d_results.size(), source_points.size() * std::min(k, static_cast<int>(target_points.size())));

    // Spot check some results
    thrust::host_vector<thrust::pair<float, int>> h_results = d_results;

    // Check first 10 results for validity
    for (size_t i = 0; i < std::min(static_cast<size_t>(10 * k), h_results.size()); ++i) {
      EXPECT_GE(h_results[i].first, 0.0f) << "Invalid distance at index " << i;
      EXPECT_GE(h_results[i].second, 0) << "Invalid target index at index " << i;
      EXPECT_LT(h_results[i].second, target_points.size()) << "Target index out of bounds at index " << i;
    }
  }
}

TEST_F(RealDataTest, PerformanceComparisonWithRealData) {
  // Compare performance between different methods with real data
  auto start_time = std::chrono::high_resolution_clock::now();

  // GPU BruteForce timing
  auto vgicp_gpu = pcl::make_shared<fast_gicp::FastVGICPCuda<pcl::PointXYZ, pcl::PointXYZ>>();
  vgicp_gpu->setNearestNeighborSearchMethod(fast_gicp::NearestNeighborMethod::GPU_BRUTEFORCE);
  vgicp_gpu->setInputTarget(target);
  vgicp_gpu->setInputSource(source);

  auto gpu_start = std::chrono::high_resolution_clock::now();
  auto aligned_gpu = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  vgicp_gpu->align(*aligned_gpu);
  auto gpu_end = std::chrono::high_resolution_clock::now();

  // CPU KDTree timing
  auto vgicp_cpu = pcl::make_shared<fast_gicp::FastVGICPCuda<pcl::PointXYZ, pcl::PointXYZ>>();
  vgicp_cpu->setNearestNeighborSearchMethod(fast_gicp::NearestNeighborMethod::CPU_PARALLEL_KDTREE);
  vgicp_cpu->setInputTarget(target);
  vgicp_cpu->setInputSource(source);

  auto cpu_start = std::chrono::high_resolution_clock::now();
  auto aligned_cpu = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  vgicp_cpu->align(*aligned_cpu);
  auto cpu_end = std::chrono::high_resolution_clock::now();

  auto gpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(gpu_end - gpu_start).count();
  auto cpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end - cpu_start).count();

  std::cout << "Performance comparison with " << target->size() << " target points:" << std::endl;
  std::cout << "  GPU BruteForce: " << gpu_duration << " ms" << std::endl;
  std::cout << "  CPU KDTree:     " << cpu_duration << " ms" << std::endl;

  // Both should converge
  EXPECT_TRUE(vgicp_gpu->hasConverged()) << "GPU registration failed to converge";
  EXPECT_TRUE(vgicp_cpu->hasConverged()) << "CPU registration failed to converge";

  // Results should be similar
  Eigen::Vector2f gpu_error = pose_error(vgicp_gpu->getFinalTransformation());
  Eigen::Vector2f cpu_error = pose_error(vgicp_cpu->getFinalTransformation());

  const double t_tol = has_real_data_ ? 0.15 : 0.1;
  const double r_tol = has_real_data_ ? (3.0 * M_PI / 180.0) : (2.0 * M_PI / 180.0);

  EXPECT_LT(gpu_error[0], t_tol) << "GPU translation error: " << gpu_error[0];
  EXPECT_LT(gpu_error[1], r_tol) << "GPU rotation error: " << gpu_error[1];
  EXPECT_LT(cpu_error[0], t_tol) << "CPU translation error: " << cpu_error[0];
  EXPECT_LT(cpu_error[1], r_tol) << "CPU rotation error: " << cpu_error[1];
}

#else

TEST_F(RealDataTest, CUDANotAvailable) {
  GTEST_SKIP() << "CUDA not available - skipping GPU real data tests";
}

#endif

int main(int argc, char** argv) {
  if (argc > 1) {
    RealDataTest::data_directory = argv[1];
  }
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
