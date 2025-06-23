/**
 * @file integration_test.cpp
 * @brief Integration tests for the Pure Thrust/CUB k-nearest neighbor implementation
 *
 * This file provides comprehensive integration testing for the Phase 4 CUDA 12.x
 * modernization, focusing on end-to-end VGICP registration with GPU KNN.
 */

#include <gtest/gtest.h>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include <fstream>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/gicp.h>

#include <fast_gicp/gicp/fast_vgicp.hpp>
#ifdef USE_VGICP_CUDA
#include <fast_gicp/gicp/fast_vgicp_cuda.hpp>
#include <fast_gicp/cuda/brute_force_knn.cuh>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#endif

class IntegrationTest : public ::testing::Test {
protected:
  using PointT = pcl::PointXYZ;
  using PointCloud = pcl::PointCloud<PointT>;
  using PointCloudPtr = PointCloud::Ptr;
  using PointCloudConstPtr = PointCloud::ConstPtr;

  void SetUp() override {
    // Initialize random number generator with fixed seed
    rng_.seed(42);

    // Set tolerance values - more realistic for synthetic data
    translation_tolerance_ = 0.15;             // 15cm for synthetic data
    rotation_tolerance_ = 3.0 * M_PI / 180.0;  // 3 degrees for synthetic data
    convergence_tolerance_ = 1e-4;
  }

  // Generate a synthetic point cloud (bunny-like shape)
  PointCloudPtr generateTestPointCloud(int num_points, float scale = 1.0f) {
    auto cloud = std::make_shared<PointCloud>();
    cloud->reserve(num_points);

    std::uniform_real_distribution<float> uniform_dist(-1.0f, 1.0f);
    std::normal_distribution<float> normal_dist(0.0f, 0.3f);

    for (int i = 0; i < num_points; ++i) {
      PointT point;

      // Create a bunny-like distribution
      float theta = uniform_dist(rng_) * M_PI;
      float phi = uniform_dist(rng_) * 2.0f * M_PI;
      float r = 0.5f + std::abs(normal_dist(rng_));

      point.x = scale * r * std::sin(theta) * std::cos(phi);
      point.y = scale * r * std::sin(theta) * std::sin(phi);
      point.z = scale * r * std::cos(theta);

      cloud->push_back(point);
    }

    cloud->width = cloud->size();
    cloud->height = 1;
    cloud->is_dense = true;

    return cloud;
  }

  // Apply a known transformation to create source cloud
  PointCloudPtr applyTransformation(PointCloudConstPtr target, const Eigen::Matrix4f& transform) {
    auto source = std::make_shared<PointCloud>();
    pcl::transformPointCloud(*target, *source, transform);
    return source;
  }

  // Create a known transformation matrix
  Eigen::Matrix4f createTestTransform(float tx = 0.1f, float ty = 0.2f, float tz = 0.3f, float rx = 0.1f, float ry = 0.2f, float rz = 0.3f) {
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();

    // Translation
    transform(0, 3) = tx;
    transform(1, 3) = ty;
    transform(2, 3) = tz;

    // Rotation
    Eigen::AngleAxisf rotation_x(rx, Eigen::Vector3f::UnitX());
    Eigen::AngleAxisf rotation_y(ry, Eigen::Vector3f::UnitY());
    Eigen::AngleAxisf rotation_z(rz, Eigen::Vector3f::UnitZ());

    Eigen::Matrix3f rotation_matrix = (rotation_z * rotation_y * rotation_x).matrix();
    transform.block<3, 3>(0, 0) = rotation_matrix;

    return transform;
  }

  // Calculate pose error between estimated and ground truth transforms
  Eigen::Vector2f calculatePoseError(const Eigen::Matrix4f& estimated, const Eigen::Matrix4f& ground_truth) {
    Eigen::Matrix4f delta = ground_truth.inverse() * estimated;

    float translation_error = delta.block<3, 1>(0, 3).norm();

    Eigen::AngleAxisf angle_axis(delta.block<3, 3>(0, 0));
    float rotation_error = std::abs(angle_axis.angle());

    return Eigen::Vector2f(translation_error, rotation_error);
  }

  // Test data
  std::mt19937 rng_;
  float translation_tolerance_;
  float rotation_tolerance_;
  float convergence_tolerance_;
};

#ifdef USE_VGICP_CUDA

TEST_F(IntegrationTest, GPUBruteForceVsCPUKDTree_SameResults) {
  // Test that GPU brute force and CPU KDTree produce similar registration results
  auto target = generateTestPointCloud(1000, 2.0f);
  auto ground_truth_transform = createTestTransform();
  auto source = applyTransformation(target, ground_truth_transform);

  // Test with CPU KDTree
  auto vgicp_cpu = std::make_shared<fast_gicp::FastVGICPCuda<PointT, PointT>>();
  vgicp_cpu->setNearestNeighborSearchMethod(fast_gicp::NearestNeighborMethod::CPU_PARALLEL_KDTREE);
  vgicp_cpu->setInputTarget(target);
  vgicp_cpu->setInputSource(source);

  auto aligned_cpu = std::make_shared<PointCloud>();
  vgicp_cpu->align(*aligned_cpu);

  // Test with GPU Brute Force
  auto vgicp_gpu = std::make_shared<fast_gicp::FastVGICPCuda<PointT, PointT>>();
  vgicp_gpu->setNearestNeighborSearchMethod(fast_gicp::NearestNeighborMethod::GPU_BRUTEFORCE);
  vgicp_gpu->setInputTarget(target);
  vgicp_gpu->setInputSource(source);

  auto aligned_gpu = std::make_shared<PointCloud>();
  vgicp_gpu->align(*aligned_gpu);

  // Both should converge
  EXPECT_TRUE(vgicp_cpu->hasConverged()) << "CPU KDTree registration failed to converge";
  EXPECT_TRUE(vgicp_gpu->hasConverged()) << "GPU BruteForce registration failed to converge";

  // Both should achieve good accuracy
  auto cpu_error = calculatePoseError(vgicp_cpu->getFinalTransformation(), ground_truth_transform);
  auto gpu_error = calculatePoseError(vgicp_gpu->getFinalTransformation(), ground_truth_transform);

  EXPECT_LT(cpu_error[0], translation_tolerance_) << "CPU translation error too large: " << cpu_error[0];
  EXPECT_LT(cpu_error[1], rotation_tolerance_) << "CPU rotation error too large: " << cpu_error[1];
  EXPECT_LT(gpu_error[0], translation_tolerance_) << "GPU translation error too large: " << gpu_error[0];
  EXPECT_LT(gpu_error[1], rotation_tolerance_) << "GPU rotation error too large: " << gpu_error[1];

  // Results should be similar between CPU and GPU
  auto diff_matrix = vgicp_cpu->getFinalTransformation().inverse() * vgicp_gpu->getFinalTransformation();
  float translation_diff = diff_matrix.block<3, 1>(0, 3).norm();
  float rotation_diff = Eigen::AngleAxisf(diff_matrix.block<3, 3>(0, 0)).angle();

  EXPECT_LT(translation_diff, 0.02f) << "Translation difference between CPU and GPU too large: " << translation_diff;
  EXPECT_LT(rotation_diff, 0.1f) << "Rotation difference between CPU and GPU too large: " << rotation_diff;
}

TEST_F(IntegrationTest, GPUBruteForceVsRBFKernel_Consistency) {
  // Test consistency between GPU brute force and GPU RBF kernel methods
  auto target = generateTestPointCloud(800, 1.5f);
  auto ground_truth_transform = createTestTransform(0.05f, 0.1f, 0.15f, 0.05f, 0.1f, 0.05f);
  auto source = applyTransformation(target, ground_truth_transform);

  // Test with GPU Brute Force
  auto vgicp_bruteforce = std::make_shared<fast_gicp::FastVGICPCuda<PointT, PointT>>();
  vgicp_bruteforce->setNearestNeighborSearchMethod(fast_gicp::NearestNeighborMethod::GPU_BRUTEFORCE);
  vgicp_bruteforce->setInputTarget(target);
  vgicp_bruteforce->setInputSource(source);

  auto aligned_bruteforce = std::make_shared<PointCloud>();
  vgicp_bruteforce->align(*aligned_bruteforce);

  // Test with GPU RBF Kernel
  auto vgicp_rbf = std::make_shared<fast_gicp::FastVGICPCuda<PointT, PointT>>();
  vgicp_rbf->setNearestNeighborSearchMethod(fast_gicp::NearestNeighborMethod::GPU_RBF_KERNEL);
  vgicp_rbf->setInputTarget(target);
  vgicp_rbf->setInputSource(source);

  auto aligned_rbf = std::make_shared<PointCloud>();
  vgicp_rbf->align(*aligned_rbf);

  // Both should converge
  EXPECT_TRUE(vgicp_bruteforce->hasConverged()) << "GPU BruteForce registration failed to converge";
  EXPECT_TRUE(vgicp_rbf->hasConverged()) << "GPU RBF registration failed to converge";

  // Both should achieve reasonable accuracy
  auto bruteforce_error = calculatePoseError(vgicp_bruteforce->getFinalTransformation(), ground_truth_transform);
  auto rbf_error = calculatePoseError(vgicp_rbf->getFinalTransformation(), ground_truth_transform);

  EXPECT_LT(bruteforce_error[0], translation_tolerance_) << "BruteForce translation error: " << bruteforce_error[0];
  EXPECT_LT(bruteforce_error[1], rotation_tolerance_) << "BruteForce rotation error: " << bruteforce_error[1];
  EXPECT_LT(rbf_error[0], translation_tolerance_) << "RBF translation error: " << rbf_error[0];
  EXPECT_LT(rbf_error[1], rotation_tolerance_) << "RBF rotation error: " << rbf_error[1];
}

TEST_F(IntegrationTest, MethodSwitchingDuringRegistration) {
  // Test switching neighbor search methods during registration
  auto target = generateTestPointCloud(500);
  auto ground_truth_transform = createTestTransform();
  auto source = applyTransformation(target, ground_truth_transform);

  auto vgicp = std::make_shared<fast_gicp::FastVGICPCuda<PointT, PointT>>();

  // Start with CPU method
  vgicp->setNearestNeighborSearchMethod(fast_gicp::NearestNeighborMethod::CPU_PARALLEL_KDTREE);
  vgicp->setInputTarget(target);
  vgicp->setInputSource(source);

  // Perform initial alignment
  auto aligned1 = std::make_shared<PointCloud>();
  vgicp->align(*aligned1);
  EXPECT_TRUE(vgicp->hasConverged()) << "Initial CPU registration failed";

  // Switch to GPU brute force
  vgicp->setNearestNeighborSearchMethod(fast_gicp::NearestNeighborMethod::GPU_BRUTEFORCE);

  // Use previous result as initial guess for refinement
  auto aligned2 = std::make_shared<PointCloud>();
  vgicp->align(*aligned2, vgicp->getFinalTransformation());
  EXPECT_TRUE(vgicp->hasConverged()) << "GPU refinement failed";

  // Final result should still be accurate
  auto final_error = calculatePoseError(vgicp->getFinalTransformation(), ground_truth_transform);
  EXPECT_LT(final_error[0], translation_tolerance_) << "Final translation error: " << final_error[0];
  EXPECT_LT(final_error[1], rotation_tolerance_) << "Final rotation error: " << final_error[1];
}

TEST_F(IntegrationTest, LargePointCloudStressTest) {
  // Test with larger point clouds to stress the GPU implementation
  auto target = generateTestPointCloud(5000, 3.0f);
  auto ground_truth_transform = createTestTransform();
  auto source = applyTransformation(target, ground_truth_transform);

  auto vgicp = std::make_shared<fast_gicp::FastVGICPCuda<PointT, PointT>>();
  vgicp->setNearestNeighborSearchMethod(fast_gicp::NearestNeighborMethod::GPU_BRUTEFORCE);
  vgicp->setInputTarget(target);
  vgicp->setInputSource(source);

  // Time the registration
  auto start = std::chrono::high_resolution_clock::now();

  auto aligned = std::make_shared<PointCloud>();
  vgicp->align(*aligned);

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

  EXPECT_TRUE(vgicp->hasConverged()) << "Large point cloud registration failed to converge";

  auto error = calculatePoseError(vgicp->getFinalTransformation(), ground_truth_transform);
  EXPECT_LT(error[0], translation_tolerance_) << "Large cloud translation error: " << error[0];
  EXPECT_LT(error[1], rotation_tolerance_) << "Large cloud rotation error: " << error[1];

  // Log performance (no hard requirement, just for monitoring)
  std::cout << "Large point cloud (" << target->size() << " points) registration took: " << duration << " ms" << std::endl;
}

TEST_F(IntegrationTest, ErrorHandlingEdgeCases) {
  // Test error handling with edge cases
  auto target = generateTestPointCloud(100);
  auto source = generateTestPointCloud(100);

  auto vgicp = std::make_shared<fast_gicp::FastVGICPCuda<PointT, PointT>>();
  vgicp->setNearestNeighborSearchMethod(fast_gicp::NearestNeighborMethod::GPU_BRUTEFORCE);

  // Test with empty target
  auto empty_cloud = std::make_shared<PointCloud>();
  vgicp->setInputTarget(empty_cloud);
  vgicp->setInputSource(source);

  auto aligned = std::make_shared<PointCloud>();
  vgicp->align(*aligned);

  // Should handle gracefully (may not converge, but shouldn't crash)
  EXPECT_NO_THROW(vgicp->hasConverged());

  // Test with empty source
  vgicp->setInputTarget(target);
  vgicp->setInputSource(empty_cloud);
  vgicp->align(*aligned);
  EXPECT_NO_THROW(vgicp->hasConverged());

  // Test with very small clouds
  auto tiny_target = std::make_shared<PointCloud>();
  auto tiny_source = std::make_shared<PointCloud>();

  for (int i = 0; i < 3; ++i) {
    PointT pt;
    pt.x = i;
    pt.y = i;
    pt.z = i;
    tiny_target->push_back(pt);
    tiny_source->push_back(pt);
  }

  vgicp->setInputTarget(tiny_target);
  vgicp->setInputSource(tiny_source);
  vgicp->align(*aligned);
  EXPECT_NO_THROW(vgicp->hasConverged());
}

TEST_F(IntegrationTest, VoxelResolutionEffects) {
  // Test how different voxel resolutions affect GPU brute force registration
  auto target = generateTestPointCloud(1000, 2.0f);
  auto ground_truth_transform = createTestTransform();
  auto source = applyTransformation(target, ground_truth_transform);

  std::vector<double> resolutions = {0.1, 0.5, 1.0, 2.0};

  for (double resolution : resolutions) {
    auto vgicp = std::make_shared<fast_gicp::FastVGICPCuda<PointT, PointT>>();
    vgicp->setNearestNeighborSearchMethod(fast_gicp::NearestNeighborMethod::GPU_BRUTEFORCE);
    vgicp->setResolution(resolution);
    vgicp->setInputTarget(target);
    vgicp->setInputSource(source);

    auto aligned = std::make_shared<PointCloud>();
    vgicp->align(*aligned);

    // Should converge for reasonable resolutions
    if (resolution <= 1.0) {
      EXPECT_TRUE(vgicp->hasConverged()) << "Failed to converge with resolution: " << resolution;

      auto error = calculatePoseError(vgicp->getFinalTransformation(), ground_truth_transform);
      EXPECT_LT(error[0], translation_tolerance_ * 2) << "Resolution " << resolution << " translation error: " << error[0];
      EXPECT_LT(error[1], rotation_tolerance_ * 2) << "Resolution " << resolution << " rotation error: " << error[1];
    }
  }
}

TEST_F(IntegrationTest, DirectKNNFunctionTesting) {
  // Test the brute_force_knn_search function directly
  auto points = generateTestPointCloud(500);

  // Convert to Thrust vectors
  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> host_points;
  for (const auto& pt : *points) {
    host_points.emplace_back(pt.x, pt.y, pt.z);
  }

  thrust::host_vector<Eigen::Vector3f> h_points(host_points.begin(), host_points.end());
  thrust::device_vector<Eigen::Vector3f> d_source = h_points;
  thrust::device_vector<Eigen::Vector3f> d_target = h_points;

  // Test various k values
  std::vector<int> k_values = {1, 5, 10, 20, 50};

  for (int k : k_values) {
    thrust::device_vector<thrust::pair<float, int>> d_results;

    // Should not throw exceptions
    EXPECT_NO_THROW({ fast_gicp::cuda::brute_force_knn_search(d_source, d_target, k, d_results, false); }) << "KNN search failed for k=" << k;

    // Verify result size
    EXPECT_EQ(d_results.size(), points->size() * std::min(k, static_cast<int>(points->size())));

    // Copy results and verify they're valid
    thrust::host_vector<thrust::pair<float, int>> h_results = d_results;

    for (size_t i = 0; i < h_results.size(); ++i) {
      EXPECT_GE(h_results[i].first, 0.0f) << "Invalid distance at index " << i;
      EXPECT_GE(h_results[i].second, 0) << "Invalid index at index " << i;
      EXPECT_LT(h_results[i].second, points->size()) << "Index out of bounds at index " << i;
    }
  }
}

#else

TEST_F(IntegrationTest, CUDANotAvailable) {
  GTEST_SKIP() << "CUDA not available - skipping GPU integration tests";
}

#endif

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
