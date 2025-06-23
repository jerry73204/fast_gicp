/**
 * @file enum_validation_test.cpp
 * @brief Tests for the new enum-based neighbor method configuration
 *
 * This file validates the enum-based design for NearestNeighborMethod
 * and ensures type safety and proper validation.
 */

#include <gtest/gtest.h>
#include <sstream>

#ifdef USE_VGICP_CUDA
#include <fast_gicp/gicp/fast_vgicp_cuda.hpp>
#endif

class EnumValidationTest : public ::testing::Test {
protected:
  void SetUp() override {}
};

TEST_F(EnumValidationTest, EnumToStringConversion) {
  // Test all enum values convert to proper strings
  EXPECT_STREQ(fast_gicp::neighborMethodToString(fast_gicp::NearestNeighborMethod::CPU_PARALLEL_KDTREE), "CPU_PARALLEL_KDTREE");
  EXPECT_STREQ(fast_gicp::neighborMethodToString(fast_gicp::NearestNeighborMethod::GPU_BRUTEFORCE), "GPU_BRUTEFORCE");
  EXPECT_STREQ(fast_gicp::neighborMethodToString(fast_gicp::NearestNeighborMethod::GPU_RBF_KERNEL), "GPU_RBF_KERNEL");

  // Test invalid enum value
  auto invalid_method = static_cast<fast_gicp::NearestNeighborMethod>(999);
  EXPECT_STREQ(fast_gicp::neighborMethodToString(invalid_method), "UNKNOWN");
}

TEST_F(EnumValidationTest, StringToEnumConversion) {
  // Test valid string conversions
  EXPECT_EQ(fast_gicp::stringToNeighborMethod("GPU_BRUTEFORCE"), fast_gicp::NearestNeighborMethod::GPU_BRUTEFORCE);
  EXPECT_EQ(fast_gicp::stringToNeighborMethod("GPU_RBF_KERNEL"), fast_gicp::NearestNeighborMethod::GPU_RBF_KERNEL);

  // Test backward compatibility
  EXPECT_EQ(fast_gicp::stringToNeighborMethod("CPU_KDTREE"), fast_gicp::NearestNeighborMethod::CPU_PARALLEL_KDTREE);

  // Test default fallback
  EXPECT_EQ(fast_gicp::stringToNeighborMethod("INVALID_METHOD"), fast_gicp::NearestNeighborMethod::CPU_PARALLEL_KDTREE);
  EXPECT_EQ(fast_gicp::stringToNeighborMethod(""), fast_gicp::NearestNeighborMethod::CPU_PARALLEL_KDTREE);
}

TEST_F(EnumValidationTest, EnumValidation) {
  // Test valid enum values
  EXPECT_TRUE(fast_gicp::isValidNeighborMethod(fast_gicp::NearestNeighborMethod::CPU_PARALLEL_KDTREE));
  EXPECT_TRUE(fast_gicp::isValidNeighborMethod(fast_gicp::NearestNeighborMethod::GPU_BRUTEFORCE));
  EXPECT_TRUE(fast_gicp::isValidNeighborMethod(fast_gicp::NearestNeighborMethod::GPU_RBF_KERNEL));

  // Test invalid enum values
  auto invalid_low = static_cast<fast_gicp::NearestNeighborMethod>(-1);
  auto invalid_high = static_cast<fast_gicp::NearestNeighborMethod>(3);
  auto invalid_very_high = static_cast<fast_gicp::NearestNeighborMethod>(999);

  EXPECT_FALSE(fast_gicp::isValidNeighborMethod(invalid_low));
  EXPECT_FALSE(fast_gicp::isValidNeighborMethod(invalid_high));
  EXPECT_FALSE(fast_gicp::isValidNeighborMethod(invalid_very_high));
}

TEST_F(EnumValidationTest, EnumValueConsistency) {
  // Test that enum values are what we expect (important for serialization)
  EXPECT_EQ(static_cast<int>(fast_gicp::NearestNeighborMethod::CPU_PARALLEL_KDTREE), 0);
  EXPECT_EQ(static_cast<int>(fast_gicp::NearestNeighborMethod::GPU_BRUTEFORCE), 1);
  EXPECT_EQ(static_cast<int>(fast_gicp::NearestNeighborMethod::GPU_RBF_KERNEL), 2);
}

TEST_F(EnumValidationTest, RoundTripConversion) {
  // Test that enum -> string -> enum conversion works
  std::vector<fast_gicp::NearestNeighborMethod> methods = {
    fast_gicp::NearestNeighborMethod::CPU_PARALLEL_KDTREE,
    fast_gicp::NearestNeighborMethod::GPU_BRUTEFORCE,
    fast_gicp::NearestNeighborMethod::GPU_RBF_KERNEL};

  for (auto method : methods) {
    const char* str = fast_gicp::neighborMethodToString(method);
    auto converted_back = fast_gicp::stringToNeighborMethod(str);
    EXPECT_EQ(method, converted_back) << "Round-trip conversion failed for " << str;
  }
}

#ifdef USE_VGICP_CUDA

TEST_F(EnumValidationTest, SetMethodValidation) {
  // Test that the setNearestNeighborSearchMethod properly validates inputs
  auto vgicp = std::make_shared<fast_gicp::FastVGICPCuda<pcl::PointXYZ, pcl::PointXYZ>>();

  // Valid methods should work without error
  EXPECT_NO_THROW(vgicp->setNearestNeighborSearchMethod(fast_gicp::NearestNeighborMethod::CPU_PARALLEL_KDTREE));
  EXPECT_NO_THROW(vgicp->setNearestNeighborSearchMethod(fast_gicp::NearestNeighborMethod::GPU_BRUTEFORCE));
  EXPECT_NO_THROW(vgicp->setNearestNeighborSearchMethod(fast_gicp::NearestNeighborMethod::GPU_RBF_KERNEL));

  // Invalid method should not throw but should fallback to default
  auto invalid_method = static_cast<fast_gicp::NearestNeighborMethod>(999);
  EXPECT_NO_THROW(vgicp->setNearestNeighborSearchMethod(invalid_method));
}

#endif

TEST_F(EnumValidationTest, GTestParameterNaming) {
  // Test that the parameter naming function works correctly
  // This simulates what happens in the parameterized tests

  struct MockParamInfo {
    std::tuple<const char*, bool, fast_gicp::NearestNeighborMethod> param;
  };

  auto naming_func = [](const MockParamInfo& info) {
    std::stringstream sst;
    sst << std::get<0>(info.param) << (std::get<1>(info.param) ? "_MT" : "_ST") << "_" << fast_gicp::neighborMethodToString(std::get<2>(info.param));
    return sst.str();
  };

  MockParamInfo test_info;
  test_info.param = std::make_tuple("VGICP_CUDA", true, fast_gicp::NearestNeighborMethod::GPU_BRUTEFORCE);

  std::string result = naming_func(test_info);
  EXPECT_EQ(result, "VGICP_CUDA_MT_GPU_BRUTEFORCE");

  test_info.param = std::make_tuple("VGICP_CUDA", false, fast_gicp::NearestNeighborMethod::CPU_PARALLEL_KDTREE);
  result = naming_func(test_info);
  EXPECT_EQ(result, "VGICP_CUDA_ST_CPU_PARALLEL_KDTREE");
}

TEST_F(EnumValidationTest, EnumSizeAndLimits) {
  // Ensure we haven't accidentally changed the number of enum values
  // This test will fail if someone adds/removes enum values without updating tests

  int valid_count = 0;
  for (int i = 0; i < 10; ++i) {  // Check first 10 values
    auto method = static_cast<fast_gicp::NearestNeighborMethod>(i);
    if (fast_gicp::isValidNeighborMethod(method)) {
      valid_count++;
    }
  }

  EXPECT_EQ(valid_count, 3) << "Expected exactly 3 valid neighbor search methods";
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
