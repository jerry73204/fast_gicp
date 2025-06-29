cmake_minimum_required(VERSION 3.18.0)  # Required for CUDA 12.x support
project(fast_gicp LANGUAGES C CXX)  # Enable C language for MPI compatibility

option(BUILD_VGICP_CUDA "Build GPU-powered VGICP" OFF)
option(BUILD_apps "Build application programs" ON)
option(BUILD_test "Build test programs" OFF)
option(BUILD_PYTHON_BINDINGS "Build python bindings" OFF)

set(CMAKE_BUILD_TYPE "Release")

# Set C++ standard for CUDA 12.x compatibility
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(MPI QUIET)  # Required for some PCL/VTK configurations
find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)
add_definitions(${PCL_DEFINITIONS})


find_package(OpenMP)
if (OPENMP_FOUND)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

if(BUILD_VGICP_CUDA)
  # Modern CUDA 12.x support
  find_package(CUDAToolkit 12.0 REQUIRED)

  # Set CUDA compiler and host compiler before enabling CUDA language
  set(CMAKE_CUDA_COMPILER ${CUDAToolkit_NVCC_EXECUTABLE})
  set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})

  # Set default architecture before enabling CUDA - only recent GPUs for faster builds
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "75;80;86;87;89;90")  # Turing, Ampere, Ada Lovelace, Hopper
  endif()

  enable_language(CUDA)

  # Set CUDA standards for modern compilation
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS OFF)

  # CUDA compilation flags for 12.x compatibility and fast builds
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DTHRUST_IGNORE_CUB_VERSION_CHECK")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DFAST_GICP_CUDA_12_MODERNIZATION")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")  # Suppress architecture warnings

  message(STATUS "CUDA 12.x enabled with architectures: ${CMAKE_CUDA_ARCHITECTURES}")
  message(STATUS "Using CUDA compiler: ${CMAKE_CUDA_COMPILER}")
  message(STATUS "Using CUDA host compiler: ${CMAKE_CUDA_HOST_COMPILER}")
endif()


###########
## Build ##
###########

add_library(fast_gicp SHARED
  src/fast_gicp/gicp/lsq_registration.cpp
  src/fast_gicp/gicp/fast_gicp.cpp
  src/fast_gicp/gicp/fast_gicp_st.cpp
  src/fast_gicp/gicp/fast_vgicp.cpp
)
target_link_libraries(fast_gicp
  ${PCL_LIBRARIES}
)
if (OPENMP_FOUND)
    if (TARGET OpenMP::OpenMP_CXX)
        target_link_libraries(fast_gicp OpenMP::OpenMP_CXX)
    endif ()
endif ()
target_include_directories(fast_gicp PUBLIC
  include
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

### APPS ###
if(BUILD_apps)
  add_executable(gicp_align src/align.cpp)
  add_dependencies(gicp_align fast_gicp)
  target_link_libraries(gicp_align
    ${PCL_LIBRARIES}
    fast_gicp
  )

  add_executable(gicp_kitti src/kitti.cpp)
  add_dependencies(gicp_kitti fast_gicp)
  target_link_libraries(gicp_kitti
    ${PCL_LIBRARIES}
    fast_gicp
  )
endif()

### Python bindings ###
if(BUILD_PYTHON_BINDINGS)
  find_package(pybind11 CONFIG)
  pybind11_add_module(pygicp
    src/python/main.cpp
  )
  target_include_directories(pygicp PUBLIC
    include
    ${PCL_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
  )
  target_link_libraries(pygicp PRIVATE
    fast_gicp
  )
  set_target_properties(pygicp PROPERTIES
    INSTALL_RPATH "$ORIGIN/"
    BUILD_WITH_INSTALL_RPATH TRUE
  )
endif()

### CUDA ###
if(BUILD_VGICP_CUDA)
  add_definitions(-DUSE_VGICP_CUDA)

  # Modern CUDA library creation (replaces cuda_add_library)
  add_library(fast_vgicp_cuda SHARED
    src/fast_gicp/cuda/fast_vgicp_cuda.cu
    src/fast_gicp/cuda/brute_force_knn.cu  # Pure Thrust/CUB implementation (Phase 4)
    src/fast_gicp/cuda/covariance_estimation.cu
    src/fast_gicp/cuda/covariance_estimation_rbf.cu
    src/fast_gicp/cuda/covariance_regularization.cu
    src/fast_gicp/cuda/gaussian_voxelmap.cu
    src/fast_gicp/cuda/find_voxel_correspondences.cu
    src/fast_gicp/cuda/compute_derivatives.cu
    src/fast_gicp/cuda/compute_mahalanobis.cu
    src/fast_gicp/cuda/ndt_cuda.cu
    src/fast_gicp/cuda/ndt_compute_derivatives.cu
  )

  # Set CUDA language properties for modern CMake
  set_property(TARGET fast_vgicp_cuda PROPERTY CUDA_LANGUAGE_STANDARD 17)
  set_property(TARGET fast_vgicp_cuda PROPERTY CUDA_STANDARD_REQUIRED ON)

  target_include_directories(fast_vgicp_cuda PRIVATE
    include
    thirdparty/Eigen
    # thirdparty/nvbio  # Disabled - nvbio dependency removed for Phase 1
    ${EIGEN3_INCLUDE_DIR}
  )
  target_link_libraries(fast_vgicp_cuda
    CUDA::cublas
    CUDA::curand
    CUDA::cusparse
  )

  # add vgicp_cuda to libfast_gicp
  target_sources(fast_gicp PRIVATE
    src/fast_gicp/gicp/fast_vgicp_cuda.cpp
    src/fast_gicp/ndt/ndt_cuda.cpp
  )
  target_link_libraries(fast_gicp
    fast_vgicp_cuda
  )
  add_dependencies(fast_gicp fast_vgicp_cuda)
endif()

### TEST ###
if(BUILD_test)
  find_package(GTest REQUIRED)

  add_executable(gicp_test src/test/gicp_test.cpp)
  add_dependencies(gicp_test fast_gicp)
  target_link_libraries(gicp_test ${GTEST_LIBRARIES} ${PCL_LIBRARIES} fast_gicp)
  gtest_add_tests(TARGET gicp_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR} EXTRA_ARGS "${CMAKE_SOURCE_DIR}/data")

  # KNN unit tests
  if(BUILD_VGICP_CUDA)
    add_executable(knn_test src/test/knn_test.cu)
    add_dependencies(knn_test fast_gicp)
    target_link_libraries(knn_test ${GTEST_LIBRARIES} fast_gicp fast_vgicp_cuda)
    set_property(TARGET knn_test PROPERTY CUDA_LANGUAGE_STANDARD 17)
    gtest_add_tests(TARGET knn_test)

    # KNN component unit tests
    add_executable(knn_unit_test src/test/knn_unit_test.cu)
    add_dependencies(knn_unit_test fast_gicp)
    target_link_libraries(knn_unit_test ${GTEST_LIBRARIES} fast_gicp fast_vgicp_cuda)
    set_property(TARGET knn_unit_test PROPERTY CUDA_LANGUAGE_STANDARD 17)
    gtest_add_tests(TARGET knn_unit_test)

    # KNN numerical accuracy tests
    add_executable(knn_accuracy_test src/test/knn_accuracy_test.cu)
    add_dependencies(knn_accuracy_test fast_gicp)
    target_link_libraries(knn_accuracy_test ${GTEST_LIBRARIES} fast_gicp fast_vgicp_cuda)
    set_property(TARGET knn_accuracy_test PROPERTY CUDA_LANGUAGE_STANDARD 17)
    gtest_add_tests(TARGET knn_accuracy_test)
  endif()

  # Integration tests
  if(BUILD_VGICP_CUDA)
    add_executable(integration_test src/test/integration_test.cu)
    add_dependencies(integration_test fast_gicp)
    target_link_libraries(integration_test ${GTEST_LIBRARIES} ${PCL_LIBRARIES} fast_gicp fast_vgicp_cuda)
    set_property(TARGET integration_test PROPERTY CUDA_LANGUAGE_STANDARD 17)
    gtest_add_tests(TARGET integration_test)
  endif()

  # Real data tests
  if(BUILD_VGICP_CUDA)
    add_executable(real_data_test src/test/real_data_test.cu)
    add_dependencies(real_data_test fast_gicp)
    target_link_libraries(real_data_test ${GTEST_LIBRARIES} ${PCL_LIBRARIES} fast_gicp fast_vgicp_cuda)
    set_property(TARGET real_data_test PROPERTY CUDA_LANGUAGE_STANDARD 17)
    gtest_add_tests(TARGET real_data_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR} EXTRA_ARGS "${CMAKE_SOURCE_DIR}/data")
  endif()

  # Enum validation tests
  add_executable(enum_validation_test src/test/enum_validation_test.cpp)
  add_dependencies(enum_validation_test fast_gicp)
  target_link_libraries(enum_validation_test ${GTEST_LIBRARIES} ${PCL_LIBRARIES} fast_gicp)
  if(BUILD_VGICP_CUDA)
    target_link_libraries(enum_validation_test fast_vgicp_cuda)
  endif()
  gtest_add_tests(TARGET enum_validation_test)

  # KNN benchmark (not a test, just for performance monitoring)
  if(BUILD_VGICP_CUDA)
    add_executable(knn_benchmark src/test/knn_benchmark.cu)
    add_dependencies(knn_benchmark fast_gicp)
    target_link_libraries(knn_benchmark fast_gicp fast_vgicp_cuda)
    set_property(TARGET knn_benchmark PROPERTY CUDA_LANGUAGE_STANDARD 17)
  endif()
endif()

###################################
## Standard CMake Installation  ##
###################################

# Install main library
install(TARGETS fast_gicp
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)

# Install headers
install(DIRECTORY include/
  DESTINATION include
  FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h"
)

# Install CUDA library if built
if(BUILD_VGICP_CUDA)
  install(TARGETS fast_vgicp_cuda
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
  )
endif()
