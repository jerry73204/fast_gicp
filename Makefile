# Fast GICP C++ Library Makefile
# Simple makefile for building and testing the fast_gicp library

# Configuration variables
BUILD_DIR := ../build_fast_gicp
SOURCE_DIR := .
CMAKE_BUILD_TYPE ?= Release
CUDA_VERSION ?= 12.8
CUDA_ROOT := /usr/local/cuda-$(CUDA_VERSION)
JOBS := $(shell nproc)

# Test configuration
TEST_BINARY := $(BUILD_DIR)/gicp_test
TEST_DATA := $(SOURCE_DIR)/data

# Default target
.PHONY: all
all: build

# Build the library
.PHONY: build
build: configure
	@echo "Building fast_gicp with $(JOBS) parallel jobs..."
	@cmake --build $(BUILD_DIR) --parallel
	@echo "Build completed successfully"

# Configure with CMake
.PHONY: configure
configure: check-deps $(BUILD_DIR)
	@echo "Configuring CMake for CUDA $(CUDA_VERSION)..."
	@cmake -S $(SOURCE_DIR) -B $(BUILD_DIR) \
		-DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) \
		-DBUILD_VGICP_CUDA=ON \
		-DBUILD_apps=OFF \
		-DBUILD_test=ON \
		-DBUILD_PYTHON_BINDINGS=OFF \
		-DCMAKE_CUDA_ARCHITECTURES="75;80;86;87;89;90" \
		-DCMAKE_CUDA_COMPILER="$(CUDA_ROOT)/bin/nvcc" \
		-DCUDA_TOOLKIT_ROOT_DIR="$(CUDA_ROOT)"

# Build tests
.PHONY: build-test
build-test: configure
	@echo "Building test executable..."
	@cmake --build $(BUILD_DIR) --target gicp_test --parallel

# Run tests
.PHONY: test
test: build-test
	@echo "Running Fast GICP C++ tests..."
	@if [ -f "$(TEST_BINARY)" ]; then \
		echo "Executing tests..."; \
		$(TEST_BINARY) $(TEST_DATA); \
	else \
		echo "Error: Test binary not found at $(TEST_BINARY)"; \
		exit 1; \
	fi

# Clean build directory
.PHONY: clean
clean:
	@echo "Cleaning build directory..."
	@rm -rf $(BUILD_DIR)
	@echo "Build directory removed"

# Check dependencies
.PHONY: check-deps
check-deps:
	@echo "Checking dependencies..."
	@command -v cmake >/dev/null 2>&1 || { echo "Error: cmake not found"; exit 1; }
	@command -v nvcc >/dev/null 2>&1 || { echo "Warning: nvcc not found - CUDA tests will fail"; }
	@pkg-config --exists pcl_common-1.12 || { echo "Error: PCL not found"; exit 1; }
	@pkg-config --exists eigen3 || { echo "Error: Eigen3 not found"; exit 1; }

# Check CUDA installation
.PHONY: check-cuda
check-cuda:
	@echo "Checking CUDA installation..."
	@if [ -d "$(CUDA_ROOT)" ]; then \
		echo "Found CUDA $(CUDA_VERSION) at $(CUDA_ROOT)"; \
		$(CUDA_ROOT)/bin/nvcc --version | head -3; \
	else \
		echo "Error: CUDA $(CUDA_VERSION) not found at $(CUDA_ROOT)"; \
		echo "Available CUDA versions:"; \
		ls -1 /usr/local/cuda-* 2>/dev/null || echo "No CUDA installations found"; \
		exit 1; \
	fi

# Verify build artifacts
.PHONY: verify
verify:
	@echo "Verifying build artifacts..."
	@if [ -f "$(BUILD_DIR)/libfast_gicp.so" ]; then \
		echo "Found: libfast_gicp.so"; \
	else \
		echo "Missing: libfast_gicp.so"; \
	fi
	@if [ -f "$(BUILD_DIR)/libfast_vgicp_cuda.so" ]; then \
		echo "Found: libfast_vgicp_cuda.so"; \
		echo "CUDA libraries linked:"; \
		ldd $(BUILD_DIR)/libfast_vgicp_cuda.so | grep -E "(cuda|cublas|curand)" || echo "No CUDA libraries found"; \
	else \
		echo "Missing: libfast_vgicp_cuda.so"; \
	fi

# Create build directory
$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

# Help target
.PHONY: help
help:
	@echo "Fast GICP C++ Library Makefile"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  all          - Build the library (default)"
	@echo "  build        - Build the library"
	@echo "  test         - Build and run tests"
	@echo "  clean        - Remove build directory"
	@echo "  configure    - Configure CMake only"
	@echo "  build-test   - Build test executable only"
	@echo "  check-deps   - Check system dependencies"
	@echo "  check-cuda   - Check CUDA installation"
	@echo "  verify       - Verify build artifacts"
	@echo "  help         - Show this help message"
	@echo ""
	@echo "Environment variables:"
	@echo "  CMAKE_BUILD_TYPE - Build type (default: Release)"
	@echo "  CUDA_VERSION     - CUDA version (default: 12.8)"
	@echo ""
	@echo "Examples:"
	@echo "  make                      # Build library"
	@echo "  make test                 # Build and run tests"
	@echo "  make clean build          # Clean rebuild"
	@echo "  CMAKE_BUILD_TYPE=Debug make build"
