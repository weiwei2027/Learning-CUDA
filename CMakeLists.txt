# Learning-CUDA 项目根 CMakeLists.txt
# 用于 CLion 等 IDE 的 CMake 支持

cmake_minimum_required(VERSION 3.18)

# 先查找 CUDA 编译器，再声明项目
find_program(CMAKE_CUDA_COMPILER nvcc
    PATHS
        /usr/local/cuda/bin
        /opt/cuda/bin
        $ENV{CUDA_HOME}/bin
        $ENV{CUDA_PATH}/bin
    NO_DEFAULT_PATH
)

if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "CUDA compiler (nvcc) not found. Please ensure CUDA is installed and in PATH, or set CUDA_HOME environment variable.")
endif()

message(STATUS "Found CUDA compiler: ${CMAKE_CUDA_COMPILER}")

# 现在声明项目，包含 CUDA 语言
project(Learning-CUDA LANGUAGES CUDA CXX)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置 CUDA 标准
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 显式启用 CUDA 语言支持
enable_language(CUDA)

# 查找 CUDA 工具包
find_package(CUDAToolkit REQUIRED)

# 设置 CUDA 架构（可以根据需要调整）
set(CMAKE_CUDA_ARCHITECTURES 80)

# 设置编译器标志
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_89")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math")

# 添加子项目
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/06_photon_sim/weiwei2027/CMakeLists.txt)
    add_subdirectory(06_photon_sim/weiwei2027)
endif()

message(STATUS "CUDA configuration completed successfully")
