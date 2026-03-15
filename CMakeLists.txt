# Learning-CUDA 项目根 CMakeLists.txt
# 作用: 为 CLion 等 IDE 提供项目入口，转发到实际作品目录
# 注意: 比赛作品位于 06_photon_sim/weiwei2027/ 目录下

cmake_minimum_required(VERSION 3.18)

# 查找 CUDA 编译器
find_program(CMAKE_CUDA_COMPILER nvcc
    PATHS
        /usr/local/cuda/bin
        /opt/cuda/bin
        $ENV{CUDA_HOME}/bin
        $ENV{CUDA_PATH}/bin
    NO_DEFAULT_PATH
)

if(NOT CMAKE_CUDA_COMPILER)
    message(WARNING "CUDA compiler not found. GPU targets will not be available.")
endif()

# 声明项目
project(Learning-CUDA LANGUAGES CXX)

# 启用 CUDA（如果可用）
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    message(STATUS "CUDA enabled for IDE support")
endif()

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================================
# 转发到实际作品目录
# ============================================================================

set(WORK_DIR ${CMAKE_CURRENT_SOURCE_DIR}/06_photon_sim/weiwei2027)

if(EXISTS ${WORK_DIR}/CMakeLists.txt)
    message(STATUS "Loading project from: ${WORK_DIR}")
    add_subdirectory(${WORK_DIR})
else()
    message(WARNING "Project not found at: ${WORK_DIR}")
endif()
