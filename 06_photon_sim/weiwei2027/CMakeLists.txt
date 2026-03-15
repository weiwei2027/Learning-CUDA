# 06_photon_sim/weiwei2027 - 医学成像光子传输模拟
# 作者: weiwei2027
# 训练营: 2025冬季训练营 CUDA方向
#
# 使用方法:
#   cmake -B build -S .                           # 配置
#   cmake --build build --parallel                # 构建 (默认NVIDIA)
#   cmake --build build --parallel -DPLATFORM=nvidia  # 指定平台
#
# 支持的PLATFORM: nvidia, iluvatar, metax, moore

cmake_minimum_required(VERSION 3.18)

# ============================================================================
# 项目声明
# ============================================================================

project(photon_sim 
    VERSION 1.0
    DESCRIPTION "医学成像光子传输模拟 - CUDA加速"
    LANGUAGES CXX
)

# 查找 CUDA（必需）
find_program(CMAKE_CUDA_COMPILER nvcc
    PATHS
        /usr/local/cuda/bin
        /opt/cuda/bin
        $ENV{CUDA_HOME}/bin
        $ENV{CUDA_PATH}/bin
    NO_DEFAULT_PATH
)

if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "CUDA compiler (nvcc) not found. Please install CUDA or set CUDA_HOME.")
endif()

message(STATUS "Found CUDA compiler: ${CMAKE_CUDA_COMPILER}")
enable_language(CUDA)

# ============================================================================
# 全局设置
# ============================================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 包含目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# ============================================================================
# 平台选择
# ============================================================================

set(PLATFORM "nvidia" CACHE STRING "Target platform: nvidia, iluvatar, metax, moore")
set_property(CACHE PLATFORM PROPERTY STRINGS nvidia iluvatar metax moore)

message(STATUS "============================================================")
message(STATUS "Platform: ${PLATFORM}")
message(STATUS "============================================================")

# ============================================================================
# 平台配置
# ============================================================================

if(PLATFORM STREQUAL "nvidia")
    # NVIDIA CUDA
    set(CMAKE_CUDA_ARCHITECTURES 80 89)
    set(MAIN_SRC src/photon_sim_nv.cu)
    set(TARGET_NAME photon_sim_nv)
    
elseif(PLATFORM STREQUAL "iluvatar")
    # 天数智芯 CoreX
    message(STATUS "Note: Iluvatar requires clang++ compiler")
    set(CMAKE_CUDA_COMPILER clang++)
    set(CMAKE_CUDA_FLAGS "--cuda-gpu-arch=BI100")
    set(CMAKE_CUDA_ARCHITECTURES BI100)
    set(MAIN_SRC src/photon_sim_iluvatar.cu)
    set(TARGET_NAME photon_sim_iluvatar)
    
elseif(PLATFORM STREQUAL "metax")
    # 沐曦 MetaX
    message(STATUS "Building for MetaX platform")
    set(CMAKE_CUDA_COMPILER mxcc)
    set(CMAKE_CUDA_STANDARD 17)
    set(MAIN_SRC src/photon_sim_metax.maca)
    set(TARGET_NAME photon_sim_metax)
    
elseif(PLATFORM STREQUAL "moore")
    # 摩尔线程 Moore
    message(STATUS "Note: Moore requires mcc compiler")
    set(CMAKE_CUDA_COMPILER mcc)
    set(CMAKE_CUDA_FLAGS "-arch=mp_21")
    set(CMAKE_CXX_STANDARD 11)  # Moore 只支持 C++11
    set(MAIN_SRC src/photon_sim_moore.mu)
    set(TARGET_NAME photon_sim_moore)
    
else()
    message(FATAL_ERROR "Unknown PLATFORM: ${PLATFORM}")
endif()

# ============================================================================
# GPU 版本
# ============================================================================

add_executable(${TARGET_NAME}
    ${MAIN_SRC}
    src/utils.cpp
)

# CUDA 架构设置
set_target_properties(${TARGET_NAME} PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    CUDA_SEPARABLE_COMPILATION ON
)

# NVIDIA 平台特定优化
if(PLATFORM STREQUAL "nvidia")
    find_package(CUDAToolkit REQUIRED)
    
    target_compile_options(${TARGET_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math>
    )
    
    target_link_libraries(${TARGET_NAME} PRIVATE
        CUDA::cudart
        CUDA::cuda_driver
        CUDA::curand
    )
else()
    # 其他平台使用标准库链接
    target_link_libraries(${TARGET_NAME} PRIVATE
        cudart
        curand
    )
endif()

# ============================================================================
# CPU 基准版本
# ============================================================================

add_executable(photon_sim_cpu
    src/photon_sim_cpu.cpp
    src/utils.cpp
)

target_include_directories(photon_sim_cpu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# ============================================================================
# 拷贝数据文件
# ============================================================================

add_custom_command(
    TARGET ${TARGET_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/data
        ${CMAKE_CURRENT_BINARY_DIR}/data
    COMMENT "Copying data files..."
)

add_custom_command(
    TARGET photon_sim_cpu POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/data
        ${CMAKE_CURRENT_BINARY_DIR}/data
    COMMENT "Copying data files for CPU..."
)

# 创建输出目录
add_custom_command(
    TARGET ${TARGET_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
        ${CMAKE_CURRENT_BINARY_DIR}/output
)

# ============================================================================
# 测试程序
# ============================================================================

add_executable(test_parser
    tests/test_parser.cpp
    src/utils.cpp
)

target_include_directories(test_parser PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# ============================================================================
# 使用说明
# ============================================================================

message(STATUS "")
message(STATUS "Build configuration:")
message(STATUS "  Platform:    ${PLATFORM}")
message(STATUS "  Target:      ${TARGET_NAME}")
message(STATUS "  CUDA Arch:   ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Build Type:  ${CMAKE_BUILD_TYPE}")
message(STATUS "")
message(STATUS "Usage:")
message(STATUS "  ./${TARGET_NAME} -g data/geometry_3layer.txt -m data/materials.csv -s data/source_point_10m.txt -o output/")
message(STATUS "")
