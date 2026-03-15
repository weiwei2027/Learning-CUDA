# 06_photon_sim 子项目 CMakeLists.txt
# 支持多平台: NVIDIA, Iluvatar, MetaX, Moore

project(photon_sim CUDA CXX)

# 设置 C++ 和 CUDA 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 包含目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# 平台选择 (可以通过 -DPLATFORM=xxx 指定)
set(PLATFORM "nvidia" CACHE STRING "Target platform: nvidia, iluvatar, metax, moore")

# ============================================================================
# 平台特定配置
# ============================================================================

if(PLATFORM STREQUAL "nvidia")
    message(STATUS "Building for NVIDIA platform")
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_ARCHITECTURES 61)
    set(MAIN_SRC src/photon_sim_nv.cu)
    set(TARGET_NAME photon_sim_nv)
    
elseif(PLATFORM STREQUAL "iluvatar")
    message(STATUS "Building for Iluvatar platform")
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    # Iluvatar 使用 clang++，需要特殊处理
    set(CMAKE_CUDA_COMPILER clang++)
    set(CMAKE_CUDA_FLAGS "--cuda-gpu-arch=BI100 -DILUVATAR_PLATFORM")
    set(MAIN_SRC src/photon_sim_iluvatar.cu)
    set(TARGET_NAME photon_sim_iluvatar)
    
elseif(PLATFORM STREQUAL "metax")
    message(STATUS "Building for MetaX platform")
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_COMPILER mxcc)
    set(CMAKE_CUDA_FLAGS "--cuda-gpu-arch=metax-c500 -DMETAX_PLATFORM")
    set(MAIN_SRC src/photon_sim_metax.maca)
    set(TARGET_NAME photon_sim_metax)
    
elseif(PLATFORM STREQUAL "moore")
    message(STATUS "Building for Moore platform")
    set(CMAKE_CUDA_STANDARD 11)  # Moore 只支持 C++11
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_COMPILER mcc)
    set(CMAKE_CUDA_FLAGS "-arch=mp_21 -DMOORE_PLATFORM")
    set(MAIN_SRC src/photon_sim_moore.mu)
    set(TARGET_NAME photon_sim_moore)
    
else()
    message(FATAL_ERROR "Unknown PLATFORM: ${PLATFORM}. Valid: nvidia, iluvatar, metax, moore")
endif()

# ============================================================================
# 主程序：光子传输模拟
# ============================================================================

add_executable(${TARGET_NAME}
    ${MAIN_SRC}
    src/utils.cpp
)

# CUDA 架构设置 (仅 NVIDIA)
if(PLATFORM STREQUAL "nvidia")
    set_target_properties(${TARGET_NAME} PROPERTIES
        CUDA_ARCHITECTURES 61
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    
    # CUDA 编译选项
    target_compile_options(${TARGET_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math>
    )
    
    # 链接 CUDA 库
    target_link_libraries(${TARGET_NAME} PRIVATE
        CUDA::cudart
        CUDA::cuda_driver
        CUDA::curand
    )
elseif(PLATFORM STREQUAL "iluvatar")
    target_link_libraries(${TARGET_NAME} PRIVATE
        cudart
        curand
    )
elseif(PLATFORM STREQUAL "metax")
    target_link_libraries(${TARGET_NAME} PRIVATE
        mcudart
        mcrand
    )
elseif(PLATFORM STREQUAL "moore")
    target_link_libraries(${TARGET_NAME} PRIVATE
        musart
        musa_rand
    )
endif()

# ============================================================================
# 拷贝数据文件到构建目录
# ============================================================================

add_custom_command(
    TARGET ${TARGET_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/data
        ${CMAKE_CURRENT_BINARY_DIR}/data
    COMMENT "Copying data directory to build folder..."
)

add_custom_command(
    TARGET ${TARGET_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
        ${CMAKE_CURRENT_BINARY_DIR}/output
    COMMENT "Creating output directory..."
)

# ============================================================================
# CPU 基准版本
# ============================================================================

add_executable(photon_sim_cpu
    src/photon_sim_cpu.cpp
    src/utils.cpp
)

target_include_directories(photon_sim_cpu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_custom_command(
    TARGET photon_sim_cpu POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/data
        ${CMAKE_CURRENT_BINARY_DIR}/data
    COMMENT "Copying data directory for CPU version..."
)

# ============================================================================
# 测试程序：解析器单元测试
# ============================================================================

add_executable(test_parser
    tests/test_parser.cpp
    src/utils.cpp
)

target_include_directories(test_parser PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_definitions(test_parser PRIVATE TESTING)
target_compile_options(test_parser PRIVATE -O2 -Wall)

# ============================================================================
# 自定义目标：运行程序
# ============================================================================

add_custom_target(run_${TARGET_NAME}
    COMMAND $<TARGET_FILE:${TARGET_NAME}>
        -g ${CMAKE_CURRENT_SOURCE_DIR}/data/geometry_3layer.txt
        -m ${CMAKE_CURRENT_SOURCE_DIR}/data/materials.csv
        -s ${CMAKE_CURRENT_SOURCE_DIR}/data/source_point_10m.txt
        -o ${CMAKE_CURRENT_SOURCE_DIR}/output/
    DEPENDS ${TARGET_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Running photon simulation on ${PLATFORM}..."
)
