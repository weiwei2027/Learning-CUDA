# 06_photon_sim 子项目 CMakeLists.txt

project(photon_sim CUDA CXX)

# 设置 C++ 和 CUDA 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 包含目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# ============================================
# 主程序：光子传输模拟
# ============================================

# 主程序源文件 - 全部作为 CUDA 编译
set(PHOTON_SIM_SOURCES
    src/main.cu
    src/photon_sim.cu
    src/utils.cpp
)

add_executable(photon_sim ${PHOTON_SIM_SOURCES})

# CUDA 架构设置
set_target_properties(photon_sim PROPERTIES
    CUDA_ARCHITECTURES 89
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# CUDA 编译选项
target_compile_options(photon_sim PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math>
)

# 链接 CUDA 库
target_link_libraries(photon_sim PRIVATE
    CUDA::cudart
    CUDA::cuda_driver
    CUDA::curand
)

# ============================================
# 拷贝数据文件到构建目录
# ============================================

# 创建 data 目录并拷贝文件（在 build 时执行）
add_custom_command(
    TARGET photon_sim POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/data
        ${CMAKE_CURRENT_BINARY_DIR}/data
    COMMENT "Copying data directory to build folder..."
)

# 同时创建 output 目录
add_custom_command(
    TARGET photon_sim POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
        ${CMAKE_CURRENT_BINARY_DIR}/output
    COMMENT "Creating output directory..."
)

# ============================================
# CPU 基准版本
# ============================================

add_executable(photon_sim_cpu
    src/photon_sim_cpu.cpp
    src/utils.cpp
)

set_target_properties(photon_sim_cpu PROPERTIES
    CUDA_ARCHITECTURES 89
)

target_include_directories(photon_sim_cpu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# 拷贝数据目录
add_custom_command(
    TARGET photon_sim_cpu POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/data
        ${CMAKE_CURRENT_BINARY_DIR}/data
    COMMENT "Copying data directory for CPU version..."
)

# ============================================
# 测试程序：解析器单元测试
# ============================================

add_executable(test_parser
    tests/test_parser.cpp
    src/utils.cpp
)

target_include_directories(test_parser PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_definitions(test_parser PRIVATE TESTING)
target_compile_options(test_parser PRIVATE -O2 -Wall)

# ============================================
# 自定义目标：运行程序
# ============================================

# 运行主程序
add_custom_target(run_photon_sim
    COMMAND $<TARGET_FILE:photon_sim>
        -g ${CMAKE_CURRENT_SOURCE_DIR}/data/geometry_3layer.txt
        -m ${CMAKE_CURRENT_SOURCE_DIR}/data/materials.csv
        -s ${CMAKE_CURRENT_SOURCE_DIR}/data/source_point_10m.txt
        -o ${CMAKE_CURRENT_SOURCE_DIR}/output/
    DEPENDS photon_sim
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Running photon simulation..."
)
