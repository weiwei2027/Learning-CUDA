/**
 * @file photon_sim.cuh
 * @brief 光子模拟平台抽象头文件 - 支持多 GPU 平台
 * @author weiwei2027
 * 
 * 支持平台:
 * - NVIDIA: CUDA Runtime (默认)
 * - Iluvatar: 天数智芯 CoreX (clang++)
 * - MetaX: 沐曦集成电路 C500 (mxcc)
 * - Moore: 摩尔线程 MTT S5000 (mcc)
 */

#ifndef PHOTON_SIM_CUH
#define PHOTON_SIM_CUH

// ============================================================================
// 平台检测与头文件包含
// ============================================================================

#if defined(METAX_PLATFORM)
    // ==================== MetaX 平台 (沐曦) ====================
    #include <mcr/mc_runtime.h>
    #include <mcrand/mcrand_kernel.h>
    
    // MetaX 运行时 API 映射
    #define gpuMalloc mcMalloc
    #define gpuFree mcFree
    #define gpuMemcpy mcMemcpy
    #define gpuMemcpyHostToDevice mcMemcpyHostToDevice
    #define gpuMemcpyDeviceToHost mcMemcpyDeviceToHost
    #define gpuMemcpyDeviceToDevice mcMemcpyDeviceToDevice
    #define gpuMemset mcMemset
    #define gpuDeviceSynchronize mcDeviceSynchronize
    #define gpuGetLastError mcGetLastError
    #define gpuGetErrorString mcGetErrorString
    #define gpuMemcpyToSymbolAsync mcMemcpyToSymbolAsync
    #define gpuStream_t mcStream_t
    
    // MetaX 随机数
    #define gpuRandState mcrandState
    #define gpuRandInit mcrand_init
    #define gpuRandUniform mcrand_uniform
    
    // MetaX 原子操作
    #define gpuAtomicAdd atomicAdd
    
    // 平台名称
    #define PLATFORM_NAME "MetaX (沐曦 C500)"
    
#elif defined(MOORE_PLATFORM)
    // ==================== Moore 平台 (摩尔线程) ====================
    #include <musa_runtime.h>
    #include <musa_curand.h>
    
    // Moore 运行时 API 映射
    #define gpuMalloc musaMalloc
    #define gpuFree musaFree
    #define gpuMemcpy musaMemcpy
    #define gpuMemcpyHostToDevice musaMemcpyHostToDevice
    #define gpuMemcpyDeviceToHost musaMemcpyDeviceToHost
    #define gpuMemcpyDeviceToDevice musaMemcpyDeviceToDevice
    #define gpuMemset musaMemset
    #define gpuDeviceSynchronize musaDeviceSynchronize
    #define gpuGetLastError musaGetLastError
    #define gpuGetErrorString musaGetErrorString
    #define gpuMemcpyToSymbolAsync musaMemcpyToSymbolAsync
    #define gpuStream_t musaStream_t
    
    // Moore 随机数
    #define gpuRandState musa_randState
    #define gpuRandInit musa_rand_init
    #define gpuRandUniform musa_rand_uniform
    
    // Moore 原子操作
    #define gpuAtomicAdd musaAtomicAdd
    
    // 平台名称
    #define PLATFORM_NAME "Moore (摩尔线程 MTT S5000)"
    
#elif defined(ILUVATAR_PLATFORM)
    // ==================== Iluvatar 平台 (天数智芯) ====================
    #include <cuda_runtime.h>
    #include <curand_kernel.h>
    
    // Iluvatar 使用标准 CUDA API
    #define gpuMalloc cudaMalloc
    #define gpuFree cudaFree
    #define gpuMemcpy cudaMemcpy
    #define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
    #define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
    #define gpuMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
    #define gpuMemset cudaMemset
    #define gpuDeviceSynchronize cudaDeviceSynchronize
    #define gpuGetLastError cudaGetLastError
    #define gpuGetErrorString cudaGetErrorString
    #define gpuMemcpyToSymbolAsync cudaMemcpyToSymbolAsync
    #define gpuStream_t cudaStream_t
    
    // Iluvatar 随机数
    #define gpuRandState curandState
    #define gpuRandInit curand_init
    #define gpuRandUniform curand_uniform
    
    // Iluvatar 原子操作
    #define gpuAtomicAdd atomicAdd
    
    // 平台名称
    #define PLATFORM_NAME "Iluvatar (天数智芯 BI-V100)"
    
#else
    // ==================== NVIDIA 平台 (默认) ====================
    #include <cuda_runtime.h>
    #include <curand_kernel.h>
    
    // NVIDIA 使用标准 CUDA API
    #define gpuMalloc cudaMalloc
    #define gpuFree cudaFree
    #define gpuMemcpy cudaMemcpy
    #define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
    #define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
    #define gpuMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
    #define gpuMemset cudaMemset
    #define gpuDeviceSynchronize cudaDeviceSynchronize
    #define gpuGetLastError cudaGetLastError
    #define gpuGetErrorString cudaGetErrorString
    #define gpuMemcpyToSymbolAsync cudaMemcpyToSymbolAsync
    #define gpuStream_t cudaStream_t
    
    // NVIDIA 随机数
    #define gpuRandState curandState
    #define gpuRandInit curand_init
    #define gpuRandUniform curand_uniform
    
    // NVIDIA 原子操作
    #define gpuAtomicAdd atomicAdd
    
    // 平台名称
    #define PLATFORM_NAME "NVIDIA (CUDA)"
    
#endif

// ============================================================================
// 通用宏定义
// ============================================================================

// 最大常量内存大小（取各平台最小值）
#define MAX_REGIONS 16
#define MAX_SPHERES 8

// GPU 错误检查宏
#define GPU_CHECK(call) \
    do { \
        auto err = call; \
        if (err != 0) { \
            fprintf(stderr, "GPU Error at %s:%d\n", __FILE__, __LINE__); \
            exit(1); \
        } \
    } while(0)

// 内核启动宏（用于错误检查）
#define KERNEL_CHECK() \
    do { \
        auto err = gpuGetLastError(); \
        if (err != 0) { \
            fprintf(stderr, "Kernel launch error at %s:%d\n", __FILE__, __LINE__); \
            exit(1); \
        } \
    } while(0)

#endif // PHOTON_SIM_CUH
