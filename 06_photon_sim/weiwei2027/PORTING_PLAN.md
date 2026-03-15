# CUDA 多平台移植计划

## 平台信息汇总

| 平台 | 厂商 | 设备名 | 编译器 | 运行时头文件 | C++标准 | 共享内存/块 |
|------|------|--------|--------|-------------|---------|------------|
| NVIDIA | NVIDIA | A100 | `nvcc` | `cuda_runtime.h` | C++17 | 48 KB |
| Iluvatar | 天数智芯 | BI-V100 | `clang++` | `cuda_runtime.h` | C++17 | 128 KB |
| MetaX | 沐曦 | C500 | `mxcc` | `mcr/mc_runtime.h` | C++17 | 64 KB |
| Moore | 摩尔线程 | MTT S5000 | `mcc` | `musa_runtime.h` | C++11 | 192 KB |

## 移植策略

### 代码组织

```
src/
├── photon_sim.cu          # NVIDIA & Iluvatar 共用 (CUDA-compatible)
├── photon_sim.maca        # MetaX 专用 (MACA runtime)
├── photon_sim.mu          # Moore 专用 (MUSA runtime)
├── photon_sim.cuh         # 公共头文件 (平台宏隔离)
├── main.cu                # 主程序 (平台无关)
└── utils.cpp              # 工具函数 (平台无关)

Makefile                    # 多平台构建系统
```

### 关键差异处理

| 特性 | NVIDIA/Iluvatar | MetaX | Moore |
|------|-----------------|-------|-------|
| CUDA 核函数语法 | `__global__` | `__global__` | `__global__` |
| 设备函数 | `__device__` | `__device__` | `__device__` |
| 内存拷贝 | `cudaMemcpy` | `mcMemcpy` | `musaMemcpy` |
| 错误检查 | `cudaGetLastError` | `mcGetLastError` | `musaGetLastError` |
| 随机数 | `curand` | `mcrand` | `musa_rand` |
| 原子操作 | `atomicAdd` | `atomicAdd` | `musaAtomicAdd` |

### 头文件平台隔离 (photon_sim.cuh)

```cpp
#if defined(METAX_PLATFORM)
    #include <mcr/mc_runtime.h>
    #include <mcr/mcrand_kernel.h>
    #define GPU_CHECK mcCheck
#elif defined(MOORE_PLATFORM)
    #include <musa_runtime.h>
    #include <musa_curand.h>
    #define GPU_CHECK musaCheck
#else  // NVIDIA & Iluvatar
    #include <cuda_runtime.h>
    #include <curand_kernel.h>
    #define GPU_CHECK cudaCheck
#endif
```

## Phase 1: NVIDIA CUDA 优化 ✅ 完成

### 已实现功能

1. **功能完善**
   - [x] 实现 GPU 球体相交检测
   - [x] 完整支持点光源和平行光源
   - [x] 与 CPU 版本结果对比验证

2. **性能优化**
   - [x] 共享内存缓存几何参数（常量内存）
   - [x] Grid-stride 循环优化
   - [x] 寄存器使用优化

3. **代码重构**
   - [x] 提取公共头文件
   - [x] 平台宏定义隔离
   - [x] 错误处理统一封装

## Phase 2-5: 多平台移植 ⏳ 待完成

每个平台移植步骤：
1. 复制 NVIDIA 版本代码
2. 替换运行时 API 调用
3. 处理随机数库差异
4. 编译测试与调试
5. 性能基准测试

### 平台特定 Makefile 配置模板

```makefile
# 需要时添加到 Makefile

ifeq ($(PLATFORM),iluvatar)
    CXX = clang++
    CXXFLAGS = -O3 --cuda-gpu-arch=BI100 -std=c++17
    LDFLAGS = -lcudart -lcurand
    
else ifeq ($(PLATFORM),metax)
    CXX = mxcc
    CXXFLAGS = -O3 --cuda-gpu-arch=metax-c500 -std=c++17 -DMETAX_PLATFORM
    LDFLAGS = -lmcudart -lmcrand
    
else ifeq ($(PLATFORM),moore)
    CXX = mcc
    CXXFLAGS = -O3 -arch=mp_21 -std=c++11 -DMOORE_PLATFORM
    LDFLAGS = -lmusart -lmusa_rand
endif
```

## 构建命令

```bash
# NVIDIA (默认)
make PLATFORM=nvidia

# Iluvatar
make PLATFORM=iluvatar

# MetaX
make PLATFORM=metax

# Moore
make PLATFORM=moore

# 运行测试
make run PLATFORM=nvidia
```

## 测试验证清单

每个平台移植完成后需要验证：

| 测试项 | 通过标准 |
|-------|---------|
| geometry_1layer.txt | 透过率 ≈ 81.87% |
| geometry_3layer.txt | 透过率 ≈ 3.62% |
| geometry_3layer_sphere.txt | 探测器中心有阴影 |
| 点光源模式 | 锥形束分布正确 |
| 平行光源模式 | 均匀分布正确 |
| CPU/GPU 一致性 | 差异 < 0.1% |
