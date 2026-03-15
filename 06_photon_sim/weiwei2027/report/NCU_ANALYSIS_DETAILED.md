# NCU 详细性能分析报告

**分析工具**: NVIDIA Nsight Compute  
**测试配置**: 10⁹ 光子 + 球体异物 (geometry_3layer_sphere.txt)  
**测试时间**: 2026-03-15  
**分析者**: weiwei2027

---

## 1. 执行摘要

### 1.1 核心性能指标

| 指标 | photonTransportKernel | 评估 |
|------|----------------------|------|
| **执行时间** | **42.59 ms** | 🟢 优秀 (10亿光子) |
| **处理速率** | **2.35×10¹⁰ photons/sec** | 🟢 优秀 |
| **SM 利用率** | **67.59%** | 🟡 良好 |
| **DRAM 利用率** | **0.20%** | 🟢 符合预期 (计算密集型) |
| **理论 Occupancy** | **66.67%** | 🟡 受寄存器限制 |
| **实际 Occupancy** | **32.33%** | 🟠 有优化空间 |

### 1.2 三个核函数对比

| 核函数 | 执行时间 | SM 利用率 | 主要特征 |
|--------|----------|-----------|----------|
| `clearDetectorKernel` | 1.79 μs | 1.66% | 内存初始化，grid太小 |
| `initRandState` | 248.93 μs | 72.10% | cuRAND 初始化，内存密集 |
| `photonTransportKernel` | 42.59 ms | 67.59% | **计算密集，主导执行时间** |

---

## 2. 详细性能分析

### 2.1 计算利用率分析

```
Compute Workload Analysis:
├── Executed IPC (Active): 2.68 inst/cycle  
├── SM Busy: 67.59%
├── Issue Slots Busy: 67.59%
└── ALU Pipeline: 42.0% (最高利用率)
```

**关键发现**:
- ✅ **计算密集型特征明显**: SM Busy 67.59%，DRAM 仅 0.20%
- ✅ **指令发射效率良好**: IPC 2.68 接近理论上限
- 🟡 **ALU 利用率适中**: 42%，有进一步提升空间

### 2.2 Occupancy 分析

| 限制因素 | 限制值 | 实际值 | 状态 |
|----------|--------|--------|------|
| Block Limit SM | 24 blocks | 256 blocks | 🟢 未限制 |
| Block Limit Registers | **4 blocks** | 256 blocks | 🔴 **主要限制** |
| Block Limit Warps | 6 blocks | 256 blocks | 🟢 未限制 |
| Theoretical Occupancy | 66.67% | - | 🟡 受寄存器限制 |
| **Achieved Occupancy** | - | **32.33%** | 🟠 **仅为理论值的一半** |

**关键问题**:
```
Registers Per Thread: 54
Block Size: 256 threads
Registers Per Block: 54 × 256 = 13,824

SM 寄存器限制导致每个 SM 只能运行 4 个 block (理论 24 个)
```

### 2.3 Warp 状态分析

| 指标 | 数值 | 分析 |
|------|------|------|
| Avg. Active Threads Per Warp | 20.17 | 🟠 理想值 32，分支导致线程减少 |
| Avg. Not Predicated Off | 19.04 | 🟠 约 40% 线程被 predication 关闭 |
| Branch Efficiency | 87.04% | 🟡 存在一定分支发散 |
| Avg. Divergent Branches | 630,752 | 🟠 球体相交检测导致分支发散 |

**主要 Stall 原因**:
1. **Fixed Latency Execution Dependency (38.3%)**
   - 等待固定延迟指令结果（如三角函数、开方）
   - 建议: 增加活跃 warp 数来隐藏延迟

2. **Warp 调度不均衡**
   - 理论每 scheduler 8 warps，硬件最大 12 warps
   - 实际 warp 数不足导致 issue slot 空闲

### 2.4 内存访问分析

```
Memory Workload Analysis:
├── Memory Throughput: 1.95 GB/s
├── DRAM Throughput: 0.20%
├── L1/TEX Hit Rate: 2.48% (极低，符合随机访问)
├── L2 Hit Rate: 91.24%
└── Mem Pipes Busy: 15.95%
```

**非合并访问问题**:
- **34% 的过度 sector** (9,339,247 excessive sectors)
- 原因: `atomicAdd` 到探测器像素的随机访问模式
- 优化潜力: **12.06% 性能提升**

---

## 3. 性能瓶颈识别

### 3.1 瓶颈排名

| 排名 | 瓶颈 | 影响 | NCU 建议 | 优化潜力 |
|------|------|------|----------|----------|
| 1 | **寄存器压力** | Occupancy 限制在 66.7% | 减少寄存器使用 | ⭐⭐⭐⭐⭐ |
| 2 | **执行依赖延迟** | 38.3% stall 时间 | 增加活跃 warp，指令重排 | ⭐⭐⭐⭐ |
| 3 | **非合并内存访问** | 34% 过度 sector | 共享内存归约 | ⭐⭐⭐ |
| 4 | **Warp 分支发散** | 平均 20.2 活跃线程 | 减少分支，统一执行路径 | ⭐⭐⭐ |
| 5 | **指令融合** | 非融合 FP32 指令 | 使用 --use_fast_math | ⭐⭐ |

### 3.2 与理论峰值对比

| 资源 | 理论峰值 | 当前利用率 | 差距分析 |
|------|----------|------------|----------|
| SM 计算吞吐 | 100% | 67.59% | 受寄存器限制和 stall 影响 |
| FP32 峰值 | 82.6 TFLOPS | 12% | 蒙特卡洛算法特性 |
| DRAM 带宽 | 1,008 GB/s | 0.20% | 计算密集型，非瓶颈 |
| Occupancy | 100% | 32.33% | **主要优化目标** |

---

## 4. 具体优化建议

### 4.1 优化 1: 降低寄存器压力 (高优先级)

**问题**: 54 registers/thread 限制 occupancy 至 66.7%

**方案 A**: 使用 launch bounds 限制寄存器
```cpp
__launch_bounds__(256, 6)  // 256 threads, min 6 blocks per SM
__global__ void photonTransportKernel(...)
```

**方案 B**: 减少寄存器变量
```cpp
// 当前可能的寄存器使用:
// Photon p (7 floats + 1 bool = 8 regs)
// localState (curandState ~12 regs)
// 循环变量、中间计算 (~20 regs)
// 预计算边界 (~10 regs)
// 总计: ~50+ regs

// 优化: 使用 shared memory 存储 Photon 数组
__shared__ Photon s_photons[256];  // 减少寄存器压力
```

**预期收益**: 
- Occupancy 从 32% → 50%+
- 性能提升 **15-25%**

### 4.2 优化 2: 隐藏执行延迟 (中优先级)

**问题**: 38.3% stall 在等待固定延迟指令

**方案**: 增加每线程处理光子数，使用更多 ILP
```cpp
// 当前: 每个线程顺序处理多个光子
for (int i = tid; i < num_photons; i += total_threads) {
    process_one_photon();
}

// 优化: 同时处理 2-4 个光子，增加指令级并行
#pragma unroll 2
for (int i = tid; i < num_photons; i += total_threads * 2) {
    process_photon(i);
    process_photon(i + total_threads);
}
```

**预期收益**: 减少 stall 10-15%，性能提升 **8-12%**

### 4.3 优化 3: 共享内存归约 (中优先级)

**问题**: `atomicAdd` 导致 34% 非合并访问

**方案**: Block 级共享内存累加
```cpp
__shared__ float s_pixels[1024];  // 每个 block 的局部缓冲区

// 1. 初始化共享内存
for (int i = threadIdx.x; i < 1024; i += blockDim.x) {
    s_pixels[i] = 0.0f;
}
__syncthreads();

// 2. 在共享内存累加
atomicAdd(&s_pixels[local_idx], 1.0f);
__syncthreads();

// 3. 统一写回全局内存
for (int i = threadIdx.x; i < 1024; i += blockDim.x) {
    atomicAdd(&g_pixels[i], s_pixels[i]);
}
```

**预期收益**: 减少全局原子竞争，性能提升 **10-15%**

### 4.4 优化 4: 快速数学函数 (低优先级)

**问题**: NCU 报告 4.33% 优化空间通过 FP32 指令融合

**方案**: 使用 `--use_fast_math` 编译选项
```bash
nvcc -O3 --use_fast_math -arch=sm_89 ...
```

**影响**: 
- `logf`, `sqrtf`, `sinf`, `cosf` 等变为低精度但快速版本
- 蒙特卡洛模拟对精度要求不高，可以接受

**预期收益**: 性能提升 **3-5%**

### 4.5 优化 5: 分支优化 (低优先级)

**问题**: 球体相交检测导致 warp 分支发散

**方案**: 减少分支，使用 predication
```cpp
// 当前: 明显的 if-else 分支
if (intersectSphere(...)) {
    // 处理球体
} else {
    // 处理普通层
}

// 优化: 减少嵌套分支，使用计算替代分支
float hit = intersectSphere(...) ? 1.0f : 0.0f;
// 使用 hit 作为权重进行计算
```

**预期收益**: Warp 效率提升，性能提升 **5-8%**

---

## 5. 优化前后性能预测

| 优化项 | 当前性能 | 优化后预测 | 提升 |
|--------|----------|------------|------|
| 基准 (10亿光子) | 2.35×10¹⁰ p/s | - | - |
| + 降低寄存器压力 | - | 2.70×10¹⁰ p/s | +15% |
| + 隐藏执行延迟 | - | 2.95×10¹⁰ p/s | +8% |
| + 共享内存归约 | - | 3.25×10¹⁰ p/s | +10% |
| + 快速数学函数 | - | 3.35×10¹⁰ p/s | +3% |
| **理论最大** | - | **~3.5×10¹⁰ p/s** | **+49%** |

---

## 6. 结论

### 6.1 当前状态评估

| 维度 | 评估 | 说明 |
|------|------|------|
| **算法效率** | 🟢 优秀 | 2.35×10¹⁰ p/s，蒙特卡洛模拟的高效实现 |
| **计算利用率** | 🟡 良好 | 67.59%，有提升空间 |
| **Occupancy** | 🟠 一般 | 32.33%，寄存器限制是主要瓶颈 |
| **内存效率** | 🟢 优秀 | 计算密集型，内存非瓶颈 |
| **代码质量** | 🟢 良好 | 结构化清晰，优化空间大 |

### 6.2 关键发现

1. **寄存器压力是最大瓶颈**: 54 registers/thread 限制 occupancy
2. **计算密集型特征明显**: DRAM 利用率仅 0.2%，符合预期
3. **执行延迟可隐藏**: 38% stall 可通过增加并行度解决
4. **原子操作有优化空间**: 共享内存归约可减少非合并访问

### 6.3 推荐优化顺序

```
Phase 1 (立即实施):
├── 1. 添加 --use_fast_math 编译选项 (+3-5%)
└── 2. 使用 __launch_bounds__ 限制寄存器 (+15-25%)

Phase 2 (短期):
├── 3. 实现共享内存归约 (+10-15%)
└── 4. 增加每线程光子数，提升 ILP (+8-12%)

Phase 3 (可选):
└── 5. 分支优化，减少 warp 发散 (+5-8%)
```

### 6.4 最终评估

本项目实现的 CUDA 光子传输模拟器已达到**生产级性能水平**，通过上述优化可进一步提升 **40-50%** 性能。

**当前版本**: ✅ 适合提交，性能优异  
**优化后版本**: 🚀 可冲击更高性能指标

---

*分析工具: NVIDIA Nsight Compute*  
*分析时间: 2026-03-15*  
*报告版本: 2.0 (基于实际 NCU 数据)*
