# 医学成像光子传输模拟 - 设计文档

> 文档版本: 1.1  
> 最后更新: 2026-03-14  
> 作者: weiwei2027

---

## 1. 项目概述

### 1.1 项目目标

实现一个基于蒙特卡洛方法的光子传输模拟器，用于医学成像（CT/X射线）中的辐射传输计算。项目支持 GPU (CUDA) 加速，并与 CPU 基准版本对比验证。

### 1.2 核心功能

- ✅ 多层平板几何（皮肤、颅骨、脑组织）的光子传输模拟
- ✅ 球体异物支持（血块、肿瘤等病变模拟）
- ✅ 点源和平行束两种 X 射线源模式
- ✅ GPU 加速（CUDA）与 CPU 基准版本
- ✅ 探测器成像和性能分析

---

## 2. 物理模型

### 2.1 基本假设

本项目采用**简化物理模型**，在计算效率和物理精度之间取得平衡：

| 假设 | 说明 | 合理性 |
|------|------|--------|
| **直穿近似** | 忽略散射，光子沿直线传播 | 高能量(>50keV)时散射较弱 |
| **几何光学** | 忽略折射，n ≈ 1 | X射线折射率≈1，折射角<0.001° |
| **纯吸收模型** | 相互作用=吸收，不考虑散射后传播 | 仅计算穿透率时误差可接受 |
| **准单能光子** | 光子能量在50-100keV范围内可调 | 支持线性插值获取中间能量材料参数 |

### 2.2 比尔-朗伯定律

光子穿过介质的衰减遵循指数规律：

$$I = I_0 \cdot e^{-\mu x}$$

其中：
- $\mu$: 线性衰减系数 (cm⁻¹)
- $x$: 路径长度 (cm)
- $I/I_0$: 穿透强度比

### 2.3 自由程采样

光子发生相互作用的距离 $s$ 服从**指数分布**：

$$P(s) = \mu \cdot e^{-\mu s}$$

**采样方法**（逆变换采样）：

```
s = -ln(ξ) / μ
```

其中 ξ ∈ (0,1) 是均匀随机数。

**物理意义**：
- 平均自由程 = 1/μ
- μ 越大，光子走不远就被吸收
- μ 越小，光子穿透距离越长

**能量相关衰减系数**：

材料参数文件提供50keV和100keV两种能量的μ值，通过**线性插值**计算任意能量（50-100keV范围内）的衰减系数：

```
μ(E) = μ₅₀ + (E - 50) / (100 - 50) × (μ₁₀₀ - μ₅₀)
```

---

## 3. 几何模型

### 3.1 坐标系定义

```
                 Z 轴（深度方向）
                 ↑
                 │
    光子源 ──────┼────────→ 探测器
    (z=-1)       │          (z=17.5)
                 │
    ─────────────┼────────────────────→ X 轴
                / 
               /   Y 轴（垂直纸面）
```

### 3.2 多层平板结构

```
层号  名称      起始z   厚度    μ值      说明
───────────────────────────────────────────────
0     皮肤      0.0     0.2cm   0.2     软组织
1     颅骨      0.2     0.8cm   0.5     致密骨
2     脑组织    1.0    16.0cm   0.18    脑实质
───────────────────────────────────────────────
探测器位置: z = 17.5cm
```

### 3.3 球体异物（可选）

球体用于模拟病变区域（如血块、肿瘤、金属碎片）：

```
球心: (cx, cy, cz)
半径: r
衰减系数: μ_sphere（不同于周围组织）
```

**光线-球相交检测**：

解二次方程求进入/离开距离：

$$t^2 + 2(\vec{D} \cdot \vec{O})t + (|\vec{O}|^2 - R^2) = 0$$

其中：
- $\vec{D}$: 光子方向向量（单位化）
- $\vec{O}$: 球心到光子起点的向量
- $R$: 球半径

判别式 $\Delta = b^2 - 4ac$：
- $\Delta < 0$: 不相交
- $\Delta \geq 0$: 相交，$t_{enter} = \frac{-b - \sqrt{\Delta}}{2}$, $t_{exit} = \frac{-b + \sqrt{\Delta}}{2}$

---

## 4. 源模型

### 4.1 点源（Point Source）

**物理特性**：
- 光线从一个点向四周发散
- 遵循平方反比定律：$I(r) = I_0 / (4\pi r^2)$
- 产生几何放大效应（中心亮、边缘暗）

**实现方法**：
在探测器张角范围内均匀采样角度，生成覆盖整个探测器的锥形束。

```cpp
// 计算探测器边界相对于源点的角度范围
dz = detector.z - source.z;
angle_x_min = atan((det_x_min - source.x) / dz);
angle_x_max = atan((det_x_max - source.x) / dz);
angle_y_min = atan((det_y_min - source.y) / dz);
angle_y_max = atan((det_y_max - source.y) / dz);

// 在角度空间均匀采样
angle_x = random(angle_x_min, angle_x_max);
angle_y = random(angle_y_min, angle_y_max);

// 从角度计算方向向量：tan(θ) = 对边/邻边
tan_x = tan(angle_x);
tan_y = tan(angle_y);

// 归一化方向向量
len = sqrt(tan_x² + tan_y² + 1);
dx = tan_x / len;
dy = tan_y / len;
dz = 1 / len;
```

**关键处理**：
- **源点到第一层传播**：光子在源点初始化后，先传播到几何模型第一层入口，避免遗漏源点到第一层的横向位移
- **归一化**：同时考虑x和y方向角度时，必须归一化确保方向向量为单位长度
- **边缘覆盖**：角度范围根据探测器尺寸自动计算，确保完整覆盖

### 4.2 平行束（Parallel Beam）

**物理特性**：
- 所有光子方向相同（通常为沿Z轴方向）
- 强度不随距离衰减
- 1:1 投影，无几何畸变

**实现方法**：
在探测器投影范围内均匀采样位置，确保全部光子都能到达探测器。

```cpp
// 计算探测器边界
det_x_min = detector.x - detector.width / 2;
det_x_max = detector.x + detector.width / 2;
det_y_min = detector.y - detector.height / 2;
det_y_max = detector.y + detector.height / 2;

// 在源平面上均匀分布（覆盖探测器投影范围）
x = random(det_x_min, det_x_max);
y = random(det_y_min, det_y_max);
initPhoton(p, x, y, source.z, source.energy);

// 方向固定（通常为沿Z轴方向）
p.dx = 0;
p.dy = 0;
p.dz = 1;
```

**关键处理**：
- **位置采样范围**：在探测器尺寸范围内均匀采样，确保光子能到达探测器
- **100%效率**：所有生成的光子都能到达探测器，无浪费
- **无畸变**：1:1投影，图像与物体尺寸相同

### 4.3 两种源的比较

| 特性 | 点源 | 平行束 |
|------|------|--------|
| **采样空间** | 角度空间均匀 | 探测器平面均匀 |
| **光线方向** | 发散（从源点出发） | 平行（固定方向） |
| **强度衰减** | 平方反比（中心亮边缘暗） | 无衰减 |
| **几何畸变** | 有（中心放大） | 无（1:1投影） |
| **探测器覆盖率** | 完整覆盖（锥形束） | 100%（所有光子到达） |
| **典型应用** | 临床CT/X光机（点源） | 理想模型/同步辐射 |

---

## 5. 算法设计

### 5.1 蒙特卡洛传输流程

```
对于每个光子:
    1. 初始化（位置、方向、能量）
    
    2. 传播到第一层入口（点源模式需要）
       - 处理源点到几何模型起点的距离
       - 计算横向位移：dx * (layer_start - source.z) / dz
    
    3. 逐层传输:
       a. 计算当前层的光学路径长度 = dz_layer / |dz|
       b. 采样自由程 s = -ln(ξ)/μ
       c. 比较: s < 层厚度 ?
          - 是: 被该层吸收，光子死亡
          - 否: 穿过该层，进入下一层
       d. 更新位置: p.x += dx * path_length
       
    4. 传播到探测器平面（最后一层到探测器的距离）
    
    5. 映射到探测器像素，累加计数
```

---

### 5.2 平板分层处理

**关键计算**（当前 z 坐标 `current_z`，层范围 `[z_start, z_end]`）：

```cpp
// 实际入射位置（可能在层内）
float enter_z = max(current_z, layer_start);

// 出射位置（层边界或探测器）
float exit_z = min(detector_z, layer_end);

// z方向投影长度
float dz = exit_z - enter_z;

// 实际物理路径（考虑角度）
float path_length = dz / fabsf(photon_dz);
```

### 5.3 球体异物处理

当存在球体时，路径分段计算：

```
路径分段示意:

平板区域 ──→ 球体边界(进入) ──→ 球体内部 ──→ 球体边界(离开) ──→ 平板区域
   μ₁              │                μ₂                 │               μ₃
            t_enter                     t_exit
            
总衰减 = exp(-μ₁·d₁ - μ₂·(t_exit-t_enter) - μ₃·d₃)
```

**算法步骤**：
1. 检测光线是否与球相交
2. 计算进入/离开距离 ($t_{enter}$, $t_{exit}$)
3. 路径分为三段：球外→球内→球外
4. 每段用对应的 μ 值计算衰减

---

## 6. 实现架构

### 6.1 代码结构

```
06_photon_sim/weiwei2027/
├── src/                         # 源代码
│   ├── photon_sim_nv.cu         # NVIDIA 版本
│   ├── photon_sim_iluvatar.cu   # Iluvatar 版本
│   ├── photon_sim_metax.maca    # MetaX 版本
│   ├── photon_sim_moore.mu      # Moore 版本
│   ├── photon_sim_cpu.cpp       # CPU 基准版本
│   └── utils.cpp                # 工具函数实现
├── include/                     # 头文件
│   ├── types.h                  # 数据结构定义
│   ├── utils.h                  # 工具函数接口
│   └── photon_sim.cuh           # 平台抽象层
├── data/                        # 测试数据
│   ├── geometry_1layer.txt      # 单层几何
│   ├── geometry_3layer.txt      # 三层几何
│   ├── geometry_3layer_sphere.txt  # 带球体几何
│   ├── materials.csv            # 材料参数
│   └── source_*.txt             # 源配置
├── scripts/                     # 可视化脚本
│   ├── visualize.py
│   ├── visualize_geometry_mpl.py
│   └── generate_report_figures.py
├── report/                      # 技术报告
│   ├── REPORT.md                # 主报告
│   └── figures/                 # 图表
├── Makefile                     # 多平台构建
└── CMakeLists.txt               # CMake构建配置
```

### 6.2 数据结构

**Region（几何区域）**：
```cpp
struct Region {
    RegionType type;        // LAYER 或 SPHERE
    char name[32];
    float thickness;        // 层厚度 / 球半径
    float x, y, z;          // 位置
    int material_id;        // 材料索引
    char material_name[32]; // 材料名称
};
```

**Source（源）**：
```cpp
struct Source {
    int type;               // 0=点源, 1=平行束
    float x, y, z;          // 位置
    float dx, dy, dz;       // 方向
    float energy;           // 能量(keV)
    int num_photons;        // 光子数
};
```

**Detector（探测器）**：
```cpp
struct Detector {
    float x, y, z;          // 位置
    float width, height;    // 尺寸(cm)
    int nx, ny;             // 像素分辨率
    float *pixels;          // 像素数据(GPU)
};
```

### 6.3 构建系统

**CMake配置要点**：
- CUDA标准：C++17
- 架构：sm_89 (RTX 4090), sm_80 (A100)
- 分离编译：启用 CUDA_SEPARABLE_COMPILATION
- 自动拷贝 data 目录到构建目录

**构建命令**（多平台支持）：
```bash
# NVIDIA平台
make PLATFORM=nvidia

# 其他平台
make PLATFORM=iluvatar   # 天數智芯
make PLATFORM=metax      # 沐曦
make PLATFORM=moore      # 摩尔线程

./photon_sim_nv      # GPU版本
./photon_sim_cpu     # CPU版本
```

---

## 7. 性能数据

### 7.1 基准测试结果

| 指标 | CPU (单线程) | GPU (RTX 4090) | 加速比 |
|------|-------------|----------------|--------|
| 检测率 | 2.76% | 2.76% | - |
| 处理速率 (1B光子) | 1.63×10⁷ photons/s | 4.88×10¹⁰ photons/s | **~2,994×** |
| 运行时间 (1B光子) | 61.4s | 0.0205s | - |

> 测试环境：CPU Intel Core i9-13900K (32核), GPU RTX 4090 24GB, CUDA 13.0

### 7.2 算法一致性验证

穿透率误差：0.01%（满足 < 0.1% 要求）

### 7.3 性能优化空间

- **共享内存**：缓存几何参数，减少全局内存访问
- **原子操作优化**：分块归约，减少竞争
- **角度发散模型**：朗伯分布/高斯分布采样

---

## 8. 注意事项与局限性

### 8.1 当前限制

1. **简化物理模型**：
   - 忽略散射（康普顿、瑞利）
   - 忽略折射
   - 单能光子（无能量谱）

2. **几何简化**：
   - 平板层垂直于Z轴
   - 未支持任意曲面

3. **性能**：
   - 探测器像素写入存在原子竞争
   - 未优化内存访问模式

### 8.2 使用建议

**适用场景**：
- 快速估算穿透剂量
- 教学演示蒙特卡洛方法
- 算法原型验证

**不适用场景**：
- 精确剂量计算（需考虑散射）
- 低能X射线(<30keV)
- 复杂几何体模

---

## 9. 参考文献

1.  https://doi.org/10.1088/0031-9155/51/14/R09
3. Cullen D E et al. 1997 *Tables and Graphs of Photon-Interaction Cross Sections*
4. Podgorsak E B. 2010 *Radiation Physics for Medical Physicists*

---

## 10. 附录：关键公式汇总

**比尔-朗伯定律**：$I = I_0 e^{-\mu x}$

**自由程采样**：$s = -\ln(\xi) / \mu$

**多层穿透率**：$T = \exp\left(-\sum_i \mu_i d_i\right)$

**光线-球相交判别式**：$\Delta = b^2 - 4ac$

**蒙特卡洛统计误差**：$\sigma \propto 1/\sqrt{N}$

---

*本文档记录了项目的设计决策和物理模型，供后续开发和报告撰写参考。*
