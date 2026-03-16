# 06 医学成像光子传输模拟

## 项目信息
- **选题**: 06 医学成像光子传输模拟（CUDA）
- **作者**: weiwei2027
- **时间**: 2025年冬季训练营
- **状态**: ✅ **已完成**（2026-03-15）

## 项目简介

本项目实现了一个 CUDA 加速的 X 射线光子传输模拟器，用于移动式头部 CT 成像系统。模拟光子穿过头部多层组织（皮肤、颅骨、脑组织）并处理球形异物（如血块），生成探测器投影图像。

### 核心特性
- ✅ 多层平板几何模型（皮肤、颅骨、脑组织）
- ✅ 球体异物支持（血块、肿瘤等病变模拟）
- ✅ 基于蒙特卡洛的光子传输模拟
- ✅ 自由程采样：L = -ln(ξ) / μ
- ✅ 点源（锥形束）和平行束两种模式
- ✅ 原子操作累加探测器像素
- ✅ 材料衰减系数 CSV 配置 + 线性插值
- ✅ GPU 加速与 CPU 基准版本对比
- ✅ 多平台支持（NVIDIA、Iluvatar、MetaX、Moore）
- ✅ Python 可视化脚本

## 目录结构

```
06_photon_sim/weiwei2027/
├── README.md                   # 本文件
├── PROJECT_REQUIREMENTS.md     # 项目要求说明
├── DESIGN.md                   # 设计文档（物理模型、算法）
├── WORKLIST.md                 # 工作清单和进度跟踪
├── PORTING_PLAN.md             # 多平台移植计划
├── Makefile                    # 多平台 Makefile
├── CMakeLists.txt              # CMake 构建配置
├── include/                    # 头文件
│   ├── types.h                 # 数据类型定义
│   ├── utils.h                 # 工具函数接口
│   └── photon_sim.cuh          # 平台抽象头文件
├── src/                        # 源代码
│   ├── photon_sim_nv.cu        # NVIDIA GPU 版本
│   ├── photon_sim_iluvatar.cu  # Iluvatar CoreX 版本
│   ├── photon_sim_metax.maca   # MetaX C500 版本
│   ├── photon_sim_moore.mu     # Moore MTT S5000 版本
│   ├── photon_sim_cpu.cpp      # CPU 基准版本
│   └── utils.cpp               # 工具函数实现
├── data/                       # 测试数据
│   ├── geometry_1layer.txt     # 单层测试
│   ├── geometry_3layer.txt     # 三层测试
│   ├── geometry_3layer_sphere.txt  # 带球体测试
│   ├── materials.csv           # 材料参数
│   └── source_*.txt            # 源参数配置
├── output/                     # 输出目录
├── scripts/                    # 辅助脚本
│   ├── visualize.py            # 图像可视化
│   ├── visualize_geometry.py   # 几何可视化
│   └── compare_sources.py      # 对比分析
├── tests/                      # 测试代码
│   └── test_parser.cpp         # 解析模块测试
└── report/                     # 总结报告
    └── report.md               # 项目报告
```

## 编译与运行

### 环境要求
- CUDA Toolkit 11.0+ (NVIDIA平台)
- GPU Compute Capability 6.1+ (RTX系列)
- Python 3.x + NumPy + Matplotlib（可视化）

### 编译（Makefile）

```bash
# NVIDIA 平台（默认）
make
make PLATFORM=nvidia

# Iluvatar 平台（天数智芯）
make PLATFORM=iluvatar

# MetaX 平台（沐曦）
make PLATFORM=metax

# Moore 平台（摩尔线程）
make PLATFORM=moore

# 清理
make clean

# 查看帮助
make help
```

### 编译（CMake）

```bash
mkdir -p build && cd build
cmake -DPLATFORM=nvidia ..
make
```

### 运行

```bash
# 基础运行（默认参数）
./photon_sim_nv

# 指定配置文件
./photon_sim_nv -g data/geometry_3layer.txt \
                -m data/materials.csv \
                -s data/source_point_10m.txt \
                -o output/

# 带球体异物
./photon_sim_nv -g data/geometry_3layer_sphere.txt \
                -m data/materials.csv \
                -s data/source_point_10m.txt \
                -o output/

# 指定 CPU 基线时间计算加速比
./photon_sim_nv ... -c 100.0
```

### 测试验证

```bash
# 单层测试（理论透过率 ~81.87%）
make test_1layer

# 三层测试（理论透过率 ~3.6%）
make test_sphere

# 解析模块单元测试
cd tests && make test_parser && ./test_parser
```

### 可视化结果

```bash
cd scripts

# 可视化图像
python3 visualize.py ../output/image.bin --info ../output/image_info.txt

# 保存为图片
python3 visualize.py ../output/image.bin --info ../output/image_info.txt --save ../output/result.png

# 3D 几何可视化
python3 visualize_geometry_mpl.py ../data/geometry_3layer_sphere.txt
```

## 输入文件格式

### 几何模型文件

```
layer skin 0.2 0.0 0.0 0.0 tissue_skin
layer skull 0.8 0.0 0.0 0.2 tissue_bone
layer brain 16.0 0.0 0.0 1.0 tissue_brain
sphere hematoma 2.0 0.0 0.0 5.0 tissue_blood
```

**格式**: `type name thickness x y z material`
- `type`: `layer` 或 `sphere`
- `thickness`: 层厚度或球体半径（cm）
- `x, y, z`: 起始坐标（层）或球心（球体）
- `material`: 材料名称

### 物理参数文件 (materials.csv)

```csv
material,energy_keV,mu
tissue_skin,50,0.2
tissue_skin,100,0.15
tissue_bone,50,0.5
tissue_brain,50,0.18
tissue_blood,50,0.22
```

### 源参数文件

```yaml
source_type = point                    # point 或 parallel
position = 0.0 0.0 -1.0                # 源位置
direction = 0.0 0.0 1.0                # 主方向
energy = 50.0                          # 光子能量 (keV)
num_photons = 10000000                 # 光子数
detector_position = 0.0 0.0 17.5       # 探测器位置
detector_size = 20.0 20.0              # 探测器尺寸 (cm)
detector_pixels = 1024 1024            # 像素分辨率
```

## 输出文件

| 文件 | 说明 |
|------|------|
| `image.bin` | 探测器图像（二进制浮点数组） |
| `image_info.txt` | 图像元数据（尺寸、数据类型） |
| `performance.log` | 性能日志（时间、吞吐量、加速比） |
| `result.png` | 可视化图像（需运行可视化脚本） |

## 算法概述

### 光子传输过程

1. **初始化**: 根据源类型生成光子
   - 点源：锥形束，覆盖整个探测器
   - 平行束：均匀覆盖探测器范围

2. **传播到第一层**: 计算源点到几何模型入口的路径

3. **逐层传输**:
   - 计算光线-层边界相交
   - 采样自由程：L = -ln(ξ) / μ
   - 若自由程 < 路径长度：光子被吸收
   - 否则：光子穿过该层
   - 对球体：分段计算路径（层→球→层）

4. **到达探测器**: 映射到像素坐标，原子累加

### CUDA 并行策略

- **线程配置**: 256 blocks × 256 threads = 65,536 线程
- **任务分配**: 每个线程处理多个光子（grid-stride）
- **随机数**: cuRAND，每个线程独立状态
- **几何缓存**: 常量内存存储区域参数
- **探测器**: `atomicAdd` 避免竞争条件

## 性能结果

### 测试环境 1: NVIDIA A100 (远程服务器)

| 组件 | 规格 |
|------|------|
| GPU | NVIDIA A100-SXM4-80GB × 8 |
| CPU | Intel Xeon @ 2.90GHz (80 cores) |
| 内存 | 1.4 TiB |
| CUDA | 12.8 |

### A100 性能测试 (10亿光子)

| 配置 | GPU 时间 | GPU 速率 | CPU 时间 | **加速比** |
|------|----------|----------|----------|-----------|
| 单层 | 0.024s | **4.13×10¹⁰ p/s** | 107.8s | **4,455×** |
| 三层 | 0.036s | **2.79×10¹⁰ p/s** | 135.9s | **3,795×** |
| 三层+球体 | 0.067s | **1.50×10¹⁰ p/s** | 160.6s | **2,412×** |
| 平行束 | 0.033s | **3.04×10¹⁰ p/s** | 100.9s | **3,067×** |

### MetaX C500 性能测试 (10亿光子)

| 配置 | GPU 时间 | GPU 速率 | CPU 时间 | **加速比** |
|------|----------|----------|----------|-----------|
| 单层 | 0.143s | **7.01×10⁹ p/s** | 93.2s | **653×** |
| 三层 | 0.135s | **7.43×10⁹ p/s** | 89.1s | **662×** |
| 三层+球体 | 0.224s | **4.47×10⁹ p/s** | 118.9s | **531×** |
| 平行束 | 0.115s | **8.72×10⁹ p/s** | 67.7s | **590×** |

### 测试环境 3: NVIDIA RTX 4090 (本地)

| 配置 | 光子数 | 时间 | 速率 |
|------|--------|------|------|
| 三层 | 10⁷ | 0.24 ms | **4.23×10¹⁰ p/s** |
| 三层+球体 | 10⁹ | ~24 ms | **~4.2×10¹⁰ p/s** |

### GPU vs CPU 加速比对比

| 平台 | 处理速率 | 加速比 |
|------|----------|--------|
| CPU (单线程) | ~2.2×10⁶ p/s | 1× |
| GPU (GTX 1060) | ~2.6×10⁹ p/s | **~1,200×** |
| GPU (MetaX C500) | ~8.7×10⁹ p/s | **~653×** |
| GPU (RTX 4090) | ~4.2×10¹⁰ p/s | **~19,000×** |
| GPU (A100) | ~4.1×10¹⁰ p/s | **~4,500×** |

### 验证结果
- ✅ 单层穿透率: 实测 80.5% ≈ 理论值
- ✅ 三层穿透率: 实测 2.76%（符合预期）
- ✅ 球体阴影: 带球体后降至 2.69%
- ✅ CPU/GPU 一致性: 误差 < 0.1%

## 优化记录

| 版本 | 优化内容 | 性能 | 状态 |
|------|----------|------|------|
| V0 | CPU 基准 | ~2×10⁶ p/s | ✅ 参考 |
| V1 | 基础 CUDA | ~1×10⁹ p/s | ✅ |
| V2 | 常量内存缓存 | ~2.6×10⁹ p/s | ✅ |
| V3 | 球体相交检测 | ~1.5×10⁹ p/s | ✅ |
| V4 | NCU 分析优化 | - | ✅ 已完成 |
| V5 | A100 远程测试 | 4.13×10¹⁰ p/s | ✅ 已完成 |
| V6 | MetaX C500 测试 | 8.72×10⁹ p/s | ✅ 已完成 |

## 多平台支持

| 平台 | 厂商 | 编译器 | 状态 |
|------|------|--------|------|
| NVIDIA | NVIDIA | nvcc | ✅ 已验证 (A100) |
| Iluvatar | 天数智芯 | clang++ | ⏳ 代码完成，待测试 |
| MetaX | 沐曦 | mxcc | ✅ 已验证 (C500) |
| Moore | 摩尔线程 | mcc | ⏳ 代码完成，待测试 |

## 已知限制

1. **简化物理模型**：
   - 忽略散射（康普顿、瑞利）
   - 忽略折射
   - 单能光子

2. **几何简化**：
   - 平板层垂直于Z轴
   - 不支持任意曲面

3. **性能**：
   - 探测器像素写入存在原子竞争

## 参考资料

- DESIGN.md - 详细设计文档
- WORKLIST.md - 工作进度跟踪
- PORTING_PLAN.md - 多平台移植说明
- report/report.md - 项目总结报告

---

**项目时间**: 2025.2.10 - 2025.3.16  
**提交地址**: Learning-CUDA 2025-winter-project 分支 /06_photon_sim/weiwei2027/

---

## 远程平台测试

### A100 服务器测试

```bash
# 1. 一键测试（完整测试，约10分钟）
./run_remote_tests.sh

# 2. 快速测试（单个测试，约2分钟）
./quick_remote_test.sh 3layer

# 3. 手动测试
# 上传
rsync -avz -e "ssh -p 2222 -i ~/workspace/InfiniTensor2025/ssh/nvidia/key.id" \
    ./ weiwei@8.145.51.96:/home/weiwei/weiwei2027/

# 远程构建和测试
ssh -p 2222 -i ~/workspace/InfiniTensor2025/ssh/nvidia/key.id weiwei@8.145.51.96
cd /home/weiwei/weiwei2027
make PLATFORM=nvidia
./photon_sim_nv -g data/geometry_3layer.txt -m data/materials.csv -s data/source_point_10m.txt -o output/
```

详细说明见: [remote_test_guide.md](remote_test_guide.md)
